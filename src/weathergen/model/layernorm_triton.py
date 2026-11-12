# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: N803  (Triton convention: pointer and constexpr arguments in capitals)

"""Config option `layernorm_impl: triton`: multi-row LayerNorm kernels (experiment O7).

PyTorch's `vectorized_layer_norm_kernel` runs one thread block per row. For the small
normalised sizes of the q/k norms (D = 64 .. 128, 0.4 .. 3.4 M rows per launch) it reaches
0.14 .. 0.4 TB/s on a GH200 (HBM peak about 4 TB/s): it is bound by block scheduling, not by
memory. The kernels here process a tile of BLOCK_M rows per program, so a launch is bandwidth
bound.

Semantics: statistics in float32 (mean, then variance of the centred values, as PyTorch), the
output rounded once to the output dtype. Not bitwise equal to PyTorch (different summation
order), equal within the rounding of the output dtype. Input dtype and output dtype may differ
(for example a bfloat16 input with a float32 output replaces `x.float()` + `layer_norm`).

Optional residual add: y = LN(x + residual) and the sum itself are produced by one kernel
(`residual=`). The sum is rounded to `residual_out_dtype` before it is normalised, as the
eager add followed by the norm would do. No module of the model uses it yet: the add sits at the
end of the previous block, the norm at the start of the next one (see o7-layernorm-kernel.md).

Backward: dx from the saved input, mean and rstd; dweight / dbias (if affine) from per-program
partial sums, reduced with one `sum(0)`.

`layer_norm_reference` and `layer_norm_backward_reference` implement the same arithmetic in
PyTorch (any device); `FastLayerNormFn` uses them when Triton or CUDA is not available, so the
CPU tests exercise the same autograd code and the same formulas as the kernels.
"""

import torch

from weathergen.model.fused_kernels import HAVE_TRITON, triton_ok

if HAVE_TRITON:
    import triton
    import triton.language as tl

# (elements per tile, num_warps) used by `_launch_params`; the benchmark sweeps these.
TILE_ELEMS = 8192
NUM_WARPS = 8
CONFIG_OVERRIDE: dict[int, tuple[int, int]] = {}  # D -> (BLOCK_M, num_warps), for tuning


####################################################################################################
# PyTorch reference (same arithmetic as the kernels)


def layer_norm_reference(x, weight, bias, eps, out_dtype, residual=None, residual_out_dtype=None):
    """Returns (y, mean, rstd, s) with s = rounded x + residual (or x if residual is None)."""
    d = x.shape[-1]
    xf = x.reshape(-1, d).float()
    s = None
    if residual is not None:
        xf = xf + residual.reshape(-1, d).float()
        s = xf.to(residual_out_dtype)
        xf = s.float()
    mean = xf.mean(-1)
    xc = xf - mean[:, None]
    rstd = torch.rsqrt((xc * xc).mean(-1) + eps)
    y = xc * rstd[:, None]
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(out_dtype).reshape(x.shape), mean, rstd, s


def layer_norm_backward_reference(dy, x, mean, rstd, weight, has_bias, dres=None):
    """Returns (dx, dweight, dbias) for the saved input x (2D view), float32 math."""
    d = x.shape[-1]
    xf = x.reshape(-1, d).float()
    dyf = dy.reshape(-1, d).float()
    xhat = (xf - mean[:, None]) * rstd[:, None]
    wdy = dyf * weight.float() if weight is not None else dyf
    c1 = (xhat * wdy).mean(-1, keepdim=True)
    c2 = wdy.mean(-1, keepdim=True)
    dx = (wdy - (xhat * c1 + c2)) * rstd[:, None]
    if dres is not None:
        dx = dx + dres.reshape(-1, d).float()
    dw = (dyf * xhat).sum(0) if weight is not None else None
    db = dyf.sum(0) if has_bias else None
    return dx, dw, db


####################################################################################################
# Triton kernels

if HAVE_TRITON:

    @triton.jit
    def _ln_fwd_kernel(
        Y,
        X,
        W,
        B,
        RES,
        RES_OUT,
        MEAN,
        RSTD,
        n_rows,
        D,
        stride_x,
        eps,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        HAS_RES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        cm = cols < D
        m = (rows < n_rows)[:, None] & cm[None, :]
        ox = rows[:, None] * stride_x + cols[None, :]
        o = rows[:, None] * D + cols[None, :]
        x = tl.load(X + ox, mask=m, other=0.0).to(tl.float32)
        if HAS_RES:
            r = tl.load(RES + o, mask=m, other=0.0).to(tl.float32)
            s = (x + r).to(RES_OUT.dtype.element_ty)
            tl.store(RES_OUT + o, s, mask=m)
            x = s.to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = tl.where(m, x - mean[:, None], 0.0)
        var = tl.sum(xc * xc, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        y = xc * rstd[:, None]
        if HAS_W:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
            y = y * w[None, :]
        if HAS_B:
            b = tl.load(B + cols, mask=cm, other=0.0).to(tl.float32)
            y = y + b[None, :]
        tl.store(Y + o, y.to(Y.dtype.element_ty), mask=m)
        rm = rows < n_rows
        tl.store(MEAN + rows, mean, mask=rm)
        tl.store(RSTD + rows, rstd, mask=rm)

    @triton.jit
    def _ln_bwd_kernel(
        DX,
        DY,
        X,
        W,
        MEAN,
        RSTD,
        DRES,
        DW,
        DB,
        n_rows,
        D,
        stride_x,
        n_tiles,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        HAS_DRES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        cols = tl.arange(0, BLOCK_D)
        cm = cols < D
        if HAS_W:
            w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
        dw_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        db_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for tile in range(pid, n_tiles, nprog):
            rows = (tile * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
            rm = rows < n_rows
            m = rm[:, None] & cm[None, :]
            ox = rows[:, None] * stride_x + cols[None, :]
            o = rows[:, None] * D + cols[None, :]
            x = tl.load(X + ox, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(DY + o, mask=m, other=0.0).to(tl.float32)
            mean = tl.load(MEAN + rows, mask=rm, other=0.0)
            rstd = tl.load(RSTD + rows, mask=rm, other=0.0)
            xhat = tl.where(m, (x - mean[:, None]) * rstd[:, None], 0.0)
            if HAS_W:
                wdy = dy * w[None, :]
            else:
                wdy = dy
            c1 = tl.sum(xhat * wdy, axis=1) / D
            c2 = tl.sum(wdy, axis=1) / D
            dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            if HAS_DRES:
                dx = dx + tl.load(DRES + o, mask=m, other=0.0).to(tl.float32)
            tl.store(DX + o, dx.to(DX.dtype.element_ty), mask=m)
            if HAS_W:
                dw_acc += tl.sum(dy * xhat, axis=0)
            if HAS_B:
                db_acc += tl.sum(dy, axis=0)
        if HAS_W:
            tl.store(DW + pid * D + cols, dw_acc, mask=cm)
        if HAS_B:
            tl.store(DB + pid * D + cols, db_acc, mask=cm)


_SM_COUNT: dict[int, int] = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _launch_params(d: int) -> tuple[int, int, int]:
    """(BLOCK_M, BLOCK_D, num_warps) for normalised size d."""
    block_d = triton.next_power_of_2(d)
    if d in CONFIG_OVERRIDE:
        bm, nw = CONFIG_OVERRIDE[d]
        return bm, block_d, nw
    return max(1, TILE_ELEMS // block_d), block_d, NUM_WARPS


def triton_supported(x: torch.Tensor, d: int) -> bool:
    return triton_ok(x) and 1 <= d <= 8192 and x.dim() >= 1 and x.shape[-1] == d


def _rows_2d(x: torch.Tensor):
    """(2D view, row stride) of x without a copy when the rows are uniformly strided."""
    d = x.shape[-1]
    if x.stride(-1) != 1:
        x = x.contiguous()
    if x.dim() == 1:
        return x.reshape(1, d), d
    try:
        v = x.view(-1, d)  # raises unless the leading dims flatten with a uniform stride
    except RuntimeError:
        x = x.contiguous()
        v = x.view(-1, d)
    return v, v.stride(0)


def _fwd_triton(x, weight, bias, eps, out_dtype, residual, residual_out_dtype):
    d = x.shape[-1]
    x2, stride_x = _rows_2d(x)
    n_rows = x2.shape[0]
    y = torch.empty((n_rows, d), dtype=out_dtype, device=x.device)
    mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    res2 = res_out = None
    if residual is not None:
        res2 = residual.reshape(n_rows, d).contiguous()
        res_out = torch.empty((n_rows, d), dtype=residual_out_dtype, device=x.device)
    bm, bd, nw = _launch_params(d)
    with torch.cuda.device(x.device.index):
        _ln_fwd_kernel[(triton.cdiv(n_rows, bm),)](
            y,
            x2,
            weight,
            bias,
            res2,
            res_out,
            mean,
            rstd,
            n_rows,
            d,
            stride_x,
            eps,
            HAS_W=weight is not None,
            HAS_B=bias is not None,
            HAS_RES=residual is not None,
            BLOCK_M=bm,
            BLOCK_D=bd,
            num_warps=nw,
        )
    return y.view(x.shape), mean, rstd, (None if res_out is None else res_out.view(x.shape))


def _bwd_triton(dy, x, mean, rstd, weight, has_bias, dres):
    d = x.shape[-1]
    x2, stride_x = _rows_2d(x)
    n_rows = x2.shape[0]
    dy2 = dy.reshape(n_rows, d).contiguous()
    dx = torch.empty((n_rows, d), dtype=x.dtype, device=x.device)
    bm, bd, nw = _launch_params(d)
    n_tiles = triton.cdiv(n_rows, bm)
    affine = weight is not None or has_bias
    grid = min(n_tiles, 8 * _sm_count(x.device)) if affine else n_tiles
    dw = torch.empty((grid, d), dtype=torch.float32, device=x.device) if weight is not None else None
    db = torch.empty((grid, d), dtype=torch.float32, device=x.device) if has_bias else None
    dres2 = dres.reshape(n_rows, d).contiguous() if dres is not None else None
    with torch.cuda.device(x.device.index):
        _ln_bwd_kernel[(grid,)](
            dx,
            dy2,
            x2,
            weight,
            mean,
            rstd,
            dres2,
            dw,
            db,
            n_rows,
            d,
            stride_x,
            n_tiles,
            HAS_W=weight is not None,
            HAS_B=has_bias,
            HAS_DRES=dres is not None,
            BLOCK_M=bm,
            BLOCK_D=bd,
            num_warps=nw,
        )
    return (
        dx.view(x.shape),
        None if dw is None else dw.sum(0),
        None if db is None else db.sum(0),
    )


####################################################################################################
# autograd


class FastLayerNormFn(torch.autograd.Function):
    """LayerNorm over the last dim, optional fused residual add (see module doc).

    forward(x, weight, bias, eps, out_dtype, residual, residual_out_dtype)
    returns y, or (y, x + residual) when `residual` is given.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps, out_dtype, residual=None, residual_out_dtype=None):
        d = x.shape[-1]
        if triton_supported(x, d):
            y, mean, rstd, s = _fwd_triton(x, weight, bias, eps, out_dtype, residual, residual_out_dtype)
        else:
            y, mean, rstd, s = layer_norm_reference(
                x, weight, bias, eps, out_dtype, residual, residual_out_dtype
            )
        has_res = residual is not None
        saved = s if has_res else x
        ctx.save_for_backward(saved, mean, rstd, weight)
        ctx.has_bias = bias is not None
        ctx.has_res = has_res
        ctx.x_dtype = x.dtype
        ctx.res_dtype = residual.dtype if has_res else None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.set_materialize_grads(False)
        return (y, s) if has_res else y

    @staticmethod
    def backward(ctx, dy, dres_out=None):
        saved, mean, rstd, weight = ctx.saved_tensors
        d = saved.shape[-1]
        if dy is None:  # only the residual sum was used downstream
            dy = torch.zeros_like(saved, dtype=torch.float32)
        if triton_supported(saved, d):
            dx, dw, db = _bwd_triton(dy, saved, mean, rstd, weight, ctx.has_bias, dres_out)
        else:
            dx, dw, db = layer_norm_backward_reference(
                dy, saved, mean, rstd, weight, ctx.has_bias, dres_out
            )
        dx_x = dx.to(ctx.x_dtype)
        dx_res = dx.to(ctx.res_dtype) if ctx.has_res else None
        dw = None if dw is None else dw.to(weight.dtype)
        db = None if db is None else db.to(ctx.bias_dtype)
        return dx_x, dw, db, None, None, dx_res, None


def fast_layer_norm(
    x, weight=None, bias=None, eps=1e-5, out_dtype=None, residual=None, residual_out_dtype=None
):
    """LayerNorm over the last dim of x. `out_dtype` defaults to x.dtype.

    With `residual`, returns (LN(x + residual), x + residual); the sum is in
    `residual_out_dtype` (default: residual.dtype).
    """
    out_dtype = out_dtype or x.dtype
    if residual is not None:
        residual_out_dtype = residual_out_dtype or residual.dtype
    return FastLayerNormFn.apply(x, weight, bias, eps, out_dtype, residual, residual_out_dtype)


####################################################################################################
# modules


class TritonLayerNorm(torch.nn.LayerNorm):
    """`torch.nn.LayerNorm` with the multi-row kernel; the float32 sites of `layernorm_bf16`.

    Matches what `torch.nn.LayerNorm` does under autocast (float32 in, float32 out) without the
    separate `x.float()` kernel; without autocast the output has the dtype of x.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.normalized_shape) != 1 or not x.is_cuda:
            return super().forward(x)
        autocast = torch.is_autocast_enabled(x.device.type)
        out_dtype = torch.float32 if autocast else x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            return fast_layer_norm(x, self.weight, self.bias, self.eps, out_dtype)


def use_triton_layernorms(model: torch.nn.Module) -> dict[str, int]:
    """Switch the LayerNorms of `model` to the multi-row kernels, in place (before sharding).

    `AutocastDtypeLayerNorm` instances get `impl = "triton"`; remaining plain
    `torch.nn.LayerNorm` instances become `TritonLayerNorm`.
    """
    from weathergen.model.norms import AutocastDtypeLayerNorm

    counts = {"autocast_dtype": 0, "float32": 0}
    for module in model.modules():
        if type(module) is AutocastDtypeLayerNorm:
            module.impl = "triton"
            counts["autocast_dtype"] += 1
        elif type(module) is torch.nn.LayerNorm:
            module.__class__ = TritonLayerNorm
            counts["float32"] += 1
    return counts
