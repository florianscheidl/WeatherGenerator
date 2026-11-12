# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: N803  (Triton convention: pointer and constexpr arguments in capitals)

"""Config option `qk_norm: rmsnorm` (experiment O9): RMSNorm without weight for the q/k norms.

y = x / sqrt(mean(x^2) + eps), no mean centring, no parameters. `QKRMSNorm` replaces the
non-affine `lnorm_q` / `lnorm_k` LayerNorms of the attention blocks (`use_qk_rmsnorm`). Same
norm of the output as LayerNorm without affine (||y||^2 = D), so the bound on the attention
logits is unchanged; the output keeps the mean component.

`qk_norm_impl: torch` runs `F.rms_norm` (PyTorch's CUDA kernel, one block per row, the same
kernel family as `F.layer_norm`); `qk_norm_impl: triton` runs the multi-row kernels below, as
in `layernorm_triton.py` but without the mean pass. This file does not import
`layernorm_triton.py`, so both can change independently.

Dtypes: statistics in float32, output rounded once. Under autocast the module casts its input
to the autocast dtype and returns that dtype, whatever the input dtype was. That keeps the
tensors saved in the forward and in the recompute of an FSDP2-aligned checkpoint identical
(FSDP2 skips the cast of the unit inputs there; see `AutocastDtypeLayerNorm`).

`rms_norm_reference` / `rms_norm_backward_reference` do the same arithmetic in PyTorch; the
autograd function falls back to them without Triton or CUDA, so CPU tests run the same code.
"""

import torch
import torch.nn.functional as F

from weathergen.model.fused_kernels import HAVE_TRITON, triton_ok

if HAVE_TRITON:
    import triton
    import triton.language as tl

TILE_ELEMS = 8192
NUM_WARPS = 8
CONFIG_OVERRIDE: dict[int, tuple[int, int]] = {}  # D -> (BLOCK_M, num_warps), for tuning


####################################################################################################
# PyTorch reference


def rms_norm_reference(x, eps, out_dtype):
    """Returns (y, rstd); float32 math, one rounding to out_dtype."""
    d = x.shape[-1]
    xf = x.reshape(-1, d).float()
    rstd = torch.rsqrt((xf * xf).mean(-1) + eps)
    y = (xf * rstd[:, None]).to(out_dtype)
    return y.reshape(x.shape), rstd


def rms_norm_backward_reference(dy, x, rstd):
    """dx (float32, 2D) for the saved input x."""
    d = x.shape[-1]
    xf = x.reshape(-1, d).float()
    dyf = dy.reshape(-1, d).float()
    xhat = xf * rstd[:, None]
    c = (xhat * dyf).mean(-1, keepdim=True)
    return (dyf - xhat * c) * rstd[:, None]


####################################################################################################
# Triton kernels

if HAVE_TRITON:

    @triton.jit
    def _rms_fwd_kernel(
        Y,
        X,
        RSTD,
        n_rows,
        D,
        stride_x,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        rm = rows < n_rows
        m = rm[:, None] & (cols < D)[None, :]
        x = tl.load(X + rows[:, None] * stride_x + cols[None, :], mask=m, other=0.0)
        x = x.to(tl.float32)
        ms = tl.sum(x * x, axis=1) / D
        rstd = 1.0 / tl.sqrt(ms + eps)
        y = x * rstd[:, None]
        tl.store(Y + rows[:, None] * D + cols[None, :], y.to(Y.dtype.element_ty), mask=m)
        tl.store(RSTD + rows, rstd, mask=rm)

    @triton.jit
    def _rms_bwd_kernel(
        DX,
        DY,
        X,
        RSTD,
        n_rows,
        D,
        stride_x,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        rm = rows < n_rows
        m = rm[:, None] & (cols < D)[None, :]
        o = rows[:, None] * D + cols[None, :]
        x = tl.load(X + rows[:, None] * stride_x + cols[None, :], mask=m, other=0.0)
        x = x.to(tl.float32)
        dy = tl.load(DY + o, mask=m, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + rows, mask=rm, other=0.0)
        xhat = x * rstd[:, None]
        c = tl.sum(xhat * dy, axis=1) / D
        dx = (dy - xhat * c[:, None]) * rstd[:, None]
        tl.store(DX + o, dx.to(DX.dtype.element_ty), mask=m)


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
        v = x.view(-1, d)
    except RuntimeError:
        x = x.contiguous()
        v = x.view(-1, d)
    return v, v.stride(0)


def _fwd_triton(x, eps, out_dtype):
    d = x.shape[-1]
    x2, stride_x = _rows_2d(x)
    n_rows = x2.shape[0]
    y = torch.empty((n_rows, d), dtype=out_dtype, device=x.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    bm, bd, nw = _launch_params(d)
    with torch.cuda.device(x.device.index):
        _rms_fwd_kernel[(triton.cdiv(n_rows, bm),)](
            y, x2, rstd, n_rows, d, stride_x, eps, BLOCK_M=bm, BLOCK_D=bd, num_warps=nw
        )
    return y.view(x.shape), rstd


def _bwd_triton(dy, x, rstd):
    d = x.shape[-1]
    x2, stride_x = _rows_2d(x)
    n_rows = x2.shape[0]
    dy2 = dy.reshape(n_rows, d).contiguous()
    dx = torch.empty((n_rows, d), dtype=x.dtype, device=x.device)
    bm, bd, nw = _launch_params(d)
    with torch.cuda.device(x.device.index):
        _rms_bwd_kernel[(triton.cdiv(n_rows, bm),)](
            dx, dy2, x2, rstd, n_rows, d, stride_x, BLOCK_M=bm, BLOCK_D=bd, num_warps=nw
        )
    return dx.view(x.shape)


class FastRMSNormFn(torch.autograd.Function):
    """RMSNorm without weight over the last dim; saves x and rstd (float32, one per row)."""

    @staticmethod
    def forward(ctx, x, eps, out_dtype):
        d = x.shape[-1]
        if triton_supported(x, d):
            y, rstd = _fwd_triton(x, eps, out_dtype)
        else:
            y, rstd = rms_norm_reference(x, eps, out_dtype)
        ctx.save_for_backward(x, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, rstd = ctx.saved_tensors
        if triton_supported(x, x.shape[-1]):
            dx = _bwd_triton(dy, x, rstd)
        else:
            dx = rms_norm_backward_reference(dy, x, rstd).to(x.dtype).reshape(x.shape)
        return dx, None, None


def fast_rms_norm(x, eps=1e-5, out_dtype=None):
    """RMSNorm (no weight) over the last dim of x; `out_dtype` defaults to x.dtype."""
    return FastRMSNormFn.apply(x, eps, out_dtype or x.dtype)


####################################################################################################
# module


class QKRMSNorm(torch.nn.Module):
    """Parameter-free RMSNorm for `lnorm_q` / `lnorm_k` (replaces a non-affine LayerNorm).

    Under autocast the output has the autocast dtype (the input is cast to it first); without
    autocast it has the dtype of the input. `impl`: "torch" (`F.rms_norm`) or "triton".
    """

    def __init__(self, dim: int, eps: float = 1e-5, impl: str = "torch"):
        super().__init__()
        assert impl in ("torch", "triton"), impl
        self.dim = dim
        self.eps = eps
        self.impl = impl

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, impl={self.impl}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
            x = x.to(dtype)
        else:
            dtype = x.dtype
        with torch.autocast(device_type=device_type, enabled=False):
            if self.impl == "triton" and x.is_cuda:
                return fast_rms_norm(x, self.eps, dtype)
            return F.rms_norm(x, (self.dim,), None, self.eps)


def use_qk_rmsnorm(model: torch.nn.Module, impl: str = "torch") -> int:
    """Replace the non-affine LayerNorm `lnorm_q` / `lnorm_k` of `model` by `QKRMSNorm`, in place.

    Call on the built model before the `layernorm_bf16` / `layernorm_impl` passes and before
    sharding. `with_qk_lnorm=False` blocks (Identity) and affine q/k norms
    (`qk_norm_type: RMSNorm`, which has a weight) are left alone. Returns the number replaced.
    """
    n = 0
    for module in list(model.modules()):
        for name in ("lnorm_q", "lnorm_k"):
            child = getattr(module, name, None)
            if isinstance(child, torch.nn.LayerNorm) and not child.elementwise_affine:
                (dim,) = child.normalized_shape
                setattr(module, name, QKRMSNorm(dim, child.eps, impl))
                n += 1
    return n


def centre_qk_projections(model: torch.nn.Module) -> int:
    """Make a LayerNorm-trained model exact under `qk_norm: rmsnorm`, in place.

    LN(W x) = RMS(P W x) with P = I - 11^T / D per head, so subtracting the per-head mean of
    the output rows of `proj_heads_q` / `proj_heads_k` (no bias in these Linears) gives the
    same q and k (up to bf16 rounding) after the RMSNorm. Returns the number of Linears changed.
    """
    n = 0
    for module in model.modules():
        if not isinstance(getattr(module, "lnorm_q", None), QKRMSNorm):
            continue
        for lin_name in ("proj_heads_q", "proj_heads_k"):
            lin = getattr(module, lin_name, None)
            if not isinstance(lin, torch.nn.Linear) or lin.bias is not None:
                continue
            dh = getattr(module, "dim_head_proj", None)
            if dh is None or lin.out_features % dh:
                continue
            with torch.no_grad():
                w = lin.weight.view(lin.out_features // dh, dh, lin.in_features)
                w.sub_(w.mean(dim=1, keepdim=True))
            n += 1
    return n
