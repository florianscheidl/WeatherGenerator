# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: N803  (Triton convention: pointer and constexpr arguments in capitals)

"""Fused elementwise kernels for 2D RoPE and the AdaLN modulation (experiment O5).

Both replace chains of eager elementwise kernels with one kernel per tensor and pass, and
reproduce the eager rounding exactly (`exact=True`): every intermediate that eager rounds to
bfloat16 is rounded to bfloat16 here too, every float32 intermediate stays a separate float32
operation (Triton launches with `enable_fp_fusion=False`, so no mul+add is contracted to an
FMA). The results are then bitwise equal to the eager ops on the same device.

Each op has a PyTorch reference (`*_reference`) that implements the same arithmetic with
explicit casts; it runs on any device and is what the CPU tests compare with the eager ops.
The Triton kernels implement the same arithmetic and are compared on the GPU
(`fused_kernels_test.py`, CUDA-only tests).

RoPE (half-split / GPT-NeoX layout, as `apply_rotary_pos_emb`): for x = [x0, x1] (halves of
the head dim) and per-position tables c, s of half width (the eager tables are [c, c] and
[s, s]):

    eager:   out = bf16(x * [c, c]) + bf16(cat(-x1, x0) * [s, s])
    here:    o0  = bf16(bf16(x0 c) - bf16(x1 s)),  o1 = bf16(bf16(x1 c) + bf16(x0 s))
    grad:    g0' = bf16(bf16(g0 c) + bf16(g1 s)),  g1' = bf16(bf16(g1 c) - bf16(g0 s))

(the eager backward of mul/cat/neg/slice gives the same two-term sums; the zero padding of
the slice backward adds exact zeros). `exact=False` rounds once at the end instead, like
`flash_attn.layers.rotary.apply_rotary_emb`.

AdaLN modulation (`AdaLayerNorm`: y = ln * (1 + scale) + shift, scale/shift = aux.split(D)):

    a = round_aux(scale + 1); p = round_m(ln * a); y = round_m(p + shift)

with round_aux / round_m the rounding to aux's dtype / to promote(ln, aux) (bfloat16 under
autocast for aux; ln is float32 under autocast). The output can be written directly in the
autocast dtype when the only consumers are autocast Linears (they cast it anyway). Backward,
as eager's AddBackward / MulBackward / SplitBackward:

    g_ln = round_m(g * a) -> ln dtype;  g_scale = round_m(g * ln) -> aux dtype;
    g_shift = g -> aux dtype;  g_aux = cat(g_scale, g_shift)  (written in place, no cat)
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # e.g. macOS
    HAVE_TRITON = False

_BF16 = torch.bfloat16


def _round_to(v: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round a float32 tensor to `dtype` and back (no-op for float32)."""
    return v if dtype == torch.float32 else v.to(dtype).float()


def triton_ok(*tensors: torch.Tensor) -> bool:
    return HAVE_TRITON and all(t.is_cuda for t in tensors)


####################################################################################################
# RoPE


def rope_reference(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    conjugate: bool = False,
    exact: bool = True,
) -> torch.Tensor:
    """Rotate x (B, T, H, D) with half-width tables cos_h, sin_h (T, D/2); see module doc."""
    h = x.shape[-1] // 2
    xf = x.float()
    c = cos_h.float()[None, :, None, :]
    s = sin_h.float()[None, :, None, :]
    x0, x1 = xf[..., :h], xf[..., h:]

    def rb(v):
        return _round_to(v, x.dtype) if exact else v

    a0, a1, b0, b1 = rb(x0 * c), rb(x1 * c), rb(x1 * s), rb(x0 * s)
    if conjugate:
        o0, o1 = a0 + b0, a1 - b1
    else:
        o0, o1 = a0 - b0, a1 + b1
    return torch.cat([o0, o1], dim=-1).to(x.dtype)


if HAVE_TRITON:

    @triton.jit
    def _rb(v):
        return v.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    @triton.jit
    def _rope_kernel(
        OUT,
        X,
        COS,
        SIN,
        n_rows,
        n_heads,
        seqlen,
        HALF: tl.constexpr,
        CONJ: tl.constexpr,
        EXACT: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        # x and out are contiguous (B, T, H, D): one row per (b, t, h), D = 2 * HALF
        rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, HALF)
        m = (rows < n_rows)[:, None]
        t = (rows // n_heads) % seqlen
        xo = rows.to(tl.int64)[:, None] * (2 * HALF) + cols[None, :]
        co = t.to(tl.int64)[:, None] * HALF + cols[None, :]
        c = tl.load(COS + co, mask=m, other=1.0).to(tl.float32)
        s = tl.load(SIN + co, mask=m, other=0.0).to(tl.float32)
        x0 = tl.load(X + xo, mask=m, other=0.0).to(tl.float32)
        x1 = tl.load(X + xo + HALF, mask=m, other=0.0).to(tl.float32)
        if EXACT:
            a0 = _rb(x0 * c)
            a1 = _rb(x1 * c)
            b0 = _rb(x1 * s)
            b1 = _rb(x0 * s)
        else:
            a0 = x0 * c
            a1 = x1 * c
            b0 = x1 * s
            b1 = x0 * s
        if CONJ:
            o0 = a0 + b0
            o1 = a1 - b1
        else:
            o0 = a0 - b0
            o1 = a1 + b1
        tl.store(OUT + xo, o0.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)
        tl.store(OUT + xo + HALF, o1.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)


def rope_triton(x, cos_h, sin_h, conjugate=False, exact=True):
    """Triton version of `rope_reference` (bfloat16 x, power-of-2 half head dim)."""
    x = x.contiguous()
    cos_h, sin_h = cos_h.contiguous(), sin_h.contiguous()
    b, t, h, d = x.shape
    out = torch.empty_like(x)
    n_rows = b * t * h
    block_r = 64
    grid = (triton.cdiv(n_rows, block_r),)
    with torch.cuda.device(x.device.index):
        _rope_kernel[grid](
            out,
            x,
            cos_h,
            sin_h,
            n_rows,
            h,
            t,
            HALF=d // 2,
            CONJ=conjugate,
            EXACT=exact,
            BLOCK_R=block_r,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def rope_supported(x: torch.Tensor, cos_h: torch.Tensor) -> bool:
    """Shapes and dtypes the fused RoPE handles: x (B, T, H, D) bf16, tables (T, D/2)."""
    d = x.shape[-1]
    return (
        x.dim() == 4
        and x.dtype == _BF16
        and cos_h.dtype == _BF16
        and d % 2 == 0
        and (d // 2) & (d // 2 - 1) == 0
        and cos_h.shape == (x.shape[1], d // 2)
    )


def _rope_apply(x, cos_h, sin_h, conjugate, exact):
    if triton_ok(x, cos_h):
        return rope_triton(x, cos_h, sin_h, conjugate, exact)
    with torch.autocast(x.device.type, enabled=False):
        return rope_reference(x, cos_h, sin_h, conjugate, exact)


class RotaryHalfSplit(torch.autograd.Function):
    """Fused RoPE for one tensor x (B, T, H, D); cos_h / sin_h (T, D/2) need no gradient."""

    @staticmethod
    def forward(ctx, x, cos_h, sin_h, exact=True):
        ctx.save_for_backward(cos_h, sin_h)
        ctx.exact = exact
        return _rope_apply(x, cos_h, sin_h, False, exact)

    @staticmethod
    def backward(ctx, g):
        cos_h, sin_h = ctx.saved_tensors
        return _rope_apply(g, cos_h, sin_h, True, ctx.exact), None, None, None


####################################################################################################
# AdaLN modulation


def adaln_modulate_reference(ln, aux, out_dtype=None):
    """y = ln * (1 + scale) + shift with eager's roundings; scale, shift = aux.split(D, -1)."""
    d = ln.shape[-1]
    m_dtype = torch.promote_types(ln.dtype, aux.dtype)
    sc, sh = aux[..., :d].float(), aux[..., d:].float()
    a = _round_to(sc + 1.0, aux.dtype)
    p = _round_to(ln.float() * a, m_dtype)
    y = (p + sh).to(m_dtype)
    return y if out_dtype is None else y.to(out_dtype)


def adaln_modulate_backward_reference(g, ln, aux):
    """Gradients (ln, aux) of `adaln_modulate_reference` for the output gradient g."""
    d = ln.shape[-1]
    m_dtype = torch.promote_types(ln.dtype, aux.dtype)
    gf = _round_to(g.float(), m_dtype)  # g arrives in the output dtype; eager's is m_dtype
    a = _round_to(aux[..., :d].float() + 1.0, aux.dtype)
    g_ln = (gf * a).to(m_dtype).to(ln.dtype)
    g_scale = (gf * ln.float()).to(m_dtype).to(aux.dtype)
    g_shift = gf.to(m_dtype).to(aux.dtype)
    return g_ln, torch.cat([g_scale, g_shift], dim=-1)


if HAVE_TRITON:

    @triton.jit
    def _adaln_fwd_kernel(
        Y,
        LN,
        AUX,
        n_rows,
        D,
        AUX_BF16: tl.constexpr,
        M_BF16: tl.constexpr,
        Y_BF16: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        m = (rows < n_rows)[:, None] & (cols < D)[None, :]
        o = rows[:, None] * D + cols[None, :]
        oa = rows[:, None] * (2 * D) + cols[None, :]
        ln = tl.load(LN + o, mask=m, other=0.0).to(tl.float32)
        sc = tl.load(AUX + oa, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(AUX + oa + D, mask=m, other=0.0).to(tl.float32)
        a = sc + 1.0
        if AUX_BF16:
            a = _rb(a)
        p = ln * a
        if M_BF16:
            p = _rb(p)
        y = p + sh
        if Y_BF16:
            tl.store(Y + o, y.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)
        else:
            tl.store(Y + o, y, mask=m)

    @triton.jit
    def _adaln_bwd_kernel(
        GLN,
        GAUX,
        G,
        LN,
        AUX,
        n_rows,
        D,
        AUX_BF16: tl.constexpr,
        M_BF16: tl.constexpr,
        LN_BF16: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
        cols = tl.arange(0, BLOCK_D)
        m = (rows < n_rows)[:, None] & (cols < D)[None, :]
        o = rows[:, None] * D + cols[None, :]
        oa = rows[:, None] * (2 * D) + cols[None, :]
        g = tl.load(G + o, mask=m, other=0.0).to(tl.float32)
        if M_BF16:
            g = _rb(g)
        ln = tl.load(LN + o, mask=m, other=0.0).to(tl.float32)
        sc = tl.load(AUX + oa, mask=m, other=0.0).to(tl.float32)
        a = sc + 1.0
        if AUX_BF16:
            a = _rb(a)
        g_ln = g * a
        g_sc = g * ln
        if M_BF16:
            g_ln = _rb(g_ln)
            g_sc = _rb(g_sc)
        if LN_BF16:
            tl.store(GLN + o, g_ln.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)
        else:
            tl.store(GLN + o, g_ln, mask=m)
        if AUX_BF16:
            tl.store(GAUX + oa, g_sc.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)
            tl.store(GAUX + oa + D, g.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)
        else:
            tl.store(GAUX + oa, g_sc, mask=m)
            tl.store(GAUX + oa + D, g, mask=m)


def _adaln_launch_params(d):
    block_d = triton.next_power_of_2(d)
    return max(1, 4096 // block_d), block_d


def adaln_modulate_triton(ln, aux, out_dtype):
    d = ln.shape[-1]
    ln2, aux2 = ln.reshape(-1, d).contiguous(), aux.reshape(-1, 2 * d).contiguous()
    m_dtype = torch.promote_types(ln.dtype, aux.dtype)
    y = torch.empty(ln2.shape, dtype=out_dtype, device=ln.device)
    block_r, block_d = _adaln_launch_params(d)
    n = ln2.shape[0]
    with torch.cuda.device(ln.device.index):
        _adaln_fwd_kernel[(triton.cdiv(n, block_r),)](
            y,
            ln2,
            aux2,
            n,
            d,
            AUX_BF16=aux.dtype == _BF16,
            M_BF16=m_dtype == _BF16,
            Y_BF16=out_dtype == _BF16,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y.view(ln.shape)


def adaln_modulate_backward_triton(g, ln, aux):
    d = ln.shape[-1]
    g2 = g.reshape(-1, d).contiguous()
    ln2, aux2 = ln.reshape(-1, d), aux.reshape(-1, 2 * d)
    m_dtype = torch.promote_types(ln.dtype, aux.dtype)
    g_ln = torch.empty_like(ln2)
    g_aux = torch.empty_like(aux2)
    block_r, block_d = _adaln_launch_params(d)
    n = ln2.shape[0]
    with torch.cuda.device(ln.device.index):
        _adaln_bwd_kernel[(triton.cdiv(n, block_r),)](
            g_ln,
            g_aux,
            g2,
            ln2,
            aux2,
            n,
            d,
            AUX_BF16=aux.dtype == _BF16,
            M_BF16=m_dtype == _BF16,
            LN_BF16=ln.dtype == _BF16,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return g_ln.view(ln.shape), g_aux.view(aux.shape)


def adaln_supported(ln: torch.Tensor, aux: torch.Tensor) -> bool:
    """One aux row per ln row (no broadcasting), float32 / bfloat16 only."""
    return (
        aux.shape[:-1] == ln.shape[:-1]
        and aux.shape[-1] == 2 * ln.shape[-1]
        and ln.dtype in (torch.float32, _BF16)
        and aux.dtype in (torch.float32, _BF16)
    )


class AdaLNModulate(torch.autograd.Function):
    """y = ln * (1 + scale) + shift (scale, shift = aux.split(D, -1)) as one kernel per pass.

    out_dtype: the eager result dtype promote(ln, aux), or the autocast dtype when every
    consumer of y casts it to that dtype anyway (an autocast Linear).
    """

    @staticmethod
    def forward(ctx, ln, aux, out_dtype):
        ln = ln.contiguous()
        aux = aux.contiguous()
        ctx.save_for_backward(ln, aux)
        if triton_ok(ln, aux):
            return adaln_modulate_triton(ln, aux, out_dtype)
        with torch.autocast(ln.device.type, enabled=False):
            return adaln_modulate_reference(ln, aux, out_dtype)

    @staticmethod
    def backward(ctx, g):
        ln, aux = ctx.saved_tensors
        if triton_ok(ln, aux):
            g_ln, g_aux = adaln_modulate_backward_triton(g, ln, aux)
        else:
            with torch.autocast(ln.device.type, enabled=False):
                g_ln, g_aux = adaln_modulate_backward_reference(g, ln, aux)
        return g_ln, g_aux, None
