# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import numpy as np
import torch


####################################################################################################
def positional_encoding_harmonic(x):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device
    dtype = x.dtype

    len_token_seq = x.shape[-2]
    pe = torch.zeros(len_token_seq, dim_embed, device=dev, dtype=dtype)
    position = torch.arange(0, len_token_seq, device=dev, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim_embed, 2, device=dev, dtype=dtype) * -(math.log(10000) / dim_embed)
    )

    pe[:, 0::2] = torch.sin(position * div[: pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_idx(x, s_idx):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    len_token_seq = x.shape[0]
    pe = torch.zeros(x.shape[-2:], device=dev)
    pos = (s_idx + 1) * torch.ones(len_token_seq, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed

    pe[:, 0::2] = torch.sin(torch.outer(pos, xs))
    pe[:, 1::2] = torch.cos(torch.outer(pos, xs))
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_global(x):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    pe = torch.zeros(x.shape[-3], x.shape[-2], dim_embed, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed
    pe[..., 0::2] = 0.5 * torch.sin(torch.outer(8 * torch.arange(x.shape[-2], device=dev), xs))
    pe[..., 0::2] += (
        torch.sin(torch.outer(torch.arange(x.shape[-3], device=dev), xs))
        .unsqueeze(1)
        .repeat((1, x.shape[-2], 1))
    )
    pe[..., 1::2] = 0.5 * torch.cos(torch.outer(8 * torch.arange(x.shape[-2], device=dev), xs))
    pe[..., 1::2] += (
        torch.cos(torch.outer(torch.arange(x.shape[-3], device=dev), xs))
        .unsqueeze(1)
        .repeat((1, x.shape[-2], 1))
    )
    x = x + pe

    return x


####################################################################################################
def positional_encoding_harmonic_coord(x, lats, lons):
    """space time harmonic positional encoding"""

    dim_embed = x.shape[-1]
    dev = x.device

    pe = torch.zeros(x.shape[0], dim_embed, device=dev)
    xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=dev) / dim_embed
    pe[..., 0::2] = 0.5 * torch.sin(torch.outer(lats, xs))
    pe[..., 1::2] = 0.5 * torch.cos(torch.outer(lons, xs))[..., : pe[..., 1::2].shape[-1]]
    x = x + pe

    return x


####################################################################################################
# The functions rotate_half() and apply_rotary_pos_emb() below are derived from LLaMA and Qwen3
# models, originally developed by Meta Platforms, Inc., The Qwen team, Alibaba Group and the
# HuggingFace Inc. team, licensed under the Apache License, Version 2.0.
# Source: https://github.com/qiuzh20/gated_attention/blob/main/modeling_qwen3.py


def rotate_half(x):
    """Rotates half the hidden dims of the input."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine embedding tensor.
        sin: Sine embedding tensor.
        unsqueeze_dim: Dimension along which to unsqueeze cos/sin for broadcasting.
    """

    cos = cos.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(dtype=q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


####################################################################################################
def rotary_embedding_2d(coords, dim, base=10000.0):
    """Create 2D RoPE embeddings from latitude/longitude coordinates.

    Args:
        coords: Tensor of shape (..., 2) with coordinates in radians (lat, lon).
        dim: Head dimension to encode; must be divisible by 4.
        base: RoPE base frequency.

    Returns:
        Tuple of (cos, sin) tensors with shape (..., dim).
    """

    assert coords.shape[-1] == 2, (
        f"coords last dimension must be 2 (lat, lon); got {coords.shape[-1]}"
    )
    assert dim % 4 == 0, f"2D rotary embeddings require dim to be divisible by 4; got {dim}"

    # Split the rotary frequencies evenly between latitude and longitude to stay local to each cell.
    half_dim = dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_dim, 2, device=coords.device, dtype=coords.dtype) / half_dim)
    )

    lat, lon = coords.unbind(dim=-1)
    freq_lat = lat.unsqueeze(-1) * inv_freq
    freq_lon = lon.unsqueeze(-1) * inv_freq

    freqs = torch.cat((freq_lat, freq_lon), dim=-1)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = torch.cos(emb)
    sin = torch.sin(emb)

    return cos, sin


####################################################################################################
# How rotary_pos_emb_2d builds and applies the tables (config `rope_impl`, set at model build):
#   eager       tables from coords on every call, applied with mul/cat/neg/add (original)
#   cached      tables computed once per coords tensor and dtype, applied as in eager;
#               bitwise equal to eager
#   fused       cached tables, applied with one kernel per tensor and pass that rounds like
#               eager (fused_kernels.RotaryHalfSplit, exact=True); bitwise equal to eager
#   fused_flash cached tables, flash_attn.layers.rotary.apply_rotary_emb (one rounding at the
#               end instead of three: differs from eager by bfloat16 rounding)
ROPE_IMPLS = ("eager", "cached", "fused", "fused_flash")
_rope_impl = "eager"


def set_rope_impl(impl: str) -> None:
    global _rope_impl
    if impl not in ROPE_IMPLS:
        raise ValueError(f"rope_impl must be one of {ROPE_IMPLS}, got {impl!r}")
    _rope_impl = impl


def rotary_tables_2d_cached(coords, dim, dtype, base=10000.0):
    """(cos, sin, cos_half, sin_half) for `coords`, as `apply_rotary_pos_emb` uses them.

    cos / sin are `rotary_embedding_2d(coords, dim)` cast to `dtype` (shape (..., dim));
    cos_half / sin_half are their first halves (the second halves are equal), contiguous.

    Computed once per coords tensor and kept on it (attribute), keyed by everything that can
    change the result: the coords version (in-place updates), dim, base, dtype and the
    autocast state (under CUDA autocast `pow` runs in float32, so the tables differ from the
    ones computed outside autocast). The coords of the global and forecasting engines are a
    buffer set once in `ModelParams.reset_parameters`, so the tables are computed in the first
    forward and reused in all later calls, recomputations and steps. Bitwise equal to
    computing them per call in the same autocast state.
    """
    dev = coords.device.type
    if coords.requires_grad or torch.compiler.is_compiling():
        cos, sin = rotary_embedding_2d(coords, dim, base=base)
        cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin, cos[..., : dim // 2].contiguous(), sin[..., : dim // 2].contiguous()
    autocast = torch.is_autocast_enabled(dev)
    key = (
        coords._version,
        dim,
        float(base),
        dtype,
        autocast,
        torch.get_autocast_dtype(dev) if autocast else None,
        torch.is_inference_mode_enabled(),
    )
    cache = coords.__dict__.setdefault("_rope_tables", {})
    tables = cache.get(key)
    if tables is None:
        with torch.no_grad():
            cos, sin = rotary_embedding_2d(coords, dim, base=base)
            cos, sin = cos.to(dtype), sin.to(dtype)
            tables = (
                cos,
                sin,
                cos[..., : dim // 2].contiguous(),
                sin[..., : dim // 2].contiguous(),
            )
        if len(cache) >= 8:
            cache.clear()
        cache[key] = tables
    return tables


def _fused_layout(x, cos_h, unsqueeze_dim):
    """x as (B, T, H, D) for the fused kernels, or None if the layout is not handled."""
    if unsqueeze_dim == 2 and x.dim() == 4:  # (B, T, H, D): MultiSelfAttentionHead
        return x
    if unsqueeze_dim == 1 and x.dim() == 3:  # (T, H, D): MultiSelfAttentionHeadVarlen
        return x.unsqueeze(0)
    return None  # e.g. (B, H, T, D) of MultiSelfAttentionHeadLocal: not handled


def apply_rotary_pos_emb_fused(q, k, cos_h, sin_h, unsqueeze_dim, impl):
    """Fused RoPE on q and k; returns None if shapes / dtypes / devices are not handled."""
    from weathergen.model import fused_kernels as fk

    if cos_h.dim() == 3 and cos_h.shape[0] == 1:  # (1, T, D/2): one table for all samples
        cos_h, sin_h = cos_h[0], sin_h[0]
    q4, k4 = _fused_layout(q, cos_h, unsqueeze_dim), _fused_layout(k, cos_h, unsqueeze_dim)
    if q4 is None or k4 is None:
        return None
    if not (fk.rope_supported(q4, cos_h) and fk.rope_supported(k4, sin_h)):
        return None
    if impl == "fused_flash":
        if not q.is_cuda:
            return None
        from flash_attn.layers.rotary import apply_rotary_emb

        qe, ke = apply_rotary_emb(q4, cos_h, sin_h), apply_rotary_emb(k4, cos_h, sin_h)
    else:
        if q.is_cuda and not fk.HAVE_TRITON:
            return None
        qe = fk.RotaryHalfSplit.apply(q4, cos_h, sin_h, True)
        ke = fk.RotaryHalfSplit.apply(k4, cos_h, sin_h, True)
    return qe.view(q.shape), ke.view(k.shape)


####################################################################################################
def rotary_pos_emb_2d(q, k, coords, base=10000.0, unsqueeze_dim=1):
    """Convenience wrapper that builds 2D RoPE embeddings and applies them to q/k."""

    if _rope_impl == "eager":
        cos, sin = rotary_embedding_2d(coords, q.shape[-1], base=base)
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    cos, sin, cos_h, sin_h = rotary_tables_2d_cached(coords, q.shape[-1], q.dtype, base=base)
    if _rope_impl != "cached":
        out = apply_rotary_pos_emb_fused(q, k, cos_h, sin_h, unsqueeze_dim, _rope_impl)
        if out is not None:
            return out
    # the tables are already in q.dtype: the .to() in apply_rotary_pos_emb is a no-op
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
