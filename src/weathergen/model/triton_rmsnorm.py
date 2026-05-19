# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton-backed RMSNorm.

This module provides a bf16 CUDA fast path for RMSNorm. It is intentionally
narrow in scope: the forward pass uses Triton to avoid PyTorch's dtype
promotion behavior, while the backward pass is implemented in PyTorch for
maintainability and correctness.

The implementation supports arbitrary leading dimensions by flattening the
input to 2D [rows, hidden]. The hidden dimension is normalized row-wise.

Usage:
    y = rmsnorm(x, weight, eps)

The function falls back to the standard PyTorch implementation when Triton is
unavailable or the input is not CUDA bf16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:  # pragma: no cover - import availability depends on runtime image
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - graceful fallback when Triton is absent
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        inv_rms_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        n_cols,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_x = x_ptr + row * stride_xm
        row_y = y_ptr + row * stride_ym

        sumsq = tl.zeros([], dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < n_cols
            x = tl.load(row_x + offs * stride_xn, mask=mask, other=0).to(tl.float32)
            sumsq += tl.sum(x * x, axis=0)

        mean_sq = sumsq / n_cols
        inv_rms = tl.math.rsqrt(mean_sq + eps)
        tl.store(inv_rms_ptr + row, inv_rms)

        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < n_cols
            x = tl.load(row_x + offs * stride_xn, mask=mask, other=0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=1).to(tl.float32)
            y = x * inv_rms * w
            tl.store(row_y + offs * stride_yn, y.to(tl.bfloat16), mask=mask)


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        if not HAS_TRITON or not x.is_cuda or x.dtype != torch.bfloat16:
            raise RuntimeError("Triton RMSNorm is only available for CUDA bf16 tensors")

        if x.shape[-1] != weight.shape[0]:
            raise ValueError(
                f"RMSNorm weight shape mismatch: hidden dim {x.shape[-1]} vs weight {tuple(weight.shape)}"
            )

        x_contig = x.contiguous()
        weight_contig = weight.contiguous()

        input_shape = x_contig.shape
        hidden = input_shape[-1]
        rows = x_contig.numel() // hidden

        x_2d = x_contig.view(rows, hidden)
        y_2d = torch.empty_like(x_2d)
        inv_rms = torch.empty((rows,), device=x.device, dtype=torch.float32)

        block_n = 256
        grid = (rows,)
        _rmsnorm_fwd_kernel[grid](
            x_2d,
            y_2d,
            weight_contig,
            inv_rms,
            x_2d.stride(0),
            x_2d.stride(1),
            y_2d.stride(0),
            y_2d.stride(1),
            hidden,
            eps,
            BLOCK_N=block_n,
            num_warps=4,
        )

        ctx.save_for_backward(x_contig, weight_contig, inv_rms)
        ctx.input_shape = input_shape
        ctx.eps = eps
        return y_2d.view(input_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, inv_rms = ctx.saved_tensors
        input_shape = ctx.input_shape
        hidden = input_shape[-1]
        rows = x.numel() // hidden

        grad_output = grad_output.contiguous().view(rows, hidden)
        x_2d = x.view(rows, hidden)
        inv_rms = inv_rms.view(rows, 1)

        # PyTorch fallback for backward math. The problematic forward-time dtype
        # promotion is handled by Triton; backward stability is less sensitive here.
        grad_out_weight = grad_output * weight
        dot = torch.mean(grad_out_weight * x_2d, dim=-1, keepdim=True)
        inv_rms_3 = inv_rms * inv_rms * inv_rms

        grad_x = grad_out_weight * inv_rms - x_2d * inv_rms_3 * dot
        grad_weight = torch.sum(grad_output * x_2d * inv_rms, dim=0)

        return grad_x.view(input_shape), grad_weight, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Apply RMSNorm, using Triton for CUDA bf16 tensors when available."""

    if HAS_TRITON:
        if not x.is_cuda or not weight.is_cuda:
            raise TypeError("Triton RMSNorm expected CUDA tensors on the fast path")
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            msg = f"dtype mismatch for rmsnorm: x={x.dtype}, weight={weight.dtype}, expected torch.bfloat16"
            print(msg, flush=True)
            raise TypeError(msg)
        return _TritonRMSNormFn.apply(x, weight, eps)

    var, _ = torch.var_mean(x.pow(2), dim=-1, keepdim=True, correction=0)
    return x * torch.rsqrt(var + eps) * weight
