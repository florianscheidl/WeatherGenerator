# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Triton-backed LayerNorm.

This module provides a CUDA bf16 fast path for LayerNorm that avoids PyTorch's
internal dtype promotion. The implementation is intentionally conservative:
- forward is fused in Triton
- backward is implemented in PyTorch using saved statistics
- fallback to the standard PyTorch path is automatic when Triton is unavailable

The kernel supports arbitrary leading dimensions by flattening the input to a
2D tensor of shape [rows, hidden]. Normalization is performed over the last
hidden dimension only.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - runtime dependent
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - graceful fallback
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _layernorm_fwd_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        mean_ptr,
        inv_std_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        n_cols,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row * stride_xm
        y_row = y_ptr + row * stride_ym

        sum_ = tl.zeros([], dtype=tl.float32)
        sumsq = tl.zeros([], dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < n_cols
            x = tl.load(x_row + offs * stride_xn, mask=mask, other=0).to(tl.float32)
            sum_ += tl.sum(x, axis=0)
            sumsq += tl.sum(x * x, axis=0)

        mean = sum_ / n_cols
        var = sumsq / n_cols - mean * mean
        inv_std = tl.math.rsqrt(var + eps)
        tl.store(mean_ptr + row, mean)
        tl.store(inv_std_ptr + row, inv_std)

        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < n_cols
            x = tl.load(x_row + offs * stride_xn, mask=mask, other=0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=1).to(tl.float32)
            b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.float32)
            y = (x - mean) * inv_std * w + b
            tl.store(y_row + offs * stride_yn, y.to(tl.bfloat16), mask=mask)


class _TritonLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        if not HAS_TRITON or not x.is_cuda or x.dtype != torch.bfloat16:
            raise RuntimeError("Triton LayerNorm is only available for CUDA bf16 tensors")

        if x.shape[-1] != weight.shape[0] or x.shape[-1] != bias.shape[0]:
            raise ValueError(
                "LayerNorm parameter shape mismatch: hidden dim does not match weight/bias"
            )

        x_contig = x.contiguous()
        weight_contig = weight.contiguous()
        bias_contig = bias.contiguous()

        input_shape = x_contig.shape
        hidden = input_shape[-1]
        rows = x_contig.numel() // hidden

        x_2d = x_contig.view(rows, hidden)
        y_2d = torch.empty_like(x_2d)
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        inv_std = torch.empty((rows,), device=x.device, dtype=torch.float32)

        block_n = 256
        _layernorm_fwd_kernel[(rows,)](
            x_2d,
            y_2d,
            weight_contig,
            bias_contig,
            mean,
            inv_std,
            x_2d.stride(0),
            x_2d.stride(1),
            y_2d.stride(0),
            y_2d.stride(1),
            hidden,
            eps,
            BLOCK_N=block_n,
            num_warps=4,
        )

        ctx.save_for_backward(x_contig, weight_contig, bias_contig, mean, inv_std)
        ctx.input_shape = input_shape
        ctx.eps = eps
        return y_2d.view(input_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias, mean, inv_std = ctx.saved_tensors
        input_shape = ctx.input_shape
        hidden = input_shape[-1]
        rows = x.numel() // hidden

        grad_output = grad_output.contiguous().view(rows, hidden)
        x_2d = x.view(rows, hidden)
        mean = mean.view(rows, 1)
        inv_std = inv_std.view(rows, 1)

        x_mu = x_2d - mean
        x_hat = x_mu * inv_std

        grad_weight = torch.sum(grad_output * x_hat, dim=0)
        grad_bias = torch.sum(grad_output, dim=0)

        dy = grad_output * weight
        sum_dy = torch.sum(dy, dim=-1, keepdim=True)
        sum_dy_xhat = torch.sum(dy * x_hat, dim=-1, keepdim=True)
        dx = (inv_std / hidden) * (hidden * dy - sum_dy - x_hat * sum_dy_xhat)

        return dx.view(input_shape), grad_weight, grad_bias, None


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """Apply LayerNorm, using Triton for CUDA bf16 tensors when available."""

    if HAS_TRITON and x.is_cuda and x.dtype == torch.bfloat16 and weight.is_cuda and bias.is_cuda:
        return _TritonLayerNormFn.apply(x, weight, bias, eps)

    var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
    x_norm = (x - mean) * torch.rsqrt(var + eps)
    return x_norm * weight + bias
