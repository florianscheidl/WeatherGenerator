# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Drop-in replacements for flash_attn_func and flash_attn_varlen_func built on torch's
scaled_dot_product_attention, for devices without flash-attn (MPS, XPU, CPU).

Tensor layouts and the output dtype (that of q) follow flash-attn. Packed variable-length
sequences are padded to the longest one, so memory grows with the number of sequences times
the longest sequence; this is meant for small (local) runs, not for performance.
"""

import torch
import torch.nn.functional as F


def _check_unsupported(softcap: float, causal: bool) -> None:
    if softcap != 0.0:
        raise NotImplementedError("softcap is only supported with flash attention")
    if causal:
        raise NotImplementedError("causal attention is only supported with flash attention")


def sdpa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Dense attention with layout (batch, seq, heads, dim), like flash_attn_func."""
    _check_unsupported(softcap, causal)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v.to(q.dtype)))
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=softmax_scale)
    return out.transpose(1, 2)


def _pad(x: torch.Tensor, seq_idx: torch.Tensor, pos: torch.Tensor, num_seqs: int, max_len: int):
    """Scatter packed (total, heads, dim) into zero-padded (num_seqs, heads, max_len, dim)."""
    padded = x.new_zeros(num_seqs, max_len, *x.shape[1:])
    padded[seq_idx, pos] = x
    return padded.transpose(1, 2)


def _packed_positions(cu_seqlens: torch.Tensor, total: int):
    """Sequence index and position within the sequence of every packed token."""
    lens = cu_seqlens.diff().long()
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lens), device=cu_seqlens.device), lens, output_size=total
    )
    pos = torch.arange(total, device=cu_seqlens.device) - cu_seqlens[:-1].long()[seq_idx]
    return lens, seq_idx, pos


def sdpa_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | torch.Tensor,
    max_seqlen_k: int | torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float = 0.0,
) -> torch.Tensor:
    """
    Attention over packed sequences with layout (total_tokens, heads, dim), like
    flash_attn_varlen_func: query sequence i attends to key/value sequence i, where sequence i
    spans cu_seqlens[i]:cu_seqlens[i + 1].
    """
    _check_unsupported(softcap, causal)
    num_seqs = cu_seqlens_q.numel() - 1
    # at least one (padding) position, so that sequences without keys have something to attend to
    max_q, max_k = max(int(max_seqlen_q), 1), max(int(max_seqlen_k), 1)

    _, seq_idx_q, pos_q = _packed_positions(cu_seqlens_q, q.shape[0])
    lens_k, seq_idx_k, pos_k = _packed_positions(cu_seqlens_k, k.shape[0])

    q_pad = _pad(q, seq_idx_q, pos_q, num_seqs, max_q)
    k_pad = _pad(k, seq_idx_k, pos_k, num_seqs, max_k)
    v_pad = _pad(v.to(q.dtype), seq_idx_k, pos_k, num_seqs, max_k)

    # Sequences without keys get output zero, as with flash attention. They attend to their
    # (zero) padding instead of to nothing, since a fully masked row yields NaN in the forward
    # and in the backward pass.
    empty_k = lens_k == 0
    mask_k = (torch.arange(max_k, device=q.device) < lens_k[:, None]) | empty_k[:, None]
    out = F.scaled_dot_product_attention(
        q_pad,
        k_pad,
        v_pad,
        attn_mask=mask_k[:, None, None, :],
        dropout_p=dropout_p,
        scale=softmax_scale,
    )

    out = out.transpose(1, 2)[seq_idx_q, pos_q]
    return torch.where(empty_k[seq_idx_q][:, None, None], 0.0, out)
