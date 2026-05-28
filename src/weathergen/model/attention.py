# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

import torch
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from weathergen.model.norms import AdaLayerNorm, RMSNorm
from weathergen.model.positional_encoding import rotary_pos_emb_2d

"""
Attention blocks used by WeatherGenerator.

Some blocks optionally apply 2D RoPE. When enabled, the caller must provide per-token 2D
coordinates aligned with the token order (lat, lon in radians).
"""


def _zero_length_segment_indices(lengths: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(lengths[1:] == 0, as_tuple=False).flatten()


def _filter_varlen_lengths(lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seg_lengths = lengths.to(torch.int64)[1:]
    keep_mask = seg_lengths > 0
    filtered_lengths = torch.cat(
        [torch.zeros(1, device=lengths.device, dtype=torch.int32), seg_lengths[keep_mask].to(torch.int32)]
    )
    return filtered_lengths, keep_mask


def _filter_varlen_tokens(tokens: torch.Tensor, lengths: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    starts = torch.cumsum(lengths.to(torch.int64), 0)[:-1]
    seg_lengths = lengths.to(torch.int64)[1:]
    kept_segments = [
        tokens[start : start + seg_len]
        for start, seg_len, keep in zip(starts.tolist(), seg_lengths.tolist(), keep_mask.tolist(), strict=False)
        if keep and seg_len > 0
    ]
    if kept_segments:
        return torch.cat(kept_segments, dim=0)
    return tokens.new_empty((0, *tokens.shape[1:]))


class MultiSelfAttentionHeadVarlen(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        with_2d_rope=False,
    ):
        super(MultiSelfAttentionHeadVarlen, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual
        self.with_2d_rope = with_2d_rope

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype

        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x, x_lens, ada_ln_aux=None, coords=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x.shape[0], self.num_heads, x.shape[-1] // self.num_heads]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s).to(self.dtype)

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=1)

        # set dropout rate according to training/eval mode as required by flash_attn
        # dropout_rate = self.dropout_rate if self.training else 0.0

        zero_idx = _zero_length_segment_indices(x_lens)
        if zero_idx.numel() > 0:
            print(
                f"[flash_attn self varlen] zero_length_segments count={zero_idx.numel()} first_indices={zero_idx[:10].tolist()}",
                flush=True,
            )
        filtered_x_lens, keep_mask = _filter_varlen_lengths(x_lens)
        qs = _filter_varlen_tokens(qs, x_lens, keep_mask)
        ks = _filter_varlen_tokens(ks, x_lens, keep_mask)
        vs = _filter_varlen_tokens(vs, x_lens, keep_mask)
        cum_x_lens = torch.cumsum(filtered_x_lens, 0, dtype=torch.int32)
        print(
            "[flash_attn self varlen] "
            f"q_shape={tuple(qs.shape)} k_shape={tuple(ks.shape)} v_shape={tuple(vs.shape)} "
            f"q_dtype={qs.dtype} k_dtype={ks.dtype} v_dtype={vs.dtype} "
            f"x_lens_shape={tuple(filtered_x_lens.shape)} x_lens_dtype={filtered_x_lens.dtype} "
            f"cum_shape={tuple(cum_x_lens.shape)} cum_dtype={cum_x_lens.dtype} cum_stride={cum_x_lens.stride()} "
            f"cum_first={cum_x_lens[0].item() if cum_x_lens.numel() else 'empty'} "
            f"cum_last={cum_x_lens[-1].item() if cum_x_lens.numel() else 'empty'} "
            f"q_tokens={qs.shape[0]} k_tokens={ks.shape[0]} v_tokens={vs.shape[0]} "
            f"cum_diffs_min={torch.diff(cum_x_lens).min().item() if cum_x_lens.numel() > 1 else 'n/a'}",
            flush=True,
        )
        # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
        outs, _ = flash_attn_varlen_func(
            q=qs,
            k=ks,
            v=vs,
            cu_seqlens_q=cum_x_lens,
            cu_seqlens_k=cum_x_lens,
        )

        out = self.proj_out(outs.flatten(-2, -1))

        if self.with_residual:
            out = out + x_in

        return out


class MultiSelfAttentionHeadVarlenFlex(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiSelfAttentionHeadVarlenFlex, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)
        self.dtype = attention_dtype

        assert with_flash, "Only flash attention supported at the moment"

        def att(qs, ks, vs, x_mask):
            def sparsity_mask(score, b, h, q_idx, kv_idx):
                return (q_idx // 16) == (kv_idx % 16)

            return flex_attention(qs, ks, vs, score_mod=sparsity_mask)

        self.compiled_flex_attention = torch.compile(att, dynamic=False)

    def forward(self, x, x_lens=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x.shape[0], 1, self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype).permute([1, 2, 0, 3])
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype).permute([1, 2, 0, 3])
        vs = self.proj_heads_v(x).reshape(s).permute([1, 2, 0, 3])

        outs = self.compiled_flex_attention(qs, ks, vs).transpose(1, 2).squeeze()

        out = self.dropout(self.proj_out(outs.flatten(-2, -1)))
        if self.with_residual:
            out = out + x_in

        return out


class MultiSelfAttentionHeadLocal(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        qkv_len,
        block_factor,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        with_2d_rope=False,
    ):
        super(MultiSelfAttentionHeadLocal, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.with_residual = with_residual
        self.with_2d_rope = with_2d_rope

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported."

        # define block mask
        def mask_block_local(batch, head, idx_q, idx_kv):
            return (idx_q // block_factor) == (idx_kv // block_factor)

        self.block_mask = create_block_mask(
            mask_block_local, B=None, H=None, Q_LEN=qkv_len, KV_LEN=qkv_len
        )
        # compile for efficiency
        self.flex_attention = torch.compile(flex_attention, dynamic=False)

    def forward(self, x, coords=None, ada_ln_aux=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads
        s = [x.shape[0], x.shape[1], self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype).permute([0, 2, 1, 3])
        vs = self.proj_heads_v(x).reshape(s).permute([0, 2, 1, 3])

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=1)

        outs = self.flex_attention(qs, ks, vs, block_mask=self.block_mask).transpose(1, 2)

        out = self.proj_out(self.dropout(outs.flatten(-2, -1)))
        if self.with_residual:
            out = x_in + out

        return out


class MultiCrossAttentionHeadVarlen(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiCrossAttentionHeadVarlen, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_flash = with_flash
        self.softcap = softcap

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        if dim_aux is not None:
            self.lnorm_in_q = AdaLayerNorm(dim_embed_q, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed_q, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        if self.with_residual:
            x_q_in = x_q
        x_q = self.lnorm_in_q(x_q) if ada_ln_aux is None else self.lnorm_in_q(x_q, ada_ln_aux)
        x_kv = self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], self.num_heads, self.dim_head_proj]
        qs = self.lnorm_q(self.proj_heads_q(x_q).reshape(s)).to(self.dtype)
        s = [x_kv.shape[0], self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x_kv).reshape(s).to(self.dtype)

        # set dropout rate according to training/eval mode as required by flash_attn
        # dropout_rate = self.dropout_rate if self.training else 0.0

        if x_kv_lens is not None:
            zero_q_idx = _zero_length_segment_indices(x_q_lens)
            zero_kv_idx = _zero_length_segment_indices(x_kv_lens)
            if zero_q_idx.numel() > 0:
                print(
                    f"[flash_attn cross varlen] zero_length_q_segments count={zero_q_idx.numel()} first_indices={zero_q_idx[:10].tolist()}",
                    flush=True,
                )
            if zero_kv_idx.numel() > 0:
                print(
                    f"[flash_attn cross varlen] zero_length_kv_segments count={zero_kv_idx.numel()} first_indices={zero_kv_idx[:10].tolist()}",
                    flush=True,
                )
            filtered_x_q_lens, keep_q_mask = _filter_varlen_lengths(x_q_lens)
            filtered_x_kv_lens, keep_kv_mask = _filter_varlen_lengths(x_kv_lens)
            qs = _filter_varlen_tokens(qs, x_q_lens, keep_q_mask)
            ks = _filter_varlen_tokens(ks, x_kv_lens, keep_kv_mask)
            vs = _filter_varlen_tokens(vs, x_kv_lens, keep_kv_mask)
            cum_x_q_lens = torch.cumsum(filtered_x_q_lens, 0, dtype=torch.int32)
            cum_x_kv_lens = torch.cumsum(filtered_x_kv_lens, 0, dtype=torch.int32)
            print(
                "[flash_attn cross varlen] "
                f"q_shape={tuple(qs.shape)} k_shape={tuple(ks.shape)} v_shape={tuple(vs.shape)} "
                f"q_dtype={qs.dtype} k_dtype={ks.dtype} v_dtype={vs.dtype} "
                f"x_q_lens_shape={tuple(filtered_x_q_lens.shape)} x_q_lens_dtype={filtered_x_q_lens.dtype} "
                f"x_kv_lens_shape={tuple(filtered_x_kv_lens.shape)} x_kv_lens_dtype={filtered_x_kv_lens.dtype} "
                f"cum_q_shape={tuple(cum_x_q_lens.shape)} cum_q_dtype={cum_x_q_lens.dtype} cum_q_stride={cum_x_q_lens.stride()} "
                f"cum_kv_shape={tuple(cum_x_kv_lens.shape)} cum_kv_dtype={cum_x_kv_lens.dtype} cum_kv_stride={cum_x_kv_lens.stride()} "
                f"cum_q_first={cum_x_q_lens[0].item() if cum_x_q_lens.numel() else 'empty'} "
                f"cum_q_last={cum_x_q_lens[-1].item() if cum_x_q_lens.numel() else 'empty'} "
                f"cum_kv_first={cum_x_kv_lens[0].item() if cum_x_kv_lens.numel() else 'empty'} "
                f"cum_kv_last={cum_x_kv_lens[-1].item() if cum_x_kv_lens.numel() else 'empty'} "
                f"q_tokens={qs.shape[0]} k_tokens={ks.shape[0]} v_tokens={vs.shape[0]} "
                f"cum_q_diffs_min={torch.diff(cum_x_q_lens).min().item() if cum_x_q_lens.numel() > 1 else 'n/a'} "
                f"cum_kv_diffs_min={torch.diff(cum_x_kv_lens).min().item() if cum_x_kv_lens.numel() > 1 else 'n/a'}",
                flush=True,
            )
            outs, _ = flash_attn_varlen_func(
                q=qs,
                k=ks,
                v=vs,
                cu_seqlens_q=cum_x_q_lens,
                cu_seqlens_k=cum_x_kv_lens,
            )
        else:
            assert False

        outs = self.proj_out(outs.flatten(-2, -1))
        if self.with_residual:
            outs = x_q_in + outs

        return outs


class MultiCrossAttentionHeadVarlenSlicedQ(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_slices_q,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        softcap=0.0,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiCrossAttentionHeadVarlenSlicedQ, self).__init__()

        self.num_slices_q = num_slices_q
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_flash = with_flash
        self.softcap = softcap

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        if dim_aux is not None:
            self.lnorm_in_q = AdaLayerNorm(dim_embed_q, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        assert num_heads % num_slices_q == 0
        num_heads_r = num_heads
        self.proj_heads_q = torch.nn.ModuleList()
        for _ in range(num_slices_q):
            self.proj_heads_q.append(
                torch.nn.Linear(dim_embed_q, num_heads_r * self.dim_head_proj, bias=False)
            )
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads_r * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads_r * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        assert with_flash, "Only flash attention supported at the moment"

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        if self.with_residual:
            x_q_in = x_q
        x_q = self.lnorm_in_q(x_q) if ada_ln_aux is None else self.lnorm_in_q(x_q, ada_ln_aux)
        x_kv = self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], self.num_heads, self.dim_head_proj]
        qs = [
            self.lnorm_q(head_proj(x_q_i).reshape(s)).to(self.dtype)
            for head_proj, x_q_i in zip(self.proj_heads_q, x_q.transpose(1, 0), strict=False)
        ]
        s = [x_kv.shape[0], self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x_kv).reshape(s).to(self.dtype)

        # set dropout rate according to training/eval mode as required by flash_attn
        # dropout_rate = self.dropout_rate if self.training else 0.0

        zero_q_idx = _zero_length_segment_indices(x_q_lens)
        zero_kv_idx = _zero_length_segment_indices(x_kv_lens)
        if zero_q_idx.numel() > 0:
            print(
                f"[flash_attn cross varlen sliced-q] zero_length_q_segments count={zero_q_idx.numel()} first_indices={zero_q_idx[:10].tolist()}",
                flush=True,
            )
        if zero_kv_idx.numel() > 0:
            print(
                f"[flash_attn cross varlen sliced-q] zero_length_kv_segments count={zero_kv_idx.numel()} first_indices={zero_kv_idx[:10].tolist()}",
                flush=True,
            )
        filtered_x_q_lens, keep_q_mask = _filter_varlen_lengths(x_q_lens)
        filtered_x_kv_lens, keep_kv_mask = _filter_varlen_lengths(x_kv_lens)
        qs = [_filter_varlen_tokens(qs_i, x_q_lens, keep_q_mask) for qs_i in qs]
        ks = _filter_varlen_tokens(ks, x_kv_lens, keep_kv_mask)
        vs = _filter_varlen_tokens(vs, x_kv_lens, keep_kv_mask)
        cum_x_q_lens = torch.cumsum(filtered_x_q_lens, 0, dtype=torch.int32)
        cum_x_kv_lens = torch.cumsum(filtered_x_kv_lens, 0, dtype=torch.int32)
        print(
            "[flash_attn cross varlen sliced-q] "
            f"num_q_slices={len(qs)} k_shape={tuple(ks.shape)} v_shape={tuple(vs.shape)} "
            f"first_q_shape={tuple(qs[0].shape) if qs else 'empty'} "
            f"first_q_dtype={qs[0].dtype if qs else 'empty'} k_dtype={ks.dtype} v_dtype={vs.dtype} "
            f"x_q_lens_shape={tuple(filtered_x_q_lens.shape)} x_q_lens_dtype={filtered_x_q_lens.dtype} "
            f"x_kv_lens_shape={tuple(filtered_x_kv_lens.shape)} x_kv_lens_dtype={filtered_x_kv_lens.dtype} "
            f"cum_q_shape={tuple(cum_x_q_lens.shape)} cum_q_dtype={cum_x_q_lens.dtype} cum_q_stride={cum_x_q_lens.stride()} "
            f"cum_kv_shape={tuple(cum_x_kv_lens.shape)} cum_kv_dtype={cum_x_kv_lens.dtype} cum_kv_stride={cum_x_kv_lens.stride()} "
            f"cum_q_first={cum_x_q_lens[0].item() if cum_x_q_lens.numel() else 'empty'} "
            f"cum_q_last={cum_x_q_lens[-1].item() if cum_x_q_lens.numel() else 'empty'} "
            f"cum_kv_first={cum_x_kv_lens[0].item() if cum_x_kv_lens.numel() else 'empty'} "
            f"cum_kv_last={cum_x_kv_lens[-1].item() if cum_x_kv_lens.numel() else 'empty'} "
            f"first_q_tokens={qs[0].shape[0] if qs else 'empty'} k_tokens={ks.shape[0]} v_tokens={vs.shape[0]} "
            f"cum_q_diffs_min={torch.diff(cum_x_q_lens).min().item() if cum_x_q_lens.numel() > 1 else 'n/a'} "
            f"cum_kv_diffs_min={torch.diff(cum_x_kv_lens).min().item() if cum_x_kv_lens.numel() > 1 else 'n/a'}",
            flush=True,
        )
        outs = []
        for _i, qs_i in enumerate(qs):
            outs += [
                flash_attn_varlen_func(
                    q=qs_i,
                    k=ks,
                    v=vs,
                    cu_seqlens_q=cum_x_q_lens,
                    cu_seqlens_k=cum_x_kv_lens,
                )[0]
            ]

        outs = self.proj_out(torch.stack(outs).transpose(1, 0).flatten(-2, -1))
        if self.with_residual:
            outs = x_q_in + outs.reshape(x_q_in.shape)

        return outs


class MultiSelfAttentionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        softcap=0.0,
        norm_type="LayerNorm",
        qk_norm_type=None,
        dim_aux=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
        with_2d_rope=False,
    ):
        super(MultiSelfAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_flash = with_flash
        self.softcap = softcap
        self.dropout_rate = dropout_rate
        self.with_residual = with_residual
        self.with_2d_rope = with_2d_rope

        assert dim_embed % num_heads == 0
        self.dim_head_proj = dim_embed // num_heads if dim_head_proj is None else dim_head_proj

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        if dim_aux is not None:
            self.lnorm = AdaLayerNorm(dim_embed, dim_aux, norm_eps=norm_eps)
        else:
            self.lnorm = norm(dim_embed, eps=norm_eps)
        self.proj_heads_q = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_v = torch.nn.Linear(dim_embed, num_heads * self.dim_head_proj, bias=False)
        self.proj_out = torch.nn.Linear(dim_embed, dim_embed, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        if with_flash:
            self.att = torch.nn.functional.scaled_dot_product_attention
        else:
            self.att = self.attention
            self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, coords=None, ada_ln_aux=None):
        if self.with_residual:
            x_in = x
        x = self.lnorm(x) if ada_ln_aux is None else self.lnorm(x, ada_ln_aux)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [*([x.shape[0], 1] if len(x.shape) == 2 else x.shape[:-1]), self.num_heads, -1]
        qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s).to(self.dtype)

        if self.with_2d_rope:
            if coords is None:
                raise ValueError("coords must be provided when with_2d_rope=True")
            qs, ks = rotary_pos_emb_2d(qs, ks, coords, unsqueeze_dim=2)

        # set dropout rate according to training/eval mode as required by flash_attn
        # dropout_rate = self.dropout_rate if self.training else 0.0

        # ordering of tensors (seq, heads, embed) (which differs from torch's flash attention implt)
        outs, _ = flash_attn_func(qs, ks, vs, softcap=self.softcap)  # , dropout_p=dropout_rate)

        out = self.proj_out(outs.flatten(-2, -1))
        if self.with_residual:
            out = out + x_in

        return out


class MultiCrossAttentionHead(torch.nn.Module):
    def __init__(
        self,
        dim_embed_q,
        dim_embed_kv,
        num_heads,
        dim_head_proj=None,
        dropout_rate=0.0,
        with_residual=True,
        with_qk_lnorm=True,
        with_flash=True,
        norm_type="LayerNorm",
        qk_norm_type=None,
        norm_eps=1e-5,
        attention_dtype=torch.bfloat16,
    ):
        super(MultiCrossAttentionHead, self).__init__()

        self.num_heads = num_heads
        self.with_residual = with_residual
        self.with_flash = with_flash

        if norm_type == "LayerNorm":
            norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            norm = RMSNorm

        assert dim_embed_q % num_heads == 0
        self.dim_head_proj = dim_embed_q // num_heads if dim_head_proj is None else dim_head_proj

        self.lnorm_in_q = norm(dim_embed_q, eps=norm_eps)
        self.lnorm_in_kv = norm(dim_embed_kv, eps=norm_eps)

        self.proj_heads_q = torch.nn.Linear(dim_embed_q, num_heads * self.dim_head_proj, bias=False)
        self.proj_heads_k = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )
        self.proj_heads_v = torch.nn.Linear(
            dim_embed_kv, num_heads * self.dim_head_proj, bias=False
        )

        self.proj_out = torch.nn.Linear(self.dim_head_proj * num_heads, dim_embed_q, bias=False)
        self.dropout = (
            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else torch.nn.Identity()
        )

        qk_norm_type = qk_norm_type or norm_type
        if qk_norm_type == "LayerNorm":
            qk_norm = partial(torch.nn.LayerNorm, elementwise_affine=False, eps=norm_eps)
        else:
            qk_norm = RMSNorm
        lnorm = qk_norm if with_qk_lnorm else torch.nn.Identity
        self.lnorm_q = lnorm(self.dim_head_proj, eps=norm_eps)
        self.lnorm_k = lnorm(self.dim_head_proj, eps=norm_eps)

        self.dtype = attention_dtype
        self.att = torch.nn.functional.scaled_dot_product_attention
        self.softmax = torch.nn.Softmax(dim=-1)

    #########################################
    def forward(self, x_q, x_kv):
        if self.with_residual:
            x_q_in = x_q
        x_q, x_kv = self.lnorm_in_q(x_q), self.lnorm_in_kv(x_kv)

        # project onto heads and q,k,v and
        # ensure these are 4D tensors as required for flash attention
        s = [x_q.shape[0], -1, self.num_heads, self.dim_head_proj]
        qs = self.lnorm_q(self.proj_heads_q(x_q).reshape(s)).to(self.dtype).transpose(-3, -2)
        s = [x_kv.shape[0], -1, self.num_heads, self.dim_head_proj]
        ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype).transpose(-3, -2)
        vs = self.proj_heads_v(x_kv).reshape(s).transpose(-3, -2)

        # correct ordering of tensors with seq dimension second but last is critical
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            outs = self.att(qs, ks, vs).transpose(2, 1)

        outs = self.dropout(self.proj_out(outs.flatten(-2, -1)))
        if self.with_residual:
            outs = x_q_in + outs

        return outs
