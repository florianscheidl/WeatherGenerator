# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.nn as nn

from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.layers import MLP
from weathergen.model.norms import AdaLayerNormLayer, RMSNorm
from weathergen.utils.utils import get_dtype


def _make_norm(dim, norm_type: str, norm_eps: float, dtype=None):
    if norm_type == "LayerNorm":
        return nn.LayerNorm(dim, eps=norm_eps)
    kwargs = {"dtype": dtype} if dtype is not None else {}
    return RMSNorm(dim, eps=norm_eps, **kwargs)


class SelfAttentionBlock(nn.Module):
    """
    A self attention block, i.e., adaptive layer norm with multi head self attenttion and adaptive
    layer norm with a FFN.
    """

    def __init__(self, dim, dim_aux, with_adanorm, num_heads, dropout_rate, **kwargs):
        super().__init__()

        self.with_adanorm = with_adanorm
        attention_kwargs = kwargs["attention_kwargs"]
        norm_type = attention_kwargs.get("norm_type", "RMSNorm")
        norm_eps = attention_kwargs.get("norm_eps", 1e-6)
        attention_dtype = attention_kwargs.get("attention_dtype", torch.bfloat16)

        self.mhsa = MultiSelfAttentionHeadVarlen(
            dim_embed=dim,
            num_heads=num_heads,
            with_residual=False,
            **attention_kwargs,
        )
        if self.with_adanorm:
            self.mhsa_block = AdaLayerNormLayer(
                dim,
                dim_aux,
                layer=self.mhsa,
                norm_type=norm_type,
                norm_eps=norm_eps,
                dropout_rate=dropout_rate,
                dtype=attention_dtype,
            )
        else:
            self.ln_sa = _make_norm(dim, norm_type, norm_eps, dtype=attention_dtype)
            self.mhsa_block = lambda x, _, **kw: self.mhsa(self.ln_sa(x), **kw) + x

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(
            dim_in=dim,
            dim_out=dim,
            hidden_factor=4,
            dropout_rate=0.1,
            nonlin=approx_gelu,
            with_residual=False,
            norm_type=norm_type,
            norm_eps=norm_eps,
        )
        if self.with_adanorm:
            self.mlp_fn = lambda x, **kw: self.mlp(x)
            self.mlp_block = AdaLayerNormLayer(
                dim,
                dim_aux,
                layer=self.mlp_fn,
                norm_type=norm_type,
                norm_eps=norm_eps,
                dropout_rate=dropout_rate,
                dtype=attention_dtype,
            )
        else:
            self.ln_mlp = _make_norm(dim, norm_type, norm_eps, dtype=attention_dtype)
            self.mlp_block = lambda x, _, **kw: self.mlp(self.ln_mlp(x), None, **kw) + x

        self.initialise_weights()
        if self.with_adanorm:
            # Has to happen after the basic weight init to ensure it is zero!
            self.mhsa_block.initialise_weights()
            self.mlp_block.initialise_weights()

    def initialise_weights(self):
        # Initialise transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, x_lens, aux=None):
        # we have aux_lens as arg to be consistent with the CrossAttentionBlock
        assert self.with_adanorm ^ (aux is None), "Conditioning is not being used"
        x = self.mhsa_block(x, aux, x_lens=x_lens)
        x = self.mlp_block(x, aux)
        return x


class CrossAttentionBlock(nn.Module):
    """
    A cross attention block, i.e., adaptive layer norm with cross attenttion and adaptive layer norm
    with a FFN.
    """

    def __init__(
        self,
        dim_q,
        dim_kv,
        dim_aux,
        with_self_attn,
        with_adanorm,
        with_mlp,
        num_heads,
        dropout_rate,
        **kwargs,
    ):
        super().__init__()

        self.with_adanorm = with_adanorm
        self.with_self_attn = with_self_attn
        self.with_mlp = with_mlp

        attention_kwargs = kwargs["attention_kwargs"]
        norm_type = attention_kwargs.get("norm_type", "RMSNorm")
        norm_eps = attention_kwargs.get("norm_eps", 1e-6)
        attention_dtype = attention_kwargs.get("attention_dtype", torch.bfloat16)

        if with_self_attn:
            self.mhsa = MultiSelfAttentionHeadVarlen(
                dim_embed=dim_q,
                num_heads=num_heads,
                with_residual=False,
                **attention_kwargs,
            )
            if self.with_adanorm:
                self.mhsa_block = AdaLayerNormLayer(
                    dim_q,
                    dim_aux,
                    layer=self.mhsa,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    dropout_rate=dropout_rate,
                    dtype=attention_dtype,
                )
            else:
                self.ln_sa = _make_norm(dim_q, norm_type, norm_eps, dtype=attention_dtype)
                self.mhsa_block = lambda x, _, **kw: self.mhsa(self.ln_sa(x), **kw) + x

        self.cross_attn = MultiCrossAttentionHeadVarlen(
            dim_embed_q=dim_q,
            dim_embed_kv=dim_kv,
            num_heads=num_heads,
            with_residual=False,
            **attention_kwargs,
        )
        if self.with_adanorm:
            self.cross_attn_block = AdaLayerNormLayer(
                dim_q,
                dim_aux,
                layer=self.cross_attn,
                norm_type=norm_type,
                norm_eps=norm_eps,
                dropout_rate=dropout_rate,
                dtype=attention_dtype,
            )
        else:
            self.ln_ca = _make_norm(dim_q, norm_type, norm_eps, dtype=attention_dtype)
            self.cross_attn_block = lambda x, _, **kw: self.cross_attn(self.ln_ca(x), **kw) + x

        if self.with_mlp:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = MLP(
                dim_in=dim_q,
                dim_out=dim_q,
                hidden_factor=4,
                nonlin=approx_gelu,
                with_residual=False,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )
            if self.with_adanorm:
                self.mlp_fn = lambda x, **kw: self.mlp(x)
                self.mlp_block = AdaLayerNormLayer(
                    dim_q,
                    dim_aux,
                    layer=self.mlp_fn,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    dropout_rate=dropout_rate,
                    dtype=attention_dtype,
                )
            else:
                self.ln_mlp = _make_norm(dim_q, norm_type, norm_eps, dtype=attention_dtype)
                self.mlp_block = lambda x, _, **kw: self.mlp(self.ln_mlp(x)) + x
        else:
            self.mlp_block = lambda x, _, **kw: x

        self.initialise_weights()
        if self.with_adanorm:
            # Has to happen after the basic weight init to ensure it is zero!
            self.mhsa_block.initialise_weights()
            self.cross_attn_block.initialise_weights()
            self.mlp_block.initialise_weights()

    def initialise_weights(self):
        # Initialise transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, x_kv, aux, x_kv_lens=None, x_lens=None):
        x = self.cross_attn_block(x, aux, x_kv=x_kv, x_lens=x_lens, x_kv_lens=x_kv_lens)
        if self.with_self_attn:
            x = self.mhsa_block(x, aux, x_lens=x_lens)
        x = self.mlp_block(x, aux, x_lens=x_lens)
        return x


class OriginalPredictionBlock(nn.Module):
    def __init__(
        self,
        config,
        dim_in,
        dim_out,
        dim_kv,
        dim_aux,
        num_heads,
        attention_kwargs,
        tr_dim_head_proj,
        tr_mlp_hidden_factor,
        tro_type,
        mlp_norm_eps=1e-6,
    ):
        super().__init__()

        self.cf = config
        self.tro_type = tro_type
        self.tr_dim_head_proj = tr_dim_head_proj
        self.tr_mlp_hidden_factor = tr_mlp_hidden_factor

        self.block = nn.ModuleList()

        # Multi-Cross Attention Head
        self.block.append(
            MultiCrossAttentionHeadVarlen(
                dim_in,
                self.cf.ae_global_dim_embed,
                self.cf.streams[0]["target_readout"]["num_heads"],
                dim_head_proj=self.tr_dim_head_proj,
                with_residual=True,
                with_qk_lnorm=True,
                dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                with_flash=self.cf.with_flash_attention,
                norm_type=attention_kwargs["norm_type"],
                qk_norm_type=attention_kwargs["qk_norm_type"],
                softcap=0.0,
                dim_aux=dim_aux,
                norm_eps=attention_kwargs["norm_eps"],
                attention_dtype=get_dtype(self.cf.attention_dtype),
            )
        )

        # Optional Self-Attention Head
        if self.cf.pred_self_attention:
            self.block.append(
                MultiSelfAttentionHeadVarlen(
                    dim_in,
                    num_heads=self.cf.streams[0]["target_readout"]["num_heads"],
                    dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                    with_qk_lnorm=True,
                    with_flash=self.cf.with_flash_attention,
                    norm_type=self.cf.norm_type,
                    qk_norm_type=self.cf.qk_norm_type,
                    dim_aux=dim_aux,
                    norm_eps=self.cf.norm_eps,
                    attention_dtype=get_dtype(self.cf.attention_dtype),
                )
            )

        # MLP Block
        self.block.append(
            MLP(
                dim_in,
                dim_out,
                with_residual=True,
                hidden_factor=self.tr_mlp_hidden_factor,
                dropout_rate=0.1,  # Assuming dropout_rate is 0.1
                norm_type=self.cf.norm_type,
                dim_aux=(dim_aux if self.cf.pred_mlp_adaln else None),
                norm_eps=self.cf.mlp_norm_eps,
            )
        )

    def forward(self, latent, output, coords, latent_lens, output_lens):
        for layer in self.block:
            if isinstance(layer, MultiCrossAttentionHeadVarlen):
                output = layer(output, latent, output_lens, latent_lens, coords)
            else:
                output = layer(output, output_lens, coords)
        return output
