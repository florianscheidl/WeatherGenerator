import sys
import types

import torch
from omegaconf import OmegaConf

if "flash_attn" not in sys.modules:
    flash_attn = types.ModuleType("flash_attn")

    def _unused_flash_attn(*args, **kwargs):
        raise RuntimeError("flash attention stub should not be called in dtype constructor tests")

    flash_attn.flash_attn_func = _unused_flash_attn
    flash_attn.flash_attn_varlen_func = _unused_flash_attn
    sys.modules["flash_attn"] = flash_attn

from weathergen.model.blocks import CrossAttentionBlock, SelfAttentionBlock
from weathergen.model.embeddings import StreamEmbedLinear, StreamEmbedTransformer
from weathergen.model.engines import (
    BilinearDecoder,
    EnsPredictionHead,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    Local2GlobalSumEngine,
    TargetPredictionEngine,
)


def _first_linear(module):
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            return submodule
    raise AssertionError("No Linear module found")


def _minimal_decoder_cfg(decoder_type: str = "CrossAttentionConditioning"):
    return OmegaConf.create(
        {
            "with_flash_attention": True,
            "norm_type": "LayerNorm",
            "qk_norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "mixed_precision_dtype": "fp16",
            "attention_dtype": "fp16",
            "decoder_type": decoder_type,
            "pred_self_attention": True,
            "pred_mlp_adaln": False,
            "ae_global_dim_embed": 16,
            "streams": [{"target_readout": {"num_heads": 2}, "name": "stream1"}],
        }
    )


def test_stream_embed_modules_use_requested_dtype():
    module = StreamEmbedTransformer(
        mode="channels",
        num_tokens=1,
        token_size=2,
        num_channels=2,
        dim_embed=8,
        dim_out=8,
        num_blocks=1,
        num_heads=2,
        dtype=torch.float16,
    )

    assert module.embed.weight.dtype == torch.float16
    assert _first_linear(module.layers[0]).weight.dtype == torch.float16
    assert _first_linear(module.layers[1]).weight.dtype == torch.float16



def test_stream_embed_linear_uses_requested_dtype():
    module = StreamEmbedLinear(4, 8, dtype=torch.float16)

    assert module.layer.weight.dtype == torch.float16



def test_local2global_sum_engine_projection_uses_mixed_precision_dtype():
    cfg = OmegaConf.create(
        {
            "ae_local_dim_embed": 8,
            "ae_global_dim_embed": 16,
            "ae_adapter_num_blocks": 1,
            "ae_adapter_dropout_rate": 0.0,
            "norm_type": "LayerNorm",
            "mlp_norm_eps": 1e-5,
            "mixed_precision_dtype": "fp16",
        }
    )

    module = Local2GlobalSumEngine(cfg)

    assert module.proj.weight.dtype == torch.float16



def test_prediction_head_modules_use_requested_dtype():
    ens = EnsPredictionHead(8, 4, 2, 2, stream_name="stream1", dtype=torch.float16)
    mlp = LatentPredictionHeadMLP(
        "latent-head",
        8,
        OmegaConf.create({"out_dim": 4, "num_layers": 2, "hidden_factor": 2}),
        use_class_token=True,
        use_patch_token=False,
        dtype=torch.float16,
    )
    bilinear = BilinearDecoder("stream1", 3, 8, 2, dtype=torch.float16)

    assert _first_linear(ens).weight.dtype == torch.float16
    assert _first_linear(mlp).weight.dtype == torch.float16
    assert bilinear.bilin.weight.dtype == torch.float16



def test_target_prediction_engine_uses_mixed_precision_dtype_for_non_attention_modules():
    cfg = _minimal_decoder_cfg("CrossAttentionConditioning")

    module = TargetPredictionEngine(
        cf=cfg,
        dims_embed=[8, 8],
        dim_coord_in=3,
        tr_dim_head_proj=4,
        tr_mlp_hidden_factor=2,
        softcap=0.0,
        stream_name="stream1",
    )

    assert module.output_in_norm.weight.dtype == torch.float16
    assert module.latent_in_norm.weight.dtype == torch.float16
    assert module.pos_embed.dtype == torch.float16
    assert _first_linear(module.tte[0]).weight.dtype == torch.float16



def test_attention_blocks_use_requested_dtype_for_outer_norms_and_mlp():
    attention_kwargs = {
        "with_qk_lnorm": True,
        "with_flash": True,
        "norm_type": "LayerNorm",
        "qk_norm_type": "LayerNorm",
        "softcap": 0.0,
        "dim_aux": 3,
        "norm_eps": 1e-5,
        "attention_dtype": torch.float16,
    }

    self_block = SelfAttentionBlock(
        dim=8,
        dim_aux=3,
        with_adanorm=False,
        num_heads=2,
        dropout_rate=0.0,
        dtype=torch.float16,
        attention_kwargs=attention_kwargs,
    )
    cross_block = CrossAttentionBlock(
        dim_q=8,
        dim_kv=8,
        dim_aux=3,
        with_self_attn=True,
        with_adanorm=False,
        with_mlp=True,
        num_heads=2,
        dropout_rate=0.0,
        dtype=torch.float16,
        attention_kwargs=attention_kwargs,
    )

    assert self_block.ln_sa.weight.dtype == torch.float16
    assert self_block.ln_mlp.weight.dtype == torch.float16
    assert _first_linear(self_block.mlp).weight.dtype == torch.float16
    assert cross_block.ln_sa.weight.dtype == torch.float16
    assert cross_block.ln_ca.weight.dtype == torch.float16
    assert cross_block.ln_mlp.weight.dtype == torch.float16
    assert _first_linear(cross_block.mlp).weight.dtype == torch.float16



def test_latent_prediction_head_transformer_uses_mixed_precision_dtype():
    cfg = OmegaConf.create(
        {
            "with_flash_attention": True,
            "norm_type": "LayerNorm",
            "qk_norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "mixed_precision_dtype": "fp16",
            "attention_dtype": "fp16",
        }
    )
    loss_conf = OmegaConf.create(
        {
            "out_dim": 4,
            "num_blocks": 1,
            "num_heads": 2,
            "with_qk_lnorm": True,
            "intermediate_dim": 8,
            "dropout_rate": 0.0,
        }
    )

    module = LatentPredictionHeadTransformer(
        cfg,
        "latent-transformer",
        in_dim=8,
        loss_conf=loss_conf,
        use_class_token=True,
        use_patch_token=False,
    )

    linear_layers = [m for m in module.modules() if isinstance(m, torch.nn.Linear)]
    assert linear_layers[0].weight.dtype == torch.float16
    assert linear_layers[-1].weight.dtype == torch.float16
