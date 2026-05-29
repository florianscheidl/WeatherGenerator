import sys
import types
from collections import defaultdict

import torch
from omegaconf import OmegaConf

if "flash_attn" not in sys.modules:
    flash_attn = types.ModuleType("flash_attn")

    def _unused_flash_attn(*args, **kwargs):
        raise RuntimeError("flash attention stub should not be called in FSDP dtype tests")

    flash_attn.flash_attn_func = _unused_flash_attn
    flash_attn.flash_attn_varlen_func = _unused_flash_attn
    sys.modules["flash_attn"] = flash_attn

from weathergen.model.embeddings import StreamEmbedTransformer
from weathergen.model.engines import (
    BilinearDecoder,
    EnsPredictionHead,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    TargetPredictionEngine,
)
from weathergen.model.parametrised_prob_dist import LatentInterpolator


def _param_dtypes(module):
    return {param.dtype for param in module.parameters()}


def _minimal_cfg(decoder_type: str = "CrossAttentionConditioning"):
    return OmegaConf.create(
        {
            "with_flash_attention": True,
            "norm_type": "LayerNorm",
            "qk_norm_type": "LayerNorm",
            "norm_eps": 1e-5,
            "mlp_norm_eps": 1e-5,
            "mixed_precision_dtype": "bf16",
            "attention_dtype": "bf16",
            "decoder_type": decoder_type,
            "pred_self_attention": True,
            "pred_mlp_adaln": False,
            "ae_global_dim_embed": 16,
            "streams": [{"target_readout": {"num_heads": 2}, "name": "stream1"}],
        }
    )


def test_stream_embed_transformer_parameters_are_uniform_dtype():
    module = StreamEmbedTransformer(
        mode="channels",
        num_tokens=1,
        token_size=2,
        num_channels=2,
        dim_embed=8,
        dim_out=8,
        num_blocks=1,
        num_heads=2,
        dtype=torch.bfloat16,
    )
    assert _param_dtypes(module) == {torch.bfloat16}



def test_target_prediction_engine_parameters_are_uniform_dtype():
    module = TargetPredictionEngine(
        cf=_minimal_cfg(),
        dims_embed=[8, 8],
        dim_coord_in=3,
        tr_dim_head_proj=4,
        tr_mlp_hidden_factor=2,
        softcap=0.0,
        stream_name="stream1",
    )
    assert _param_dtypes(module) == {torch.bfloat16}



def test_latent_heads_and_bilinear_decoder_parameters_are_uniform_dtype():
    transformer = LatentPredictionHeadTransformer(
        _minimal_cfg(),
        "latent-transformer",
        in_dim=8,
        loss_conf=OmegaConf.create(
            {
                "out_dim": 4,
                "num_blocks": 1,
                "num_heads": 2,
                "with_qk_lnorm": True,
                "intermediate_dim": 8,
                "dropout_rate": 0.0,
            }
        ),
        use_class_token=True,
        use_patch_token=False,
    )
    mlp = LatentPredictionHeadMLP(
        "latent-mlp",
        8,
        OmegaConf.create({"out_dim": 4, "num_layers": 2, "hidden_factor": 2}),
        use_class_token=True,
        use_patch_token=False,
        dtype=torch.bfloat16,
    )
    pred_head = EnsPredictionHead(8, 4, 2, 2, stream_name="stream1", dtype=torch.bfloat16)
    decoder = BilinearDecoder("stream1", 3, 8, 2, dtype=torch.bfloat16)

    assert _param_dtypes(transformer) == {torch.bfloat16}
    assert _param_dtypes(mlp) == {torch.bfloat16}
    assert _param_dtypes(pred_head) == {torch.bfloat16}
    assert _param_dtypes(decoder) == {torch.bfloat16}



def test_latent_interpolator_parameters_are_uniform_dtype():
    module = LatentInterpolator(gamma=1.0, dim=8, dtype=torch.bfloat16)
    assert _param_dtypes(module) == {torch.bfloat16}
