# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at the root of this repository.

import importlib.util
import sys
import types
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

if importlib.util.find_spec("flash_attn") is None:
    flash_attn = types.ModuleType("flash_attn")

    def unavailable_flash_attention(*args, **kwargs):
        raise AssertionError("this CPU test must not execute flash attention")

    flash_attn.flash_attn_func = unavailable_flash_attention
    flash_attn.flash_attn_varlen_func = unavailable_flash_attention
    sys.modules["flash_attn"] = flash_attn

from weathergen.model.engines import EmbeddingEngine  # noqa: E402


def test_embedding_engine_empty_stream_runs_zero_gradient_dummy() -> None:
    config = OmegaConf.create(
        {
            "mixed_precision_dtype": "fp32",
            "ae_local_dim_embed": 4,
            "streams": {
                "stream": {
                    "diagnostic": False,
                    "token_size": 2,
                    "embed": {"net": "linear"},
                }
            },
        }
    )
    engine = EmbeddingEngine(config, sources_size=[1])
    stream_data = SimpleNamespace(source_tokens_cells=[torch.empty((0, 2, 1))])
    sample = SimpleNamespace(streams_data={"stream": stream_data})
    batch = SimpleNamespace(
        tokens_lens=torch.zeros((1, 1, 1, 4), dtype=torch.int32),
        get_num_source_steps=lambda: 1,
        get_samples=lambda: [sample],
        get_device=lambda: torch.device("cpu"),
    )

    output = checkpoint(engine, batch, torch.zeros((1, 4)), use_reentrant=False)
    output.sum().backward()

    assert output.shape == (0, 4)
    for parameter in engine.embeds["stream"].parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_embedding_engine_dummy_does_not_enter_packed_tokens() -> None:
    stream_config = {
        "diagnostic": False,
        "token_size": 2,
        "embed": {"net": "linear"},
    }
    config = OmegaConf.create(
        {
            "mixed_precision_dtype": "fp32",
            "ae_local_dim_embed": 4,
            "streams": {"empty": stream_config, "real": stream_config},
        }
    )
    engine = EmbeddingEngine(config, sources_size=[1, 1])
    real_tokens = torch.tensor([[[2.0], [3.0]]])
    sample = SimpleNamespace(
        streams_data={
            "empty": SimpleNamespace(source_tokens_cells=[torch.empty((0, 2, 1))]),
            "real": SimpleNamespace(source_tokens_cells=[real_tokens]),
        }
    )
    token_lens = torch.zeros((1, 1, 2, 4), dtype=torch.int32)
    token_lens[0, 0, 1, 0] = 1
    batch = SimpleNamespace(
        tokens_lens=token_lens,
        get_num_source_steps=lambda: 1,
        get_samples=lambda: [sample],
        get_device=lambda: torch.device("cpu"),
    )
    pe_embed = torch.zeros((1, 4))

    output = engine(batch, pe_embed)
    expected = engine.embeds["real"](real_tokens).flatten(0, 1)
    output.sum().backward()

    torch.testing.assert_close(output, expected)
    for parameter in engine.embeds["empty"].parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
