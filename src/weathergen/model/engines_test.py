from types import SimpleNamespace

import pytest
import torch

from weathergen.model.engines import EmbeddingEngine
from weathergen.utils.sync_counter import SyncCounter


def _reference_pe_idxs(tokens_lens):
    """Boolean-mask formulation the searchsorted one replaced (syncs on a device)."""
    tok_counts = tokens_lens.permute([2, 0, 1, 3]).sum(0).flatten()
    rows = torch.arange(tok_counts.max()).unsqueeze(0).expand(tok_counts.shape[0], -1)
    return rows[rows < tok_counts.unsqueeze(1)]


def _reference_scatter_idxs(tokens_lens, streams_active):
    tok_counts = tokens_lens[:, :, streams_active].permute([2, 0, 1, 3]).flatten(1, -1)
    pad = torch.zeros((1, tok_counts.shape[1]), dtype=torch.int64)
    offset = torch.cat([pad, tok_counts.cumsum(0)])[:-1]
    offset[:, 1:] += tok_counts.sum(0).cumsum(0)[:-1]
    ranges = torch.arange(tok_counts.max()).repeat((tok_counts.numel(), 1))
    idxs = (offset.flatten() + ranges.transpose(1, 0)).transpose(1, 0)
    col_indices = torch.arange(idxs.shape[1]).unsqueeze(0)
    valid_mask = col_indices < tok_counts.flatten().unsqueeze(1)
    return idxs[valid_mask].to(torch.int64)


def _random_tokens_lens(seed, num_streams, empty_stream=None):
    """(input steps, samples, streams, cells) with many empty cells."""
    gen = torch.Generator().manual_seed(seed)
    lens = torch.randint(0, 5, (2, 3, num_streams, 12), generator=gen, dtype=torch.int32)
    lens[torch.rand(lens.shape, generator=gen) < 0.4] = 0
    if empty_stream is not None:
        lens[:, :, empty_stream] = 0
    return lens


def _engine(stream_names, inactive=()):
    embeds = {
        name: torch.nn.Identity() if name in inactive else torch.nn.Linear(1, 1)
        for name in stream_names
    }
    return SimpleNamespace(streams=dict.fromkeys(stream_names), embeds=embeds)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("empty_stream", [None, 1])
def test_pe_idxs_match_the_boolean_mask_formulation(seed, empty_stream):
    tokens_lens = _random_tokens_lens(seed, num_streams=3, empty_stream=empty_stream)
    batch = SimpleNamespace(tokens_lens=tokens_lens)

    pe_idxs = EmbeddingEngine.get_pe_idxs_vectorized(None, batch, int(tokens_lens.sum()))

    assert torch.equal(pe_idxs, _reference_pe_idxs(tokens_lens))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("inactive", [(), ("b",)])
def test_scatter_idxs_match_the_boolean_mask_formulation(seed, inactive):
    names = ["a", "b", "c"]
    tokens_lens = _random_tokens_lens(seed, num_streams=3)
    engine = _engine(names, inactive)
    batch = SimpleNamespace(tokens_lens=tokens_lens, get_device=lambda: "cpu")
    active = [i for i, n in enumerate(names) if n not in inactive]
    num_tokens = int(tokens_lens[:, :, active].sum())

    scatter_idxs = EmbeddingEngine.get_scatter_idxs_vectorized(engine, batch, num_tokens)

    assert torch.equal(scatter_idxs, _reference_scatter_idxs(tokens_lens, active))
    assert torch.equal(scatter_idxs.sort().values, torch.arange(num_tokens)), "a permutation"


def test_index_builders_do_not_sync():
    tokens_lens = _random_tokens_lens(0, num_streams=3)
    num_tokens = int(tokens_lens.sum())
    batch = SimpleNamespace(tokens_lens=tokens_lens, get_device=lambda: "cpu")
    engine = _engine(["a", "b", "c"])

    with SyncCounter({"cpu"}) as counter:
        EmbeddingEngine.get_pe_idxs_vectorized(engine, batch, num_tokens)
        EmbeddingEngine.get_scatter_idxs_vectorized(engine, batch, num_tokens)

    assert counter.report()["totals"] == {}
