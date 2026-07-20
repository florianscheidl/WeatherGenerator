# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the per-rank/worker/mini-epoch RNG seed derivation of the data sampler."""

from types import SimpleNamespace

import numpy as np
import pytest

from weathergen.datasets import multi_stream_data_sampler as msds
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler


def _derive(base_seed: int, rank: int, mini_epoch: int) -> np.random.SeedSequence:
    """Derive a seed off a stub, avoiding the cost of a full sampler instance."""
    stub = SimpleNamespace(rng_base_seed=base_seed, rank=rank, mini_epoch=mini_epoch)
    return MultiStreamDataSampler._derive_rng_seed(stub)


def _first_draw(seed_seq: np.random.SeedSequence) -> float:
    return np.random.default_rng(seed_seq).random()


@pytest.fixture()
def worker_id(monkeypatch):
    """Set the worker id reported by torch's dataloader worker info."""

    def _set(wid: int | None):
        info = None if wid is None else SimpleNamespace(id=wid, num_workers=8)
        monkeypatch.setattr(msds.torch.utils.data, "get_worker_info", lambda: info)

    return _set


def test_seed_is_deterministic(worker_id):
    """The same (base seed, rank, worker, mini epoch) always derives the same stream."""
    worker_id(3)
    assert _first_draw(_derive(26, rank=1, mini_epoch=2)) == _first_draw(
        _derive(26, rank=1, mini_epoch=2)
    )


def test_seed_does_not_accumulate(worker_id):
    """Repeated derivation is a pure function, so re-iterating in place cannot drift.

    Guards the num_workers == 0 case, where no worker process is forked and the
    sampler instance is re-iterated directly.
    """
    worker_id(None)
    stub = SimpleNamespace(rng_base_seed=26, rank=0, mini_epoch=0)

    first = _first_draw(MultiStreamDataSampler._derive_rng_seed(stub))
    for _ in range(5):
        MultiStreamDataSampler._derive_rng_seed(stub)
    last = _first_draw(MultiStreamDataSampler._derive_rng_seed(stub))

    assert first == last


def test_no_worker_info_is_treated_as_worker_zero(worker_id):
    """Running without worker processes matches worker 0, so debug runs stay comparable."""
    worker_id(None)
    without = _first_draw(_derive(26, rank=1, mini_epoch=2))
    worker_id(0)
    with_info = _first_draw(_derive(26, rank=1, mini_epoch=2))

    assert without == with_info


def test_seeds_are_unique_across_ranks_workers_and_mini_epochs(worker_id):
    """Every (rank, worker, mini epoch) combination yields a distinct stream.

    A multiplicative derivation collides here, since distinct triples can share a
    product; the SeedSequence hash is order-sensitive and does not.
    """
    draws = []
    for rank in range(6):
        for wid in range(6):
            worker_id(wid)
            for mini_epoch in range(6):
                draws.append(_first_draw(_derive(26, rank, mini_epoch)))

    assert len(set(draws)) == len(draws)


def test_different_base_seeds_diverge(worker_id):
    """The base seed still controls the stream (guards against it being dropped)."""
    worker_id(0)
    assert _first_draw(_derive(26, rank=0, mini_epoch=0)) != _first_draw(
        _derive(27, rank=0, mini_epoch=0)
    )


def test_base_seed_zero_is_usable(worker_id):
    """Seed 0 is a valid base seed (init_ddp clamps only negatives)."""
    worker_id(0)
    assert _first_draw(_derive(0, rank=0, mini_epoch=0)) != _first_draw(
        _derive(1, rank=0, mini_epoch=0)
    )
