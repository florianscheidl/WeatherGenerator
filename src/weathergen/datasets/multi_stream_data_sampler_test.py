# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the RNG seed derivation and the spatial cell partition of the data sampler."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets import multi_stream_data_sampler as msds
from weathergen.datasets.masking import MaskData
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.datasets.utils import cell_partition_mask


def _derive(base_seed: int, rank: int, mini_epoch: int) -> np.random.SeedSequence:
    """Derive a seed off a stub, avoiding the cost of a full sampler instance.

    `rank` is the data shard index; without spatial parallelism they coincide.
    """
    stub = SimpleNamespace(rng_base_seed=base_seed, data_shard_idx=rank, mini_epoch=mini_epoch)
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
    stub = SimpleNamespace(rng_base_seed=26, data_shard_idx=0, mini_epoch=0)

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


def _partition_stub(healpix_level: int, rank: int, world_size: int) -> SimpleNamespace:
    """Stub carrying only what _init_cell_partition reads."""
    return SimpleNamespace(
        healpix_level=healpix_level,
        num_healpix_cells=12 * 4**healpix_level,
        rank=rank,
        world_size=world_size,
    )


def test_cell_partition_disabled_without_config():
    """Run configs predating spatial parallelism must be unaffected."""
    cf = OmegaConf.create({"healpix_level": 5})

    assert MultiStreamDataSampler._init_cell_partition(_partition_stub(5, 0, 4), cf) is None


def test_cell_partition_disabled_by_explicit_null():
    cf = OmegaConf.create({"spatial_parallel": {"healpix_level_split": None}})

    assert MultiStreamDataSampler._init_cell_partition(_partition_stub(5, 0, 4), cf) is None


def test_cell_partition_built_from_rank_and_world_size():
    cf = OmegaConf.create({"spatial_parallel": {"healpix_level_split": 3}})

    partitions = [
        MultiStreamDataSampler._init_cell_partition(_partition_stub(5, rank, 4), cf)
        for rank in range(4)
    ]

    for partition in partitions:
        assert int(partition.sum()) == (12 * 4**5) // 4
    assert np.array([p.numpy() for p in partitions]).sum(axis=0).tolist() == [1] * (12 * 4**5)


def test_get_source_target_masks_restricts_both_sides():
    """Ownership must be applied to source and target masks alike."""
    ones = torch.ones(12 * 4**1, dtype=torch.bool)
    target_masks, source_masks = MaskData(), MaskData()
    target_masks.add_mask(ones.clone(), {}, {}, [], 0, None, None)
    source_masks.add_mask(ones.clone(), {}, {}, [], 0, None, None)

    partition = cell_partition_mask(1, 0, num_shards=4, shard_idx=0)
    stub = SimpleNamespace(
        streams_datasets={"S": SimpleNamespace(info={"name": "S"})},
        num_healpix_cells=12 * 4**1,
        cell_partition=partition,
        tokenizer=SimpleNamespace(
            build_samples_for_stream=lambda *_: (target_masks, source_masks, np.array([0]))
        ),
    )

    masks, _, _, _ = MultiStreamDataSampler._get_source_target_masks(stub, "masking")

    assert masks["S"][0].masks[0].tolist() == partition.tolist()
    assert masks["S"][1].masks[0].tolist() == partition.tolist()


def test_get_source_target_masks_unchanged_when_partition_disabled():
    ones = torch.ones(12 * 4**1, dtype=torch.bool)
    target_masks, source_masks = MaskData(), MaskData()
    target_masks.add_mask(ones.clone(), {}, {}, [], 0, None, None)
    source_masks.add_mask(ones.clone(), {}, {}, [], 0, None, None)

    stub = SimpleNamespace(
        streams_datasets={"S": SimpleNamespace(info={"name": "S"})},
        num_healpix_cells=12 * 4**1,
        cell_partition=None,
        tokenizer=SimpleNamespace(
            build_samples_for_stream=lambda *_: (target_masks, source_masks, np.array([0]))
        ),
    )

    masks, _, _, _ = MultiStreamDataSampler._get_source_target_masks(stub, "masking")

    assert masks["S"][0].masks[0].all()
    assert masks["S"][1].masks[0].all()


def test_get_source_target_masks_keeps_unrestricted_copies():
    """The unrestricted masks drive the rank-independent skip decision."""
    ones = torch.ones(12 * 4**1, dtype=torch.bool)
    target_masks, source_masks = MaskData(), MaskData()
    target_masks.add_mask(ones.clone(), {}, {}, [], 0, None, None)
    source_masks.add_mask(ones.clone(), {}, {}, [], 0, None, None)

    stub = SimpleNamespace(
        streams_datasets={"S": SimpleNamespace(info={"name": "S"})},
        num_healpix_cells=12 * 4**1,
        cell_partition=cell_partition_mask(1, 0, num_shards=4, shard_idx=0),
        tokenizer=SimpleNamespace(
            build_samples_for_stream=lambda *_: (target_masks, source_masks, np.array([0]))
        ),
    )

    _, masks_global, _, _ = MultiStreamDataSampler._get_source_target_masks(stub, "masking")

    # restriction must not have reached back into the copies
    assert masks_global["S"][0][0].all()
    assert masks_global["S"][1][0].all()


# --- data shard split ------------------------------------------------------------------


def _shard_stub(cell_partition, rank: int, world_size: int) -> SimpleNamespace:
    stub = SimpleNamespace(cell_partition=cell_partition, rank=rank, world_size=world_size)
    spatially_parallel = cell_partition is not None
    stub.num_data_shards = 1 if spatially_parallel else world_size
    stub.data_shard_idx = 0 if spatially_parallel else rank
    return stub


def test_spatial_ranks_walk_the_same_temporal_indices():
    """Ranks are shards of one sample, so they must not split the index range."""
    partition = cell_partition_mask(5, 3, num_shards=4, shard_idx=2)
    stub = _shard_stub(partition, rank=2, world_size=4)
    stub.len = 64

    starts = []
    for rank in range(4):
        s = _shard_stub(partition, rank=rank, world_size=4)
        s.len = 64
        starts.append(s.data_shard_idx * s.len)

    assert starts == [0, 0, 0, 0]
    assert stub.num_data_shards == 1


def test_data_parallel_ranks_still_split_the_index_range():
    """Without a cell partition the behaviour is unchanged."""
    starts = []
    for rank in range(4):
        s = _shard_stub(None, rank=rank, world_size=4)
        s.len = 64
        starts.append(s.data_shard_idx * s.len)

    assert starts == [0, 64, 128, 192]


def test_seed_is_shared_across_spatial_ranks(worker_id):
    """All spatial ranks must draw the same masks and the same temporal permutation."""
    worker_id(0)
    partition = cell_partition_mask(5, 3, num_shards=4, shard_idx=0)

    draws = []
    for rank in range(4):
        stub = _shard_stub(partition, rank=rank, world_size=4)
        stub.rng_base_seed = 26
        stub.mini_epoch = 0
        draws.append(_first_draw(MultiStreamDataSampler._derive_rng_seed(stub)))

    assert len(set(draws)) == 1


# --- global validity verdict -----------------------------------------------------------


def _verdict(occupancy, mode="masking"):
    return MultiStreamDataSampler._batch_valid_globally(None, occupancy, mode)


def test_validity_verdict_is_none_without_spatial_parallelism():
    """None hands the decision back to the per-rank predicates, as before."""
    assert _verdict(None) is None


def test_validity_verdict_accepts_a_populated_batch():
    # (sources_nonempty, targets_nonempty, sources_nan, targets_nan) per stream
    assert _verdict([(True, True, False, False)]) is True


def test_validity_verdict_rejects_globally_empty_sources():
    assert _verdict([(False, True, False, False)]) is False


def test_validity_verdict_rejects_empty_targets_only_when_masking():
    assert _verdict([(True, False, False, False)], mode="masking") is False
    assert _verdict([(True, False, False, False)], mode="student_teacher") is True


def test_validity_verdict_needs_all_streams_nan_to_reject():
    assert _verdict([(True, True, True, False), (True, True, False, False)]) is True
    assert _verdict([(True, True, True, False), (True, True, True, False)]) is False


def test_validity_verdict_accepts_when_one_stream_carries_the_batch():
    """Emptiness is an any() over streams, matching BatchSamples."""
    assert _verdict([(False, False, False, False), (True, True, False, False)]) is True


# --- cell occupancy --------------------------------------------------------------------


def test_cells_with_tokens_marks_populated_cells():
    num_cells = 12 * 4**1
    lens = [[] for _ in range(num_cells)]
    lens[3] = [8, 8]
    windows = [(None, lens)]

    occupied = msds.cells_with_tokens(windows, num_cells)

    assert np.flatnonzero(occupied).tolist() == [3]


def test_cells_with_tokens_unions_over_windows():
    num_cells = 12 * 4**1
    first = [[] for _ in range(num_cells)]
    first[1] = [8]
    second = [[] for _ in range(num_cells)]
    second[5] = [8]

    occupied = msds.cells_with_tokens([(None, first), (None, second)], num_cells)

    assert np.flatnonzero(occupied).tolist() == [1, 5]


def test_cells_with_tokens_skips_empty_windows():
    num_cells = 12 * 4**1

    occupied = msds.cells_with_tokens([(None, None)], num_cells)

    assert not occupied.any()


def test_windows_all_nan():
    full = SimpleNamespace(data=np.full((4, 2), np.nan))
    mixed = SimpleNamespace(data=np.array([[np.nan, 1.0]]))
    empty = SimpleNamespace(data=np.zeros((0, 2)))

    assert msds.windows_all_nan([full, full]) is True
    assert msds.windows_all_nan([full, mixed]) is False
    # a window without data says nothing about NaN; emptiness is judged separately
    assert msds.windows_all_nan([empty]) is False
    assert msds.windows_all_nan([empty, full]) is True


def test_schedule_stream_is_insulated_from_data_draws(worker_id):
    """Mirrors the spawn(2) split in reset().

    The number of draws the data stream makes depends on how many points a reader
    returned. Once ranks load different amounts of data (reader-level cell filtering),
    a shared generator would shift the schedule and the masks along with it, and the
    ranks would silently stop agreeing on which cells and which windows they are
    covering. Spawning two streams is what prevents that.
    """
    worker_id(0)
    stub = _shard_stub(cell_partition_mask(5, 3, 4, 0), rank=0, world_size=4)
    stub.rng_base_seed = 26
    stub.mini_epoch = 0

    def schedule_after(n_data_draws: int) -> list:
        seed_data, seed_schedule = MultiStreamDataSampler._derive_rng_seed(stub).spawn(2)
        rng, rng_schedule = (
            np.random.default_rng(seed_data),
            np.random.default_rng(seed_schedule),
        )
        for _ in range(n_data_draws):
            rng.random(size=1234)
        return rng_schedule.permutation(np.arange(32)).tolist()

    assert schedule_after(0) == schedule_after(97)


def test_data_and_schedule_streams_are_independent(worker_id):
    worker_id(0)
    stub = _shard_stub(None, rank=0, world_size=1)
    stub.rng_base_seed = 26
    stub.mini_epoch = 0

    seed_data, seed_schedule = MultiStreamDataSampler._derive_rng_seed(stub).spawn(2)

    assert (
        np.random.default_rng(seed_data).random() != np.random.default_rng(seed_schedule).random()
    )
