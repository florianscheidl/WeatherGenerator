# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import torch

from weathergen.datasets.healpix_domain import SpatialShard, build_local_healpix_cell_splits
from weathergen.utils import distributed


def test_spatial_shard_tiles_all_cells_without_overlap():
    num_cells, size = 48, 4
    shards = [SpatialShard.for_rank(num_cells, size, rank) for rank in range(size)]

    assert [s.start for s in shards] == [0, 12, 24, 36]
    assert [s.end for s in shards] == [12, 24, 36, 48]
    assert all(s.local_num_cells == 12 for s in shards)
    assert all(s.is_sharded for s in shards)

    covered = [cell for s in shards for cell in range(s.start, s.end)]
    assert covered == list(range(num_cells))


def test_spatial_shard_unsharded_owns_everything():
    shard = SpatialShard.for_rank(48, 1, 0)

    assert (shard.start, shard.end) == (0, 48)
    assert shard.local_num_cells == 48
    assert not shard.is_sharded


def test_spatial_shard_rejects_indivisible_and_out_of_range():
    with pytest.raises(ValueError, match="must be divisible"):
        SpatialShard.for_rank(48, 5, 0)

    with pytest.raises(ValueError, match="out of range"):
        SpatialShard.for_rank(48, 4, 4)


def test_spatial_shard_range_is_a_valid_cell_split_domain():
    """The shard is exactly what build_local_healpix_cell_splits expects."""

    num_cells, size = 48, 4
    cell_ids = np.repeat(np.arange(num_cells), np.arange(num_cells) % 3 + 1)

    for rank in range(size):
        shard = SpatialShard.for_rank(num_cells, size, rank)
        splits = build_local_healpix_cell_splits(
            cell_ids, num_cells, cell_start=shard.start, cell_end=shard.end
        )
        assert len(splits) == shard.local_num_cells


def test_local_healpix_construction_matches_global_cell_slices():
    num_cells = 48
    cell_ids = np.repeat(np.arange(num_cells), np.arange(num_cells) % 3 + 1)
    rng = np.random.default_rng(7)
    cell_ids = cell_ids[rng.permutation(len(cell_ids))]
    global_cells = build_local_healpix_cell_splits(
        cell_ids,
        num_cells,
        cell_start=0,
        cell_end=num_cells,
    )
    cells_per_rank = len(global_cells) // 4

    local_cells_all = []
    for spatial_rank in range(4):
        cell_start = spatial_rank * cells_per_rank
        cell_end = cell_start + cells_per_rank
        local_cells = build_local_healpix_cell_splits(
            cell_ids,
            num_cells,
            cell_start=cell_start,
            cell_end=cell_end,
        )

        assert len(local_cells) == cells_per_rank
        for local_cell, global_cell in zip(
            local_cells,
            global_cells[cell_start:cell_end],
            strict=True,
        ):
            np.testing.assert_array_equal(local_cell, global_cell)
        local_cells_all.extend(local_cells)

    assert len(local_cells_all) == len(global_cells)
    for local_cell, global_cell in zip(local_cells_all, global_cells, strict=True):
        np.testing.assert_array_equal(local_cell, global_cell)


def test_local_healpix_construction_rejects_invalid_range():
    with pytest.raises(ValueError, match="invalid HEALPix cell range"):
        build_local_healpix_cell_splits(
            np.arange(12),
            num_cells=12,
            cell_start=6,
            cell_end=13,
        )

def test_spatial_parallel_size_requires_whole_rank_groups(monkeypatch):
    monkeypatch.setattr(distributed, "get_world_size", lambda: 16)
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 4}) == 4
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 8}) == 8

    with pytest.raises(ValueError, match="must be divisible"):
        distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 6})


def test_encoder_spatial_parallel_rank_wraps_within_group(monkeypatch):
    monkeypatch.setattr(distributed, "get_world_size", lambda: 16)
    cf = {"encoder_spatial_parallel_size": 4}

    for global_rank, expected in [(0, 0), (3, 3), (4, 0), (5, 1), (11, 3), (15, 3)]:
        monkeypatch.setattr(distributed, "get_rank", lambda r=global_rank: r)
        assert distributed.get_encoder_spatial_parallel_rank(cf) == expected


def test_encoder_spatial_parallel_rank_is_zero_when_unsharded(monkeypatch):
    monkeypatch.setattr(distributed, "get_world_size", lambda: 16)
    monkeypatch.setattr(distributed, "get_rank", lambda: 7)

    assert distributed.get_encoder_spatial_parallel_rank({"encoder_spatial_parallel_size": 1}) == 0


def test_sampler_and_encoder_derive_the_same_shard(monkeypatch):
    """Both sides must resolve one global rank to exactly one HEALPix domain."""

    monkeypatch.setattr(distributed, "get_world_size", lambda: 8)
    cf = {"encoder_spatial_parallel_size": 4}
    num_cells = 48

    shards = []
    for global_rank in range(8):
        monkeypatch.setattr(distributed, "get_rank", lambda r=global_rank: r)
        # Both the sampler and the encoder reach the shard through this pair of
        # calls; there is no second formula that could drift from it.
        shards.append(
            SpatialShard.for_rank(
                num_cells,
                distributed.get_encoder_spatial_parallel_size(cf),
                distributed.get_encoder_spatial_parallel_rank(cf),
            )
        )

    # Each of the two spatial groups (global ranks 0-3 and 4-7) tiles the full domain.
    assert [s.start for s in shards[:4]] == [0, 12, 24, 36]
    assert [s.start for s in shards[4:]] == [0, 12, 24, 36]
