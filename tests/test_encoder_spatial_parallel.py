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

from weathergen.datasets.healpix_domain import (
    build_local_healpix_cell_splits,
    get_local_healpix_cell_range,
)
from weathergen.model.spatial_parallel import select_packed_cell_shard
from weathergen.utils import distributed


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


@pytest.mark.parametrize("spatial_parallel_size", [4, 8, 16])
def test_spatial_ranges_keep_level_one_parent_descendants_together(
    spatial_parallel_size,
):
    healpix_level = 5
    descendants_per_parent = 4 ** (healpix_level - 1)
    num_cells = 12 * 4**healpix_level
    rank_owner = np.full(num_cells, -1, dtype=np.int32)

    for spatial_rank in range(spatial_parallel_size):
        cell_start, cell_end = get_local_healpix_cell_range(
            healpix_level,
            spatial_parallel_size,
            spatial_rank,
            parent_level=1,
        )
        assert cell_start % descendants_per_parent == 0
        assert cell_end % descendants_per_parent == 0
        rank_owner[cell_start:cell_end] = spatial_rank

    assert np.all(rank_owner >= 0)
    for parent_cell in range(48):
        child_start = parent_cell * descendants_per_parent
        child_end = child_start + descendants_per_parent
        assert np.unique(rank_owner[child_start:child_end]).size == 1


def test_all_variable_tokens_from_one_level_one_parent_go_to_one_rank():
    healpix_level = 5
    spatial_parallel_size = 4
    descendants_per_parent = 4 ** (healpix_level - 1)
    parent_cell = 17
    child_start = parent_cell * descendants_per_parent
    child_ids = np.arange(child_start, child_start + descendants_per_parent)
    cell_ids = np.repeat(child_ids, child_ids % 5 + 1)

    token_owners = []
    for spatial_rank in range(spatial_parallel_size):
        cell_start, cell_end = get_local_healpix_cell_range(
            healpix_level,
            spatial_parallel_size,
            spatial_rank,
            parent_level=1,
        )
        local_cells = build_local_healpix_cell_splits(
            cell_ids,
            12 * 4**healpix_level,
            cell_start,
            cell_end,
        )
        num_local_points = sum(len(point_idxs) for point_idxs in local_cells)
        token_owners.extend([spatial_rank] * num_local_points)

    assert len(token_owners) == len(cell_ids)
    assert len(set(token_owners)) == 1


def test_spatial_size_cannot_split_level_one_parent_cells():
    with pytest.raises(ValueError, match="otherwise one parent cell would be split"):
        get_local_healpix_cell_range(
            healpix_level=5,
            spatial_parallel_size=32,
            spatial_parallel_rank=0,
            parent_level=1,
        )


def test_select_packed_cell_shard_preserves_cell_boundaries_across_rows():
    cell_lens = torch.tensor(
        [
            [1, 0, 2, 1, 3, 0, 1, 2],
            [0, 2, 1, 0, 1, 2, 0, 1],
        ],
        dtype=torch.int32,
    )
    tokens = torch.arange(cell_lens.sum(), dtype=torch.float32).unsqueeze(1)

    shard, shard_lens = select_packed_cell_shard(
        tokens, cell_lens.flatten(), num_cells=8, cell_start=2, cell_end=4
    )

    assert shard_lens.tolist() == [2, 1, 1, 0]
    assert shard.squeeze(1).tolist() == [1, 2, 3, 12]


def test_eight_shards_cover_every_packed_token_once_and_keep_gradients():
    num_cells = 16
    cell_lens = torch.tensor(
        [
            [0, 1, 2, 0, 1, 3, 0, 2, 1, 0, 2, 1, 0, 1, 2, 1],
            [1, 0, 1, 2, 0, 1, 2, 0, 3, 1, 0, 1, 2, 0, 1, 1],
        ],
        dtype=torch.int32,
    )
    tokens = torch.arange(cell_lens.sum(), dtype=torch.float32, requires_grad=True)
    shard_width = num_cells // 8

    selected = []
    for rank in range(8):
        shard, _ = select_packed_cell_shard(
            tokens,
            cell_lens.flatten(),
            num_cells,
            rank * shard_width,
            (rank + 1) * shard_width,
        )
        selected.append(shard)
        shard.sum().backward(retain_graph=rank < 7)

    assert sum(shard.numel() for shard in selected) == tokens.numel()
    assert torch.equal(tokens.grad, torch.ones_like(tokens))


@pytest.mark.parametrize(
    ("num_cells", "cell_start", "cell_end"),
    [(8, -1, 1), (8, 3, 3), (8, 0, 9)],
)
def test_select_packed_cell_shard_rejects_invalid_ranges(num_cells, cell_start, cell_end):
    with pytest.raises(ValueError, match="invalid HEALPix cell range"):
        select_packed_cell_shard(
            torch.arange(num_cells),
            torch.ones(num_cells, dtype=torch.int32),
            num_cells,
            cell_start,
            cell_end,
        )


def test_spatial_parallel_size_requires_whole_rank_groups(monkeypatch):
    monkeypatch.setattr(distributed, "get_world_size", lambda: 16)
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 4}) == 4
    assert distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 8}) == 8

    with pytest.raises(ValueError, match="must be divisible"):
        distributed.get_encoder_spatial_parallel_size({"encoder_spatial_parallel_size": 6})
