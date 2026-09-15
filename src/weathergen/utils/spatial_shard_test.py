# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses

import pytest

from weathergen.utils.spatial_shard import SpatialShard


def test_spatial_shards_are_contiguous_disjoint_and_exhaustive() -> None:
    shards = [
        SpatialShard(healpix_level=2, spatial_parallel_size=4, spatial_rank=r) for r in range(4)
    ]

    assert shards[0].num_cells == 192
    assert all(shard.cells_per_rank == 48 for shard in shards)
    assert [(shard.cell_start, shard.cell_end) for shard in shards] == [
        (0, 48),
        (48, 96),
        (96, 144),
        (144, 192),
    ]
    assert [cell for shard in shards for cell in range(*shard.cell_slice.indices(192))] == list(
        range(192)
    )


def test_spatial_shards_repeat_across_data_parallel_groups() -> None:
    first_group = [SpatialShard.from_global_rank(1, 4, rank) for rank in range(4)]
    second_group = [SpatialShard.from_global_rank(1, 4, rank) for rank in range(4, 8)]

    assert first_group == second_group


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"healpix_level": -1, "spatial_parallel_size": 1, "spatial_rank": 0}, "non-negative"),
        ({"healpix_level": 0, "spatial_parallel_size": 0, "spatial_rank": 0}, "at least 1"),
        ({"healpix_level": 0, "spatial_parallel_size": 4, "spatial_rank": 4}, "must be in"),
        ({"healpix_level": 0, "spatial_parallel_size": 5, "spatial_rank": 0}, "divisible"),
    ],
)
def test_spatial_shard_rejects_invalid_partitions(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SpatialShard(**kwargs)


def test_spatial_shard_is_immutable() -> None:
    shard = SpatialShard(healpix_level=0, spatial_parallel_size=1, spatial_rank=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        shard.spatial_rank = 1  # type: ignore[misc]


def test_spatial_shard_rejects_negative_global_rank() -> None:
    with pytest.raises(ValueError, match="global_rank must be non-negative"):
        SpatialShard.from_global_rank(0, 1, -1)
