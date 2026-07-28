# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np


def get_local_healpix_cell_range(
    healpix_level: int,
    spatial_parallel_size: int,
    spatial_parallel_rank: int,
    parent_level: int = 1,
) -> tuple[int, int]:
    """Return a nested-order cell range aligned to complete parent cells."""

    if spatial_parallel_size < 1:
        raise ValueError("spatial_parallel_size must be at least 1")
    if not 0 <= spatial_parallel_rank < spatial_parallel_size:
        raise ValueError(
            f"spatial_parallel_rank {spatial_parallel_rank} is outside [0, {spatial_parallel_size})"
        )

    num_cells = 12 * 4**healpix_level
    if spatial_parallel_size == 1:
        return 0, num_cells
    if not 0 <= parent_level <= healpix_level:
        raise ValueError(
            f"HEALPix parent level {parent_level} must be between 0 and data level {healpix_level}"
        )

    num_parent_cells = 12 * 4**parent_level
    if num_parent_cells % spatial_parallel_size:
        raise ValueError(
            f"encoder_spatial_parallel_size ({spatial_parallel_size}) must divide "
            f"the {num_parent_cells} HEALPix level-{parent_level} parent cells; "
            "otherwise one parent cell would be split across ranks"
        )

    parents_per_rank = num_parent_cells // spatial_parallel_size
    descendants_per_parent = 4 ** (healpix_level - parent_level)
    cell_start = spatial_parallel_rank * parents_per_rank * descendants_per_parent
    cell_end = cell_start + parents_per_rank * descendants_per_parent
    return cell_start, cell_end


def build_local_healpix_cell_splits(
    cell_ids: np.typing.NDArray[np.integer],
    num_cells: int,
    cell_start: int,
    cell_end: int,
) -> list[np.typing.NDArray[np.int64]]:
    """Group original point indices for one consecutive HEALPix-cell domain."""

    if not 0 <= cell_start < cell_end <= num_cells:
        raise ValueError(
            f"invalid HEALPix cell range [{cell_start}, {cell_end}) for {num_cells} cells"
        )

    # Domain-parallel filtering mask: this is applied independently to every
    # stream immediately after its coordinates have been mapped to nested
    # HEALPix cell IDs.
    local_domain_mask = (cell_ids >= cell_start) & (cell_ids < cell_end)
    local_point_idxs = np.flatnonzero(local_domain_mask)
    local_cell_ids = cell_ids[local_point_idxs]
    cell_splits = [np.array([], dtype=np.int64) for _ in range(cell_end - cell_start)]
    if local_point_idxs.size == 0:
        return cell_splits

    stable_args = {"stable": True} if int(np.__version__.split(".")[0]) >= 2 else {}
    local_order = np.argsort(local_cell_ids, **stable_args)
    sorted_point_idxs = local_point_idxs[local_order]
    sorted_cell_ids = local_cell_ids[local_order]
    split_offsets = np.flatnonzero(np.diff(sorted_cell_ids))
    point_idxs_by_occupied_cell = np.split(sorted_point_idxs, split_offsets + 1)

    for cell_id, point_idxs in zip(
        np.unique(sorted_cell_ids),
        point_idxs_by_occupied_cell,
        strict=True,
    ):
        cell_splits[cell_id - cell_start] = point_idxs

    return cell_splits
