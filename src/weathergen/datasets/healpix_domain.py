# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialShard:
    """The consecutive HEALPix-cell range owned by one encoder-spatial rank.

    This is the single definition of encoder cell ownership. It is deliberately
    pure config arithmetic with no ``torch.distributed`` involvement, so that the
    data sampler (which is pickled into dataloader workers that have no process
    group) and the encoder (which does have one) derive the same range from the
    same formula instead of each reimplementing it.
    """

    #: number of ranks cooperating on one encoder input
    size: int
    #: rank within the spatial-parallel group, in ``[0, size)``
    rank: int
    #: total number of HEALPix cells at the encoder's level
    num_cells: int
    #: first cell owned by this rank
    start: int
    #: one past the last cell owned by this rank
    end: int

    @property
    def local_num_cells(self) -> int:
        """Number of cells owned by this rank."""

        return self.end - self.start

    @property
    def is_sharded(self) -> bool:
        """Whether cells are split across more than one rank."""

        return self.size > 1

    @classmethod
    def for_rank(cls, num_cells: int, size: int, rank: int) -> "SpatialShard":
        """Build the shard owned by ``rank`` in a group of ``size`` ranks."""

        if size < 1:
            raise ValueError(f"encoder_spatial_parallel_size ({size}) must be at least 1")
        if not 0 <= rank < size:
            raise ValueError(
                f"encoder spatial rank ({rank}) out of range for "
                f"encoder_spatial_parallel_size ({size})"
            )
        if num_cells % size:
            raise ValueError(
                f"number of HEALPix cells ({num_cells}) must be divisible by "
                f"encoder_spatial_parallel_size ({size})"
            )

        local_num_cells = num_cells // size
        start = rank * local_num_cells
        return cls(
            size=size,
            rank=rank,
            num_cells=num_cells,
            start=start,
            end=start + local_num_cells,
        )


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
