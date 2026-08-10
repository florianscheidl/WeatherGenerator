# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialShard:
    """One rank's contiguous range in the nested HEALPix cell ordering.

    The descriptor is independent of ``torch.distributed`` so it can be created
    before DataLoader workers start and safely copied into worker processes.
    """

    healpix_level: int
    spatial_parallel_size: int
    spatial_rank: int

    def __post_init__(self) -> None:
        if self.healpix_level < 0:
            raise ValueError("healpix_level must be non-negative")
        if self.spatial_parallel_size < 1:
            raise ValueError("spatial_parallel_size must be at least 1")
        if not 0 <= self.spatial_rank < self.spatial_parallel_size:
            raise ValueError(
                f"spatial_rank ({self.spatial_rank}) must be in [0, {self.spatial_parallel_size})"
            )
        if self.num_cells % self.spatial_parallel_size:
            raise ValueError(
                f"number of HEALPix cells ({self.num_cells}) must be divisible by "
                f"spatial_parallel_size ({self.spatial_parallel_size})"
            )

    @classmethod
    def from_global_rank(
        cls,
        healpix_level: int,
        spatial_parallel_size: int,
        global_rank: int,
    ) -> "SpatialShard":
        """Construct the shard owned by ``global_rank`` in consecutive groups."""

        if global_rank < 0:
            raise ValueError("global_rank must be non-negative")
        return cls(
            healpix_level=healpix_level,
            spatial_parallel_size=spatial_parallel_size,
            spatial_rank=global_rank % spatial_parallel_size,
        )

    @property
    def num_cells(self) -> int:
        return 12 * 4**self.healpix_level

    @property
    def cells_per_rank(self) -> int:
        return self.num_cells // self.spatial_parallel_size

    @property
    def cell_start(self) -> int:
        return self.spatial_rank * self.cells_per_rank

    @property
    def cell_end(self) -> int:
        return self.cell_start + self.cells_per_rank

    @property
    def cell_slice(self) -> slice:
        return slice(self.cell_start, self.cell_end)
