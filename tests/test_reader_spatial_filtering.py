# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Reader-boundary early filtering vs the tokenizer's late filtering.

The pure tests validate that `shard_grid_point_rows` selects exactly the
rows the tokenizer's late HEALPix selection would keep. The dataset tests compare
a full-read `DataReaderAnemoi` (the oracle) against per-rank early-filtering
readers on a local Anemoi ERA5 O96 store; they are skipped when the store is
absent (see `datasets/download_configs/era5_o96_2020_1m.yaml` to create it).
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from omegaconf import OmegaConf

from weathergen.datasets.healpix_domain import (
    build_local_healpix_cell_splits,
    shard_grid_point_rows,
    theta_phi_to_standard_coords,
)
from weathergen.utils.spatial_shard import SpatialShard

HEALPIX_LEVEL = 5
NUM_CELLS = 12 * 4**HEALPIX_LEVEL
SPATIAL_SIZE = 4

O96_ZARR = Path(__file__).parents[1] / "datasets" / "era5-o96-2020-1pct-6h-v1.zarr"

needs_o96 = pytest.mark.skipif(
    not O96_ZARR.exists(),
    reason=f"local O96 test dataset not found at {O96_ZARR}",
)


def _domains(spatial_size: int) -> list[SpatialShard]:
    return [SpatialShard(HEALPIX_LEVEL, spatial_size, rank) for rank in range(spatial_size)]


def _random_coords(num_points: int, seed: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    lats = rng.uniform(-90.0, 90.0, num_points).astype(np.float32)
    lons = rng.uniform(-180.0, 180.0, num_points).astype(np.float32)
    return np.stack([lats, lons], axis=1)


def test_grid_point_rows_matches_tokenizer_cell_assignment():
    """Early filtering must keep exactly the rows late filtering keeps."""
    coords = _random_coords(20_000, seed=3)
    from astropy_healpix.healpy import ang2pix

    thetas, phis = theta_phi_to_standard_coords(coords)
    cell_ids = ang2pix(2**HEALPIX_LEVEL, thetas, phis, nest=True)

    for domain in _domains(SPATIAL_SIZE):
        rows = shard_grid_point_rows(domain, coords[:, 0], coords[:, 1])
        late_splits = build_local_healpix_cell_splits(
            cell_ids, NUM_CELLS, domain.cell_start, domain.cell_end
        )
        late_rows = np.sort(np.concatenate(late_splits))
        np.testing.assert_array_equal(np.sort(rows), late_rows)


def test_grid_point_rows_disjoint_partition():
    coords = _random_coords(10_000, seed=11)
    all_rows = [
        shard_grid_point_rows(d, coords[:, 0], coords[:, 1]) for d in _domains(SPATIAL_SIZE)
    ]
    union = np.sort(np.concatenate(all_rows))
    np.testing.assert_array_equal(union, np.arange(len(coords)))


def test_grid_point_rows_excludes_nan_coordinates():
    """Off-disk pixels (NaN coords) belong to no rank; finite rows still partition."""
    coords = _random_coords(5_000, seed=23)
    coords[::7, 0] = np.nan
    coords[::11, 1] = np.nan
    finite = np.flatnonzero(np.isfinite(coords).all(axis=1))

    all_rows = [
        shard_grid_point_rows(d, coords[:, 0], coords[:, 1]) for d in _domains(SPATIAL_SIZE)
    ]
    union = np.sort(np.concatenate(all_rows))
    np.testing.assert_array_equal(union, finite)


def test_spatial_shard_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="spatial_rank"):
        SpatialShard(HEALPIX_LEVEL, SPATIAL_SIZE, SPATIAL_SIZE)
    with pytest.raises(ValueError, match="divisible"):
        SpatialShard(HEALPIX_LEVEL, 5, 0)


@pytest.fixture(scope="module")
def o96_readers():
    """One full-read oracle reader plus one early-filtering reader per rank."""
    from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
    from weathergen.datasets.data_reader_base import TimeWindowHandler

    tw_handler = TimeWindowHandler(
        np.datetime64("2020-01-05T00:00"),
        np.datetime64("2020-01-20T00:00"),
        np.timedelta64(12, "h"),
        np.timedelta64(6, "h"),
    )
    stream_info = OmegaConf.create({"name": "era5_o96_test", "type": "anemoi"})

    def make_reader(domain: SpatialShard | None) -> DataReaderAnemoi:
        return DataReaderAnemoi(
            tw_handler=tw_handler,
            filename=O96_ZARR,
            stream_info=stream_info,
            stage="train",
            spatial_shard=domain,
        )

    full = make_reader(None)
    local = [make_reader(domain) for domain in _domains(SPATIAL_SIZE)]
    return full, local


@needs_o96
def test_local_grid_rows_partition_o96_grid(o96_readers):
    full, local = o96_readers
    num_points = len(full.latitudes)
    all_rows = np.sort(np.concatenate([r.local_grid_rows for r in local]))
    np.testing.assert_array_equal(all_rows, np.arange(num_points))
    # contiguous nested cells keep the per-rank load balanced: ~1/P each
    for reader in local:
        assert len(reader.local_grid_rows) == pytest.approx(num_points / SPATIAL_SIZE, rel=0.1)


@needs_o96
@pytest.mark.parametrize("window_idx", [0, 7])
def test_early_filtered_sources_reassemble_to_full_read(o96_readers, window_idx):
    """Ordered union of the rank shards reproduces the oracle full read."""
    full, local = o96_readers
    idx = np.int64(window_idx)
    rdata_full = full.get_source(idx)
    num_points = len(full.latitudes)
    num_steps = len(rdata_full.data) // num_points

    for reader in local:
        rdata_local = reader.get_source(idx)
        rows = reader.local_grid_rows
        assert len(rdata_local.data) == num_steps * len(rows)
        # global row ids of the shard: per timestep t, t * num_points + rows
        global_rows = np.concatenate([t * num_points + rows for t in range(num_steps)])
        np.testing.assert_array_equal(rdata_local.data, rdata_full.data[global_rows])
        np.testing.assert_array_equal(rdata_local.coords, rdata_full.coords[global_rows])
        np.testing.assert_array_equal(rdata_local.geoinfos, rdata_full.geoinfos[global_rows])
        np.testing.assert_array_equal(rdata_local.datetimes, rdata_full.datetimes[global_rows])


@needs_o96
def test_legacy_get_signature_subclass_still_works(o96_readers):
    """Subclasses overriding _get without grid_rows (e.g. anemoi_operan) must not break.

    They never receive a spatial_shard, so get_source must call _get with the
    legacy two-argument form for them.
    """
    from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi

    full, _ = o96_readers

    class LegacyReader(DataReaderAnemoi):
        def _get(self, idx, channels_idx):  # type: ignore[override]
            return super()._get(idx, channels_idx)

    legacy = LegacyReader(
        tw_handler=full.time_window_handler,
        filename=O96_ZARR,
        stream_info=full.stream_info,
        stage="train",
    )
    rdata = legacy.get_source(np.int64(0))
    np.testing.assert_array_equal(rdata.data, full.get_source(np.int64(0)).data)


@needs_o96
def test_targets_stay_global(o96_readers):
    full, local = o96_readers
    idx = np.int64(0)
    rdata_full = full.get_target(idx)
    for reader in local:
        rdata_target = reader.get_target(idx)
        np.testing.assert_array_equal(rdata_target.data, rdata_full.data)
        np.testing.assert_array_equal(rdata_target.coords, rdata_full.coords)


@needs_o96
def test_operan_reader_filters_sources(o96_readers):
    """DataReaderAnemoiOperan inherits early filtering through _read_window."""
    from weathergen.readers_extra.data_reader_anemoi_operan import DataReaderAnemoiOperan

    full = o96_readers[0]
    # identity mapping: every nominal hour is its own availability hour
    stream_info = OmegaConf.create(
        {
            "name": "operan_test",
            "type": "anemoi_operan",
            "nominal_time_mapping": {str(h): h for h in range(24)},
        }
    )

    def make(domain):
        return DataReaderAnemoiOperan(
            tw_handler=full.time_window_handler,
            filename=O96_ZARR,
            stream_info=stream_info,
            stage="train",
            spatial_shard=domain,
        )

    idx = np.int64(2)  # >= 1: operan prepends one earlier timestep
    rdata_full = make(None).get_source(idx)
    num_points = len(full.latitudes)
    num_steps = len(rdata_full.data) // num_points
    assert num_steps >= 1

    for domain in _domains(SPATIAL_SIZE):
        reader = make(domain)
        rdata_local = reader.get_source(idx)
        rows = reader.local_grid_rows
        global_rows = np.concatenate([t * num_points + rows for t in range(num_steps)])
        np.testing.assert_array_equal(rdata_local.data, rdata_full.data[global_rows])
        np.testing.assert_array_equal(rdata_local.coords, rdata_full.coords[global_rows])
        # targets stay global for operan too
        np.testing.assert_array_equal(reader.get_target(idx).data, make(None).get_target(idx).data)


@needs_o96
def test_shard_rows_lie_in_local_cells(o96_readers):
    """Every returned coordinate maps back into the rank's cell range."""
    from astropy_healpix.healpy import ang2pix

    _, local = o96_readers
    for domain, reader in zip(_domains(SPATIAL_SIZE), local, strict=True):
        rdata = reader.get_source(np.int64(0))
        thetas, phis = theta_phi_to_standard_coords(rdata.coords)
        cell_ids = ang2pix(2**HEALPIX_LEVEL, thetas, phis, nest=True)
        assert cell_ids.min() >= domain.cell_start
        assert cell_ids.max() < domain.cell_end
