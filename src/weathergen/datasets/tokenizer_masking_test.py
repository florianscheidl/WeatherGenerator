# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest
import torch
from astropy_healpix import healpy

from weathergen.common.io import IOReaderData
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.utils.spatial_shard import SpatialShard


def _target_fixture() -> tuple[IOReaderData, tuple[list, list], npt.NDArray[np.bool_]]:
    num_cells = 12
    theta, phi = healpy.pix2ang(1, np.arange(num_cells), nest=True)
    cell_coords = np.column_stack((90.0 - np.degrees(theta), np.degrees(phi) - 180.0))
    coords = torch.tensor(np.repeat(cell_coords, 2, axis=0), dtype=torch.float32)
    num_points = len(coords)
    rdata = IOReaderData(
        coords=coords,
        geoinfos=torch.arange(num_points * 2, dtype=torch.float32).reshape(num_points, 2),
        data=torch.arange(num_points * 3, dtype=torch.float32).reshape(num_points, 3),
        datetimes=np.full(num_points, np.datetime64("2026-01-01T00:00:00")),
    )
    idxs_cells = [[torch.tensor([2 * cell, 2 * cell + 1])] for cell in range(num_cells)]
    idxs_cells_lens = [[2] for _ in range(num_cells)]
    return rdata, (idxs_cells, idxs_cells_lens), np.ones(num_cells, dtype=bool)


def test_rank_local_target_coords_match_global_oracle_while_values_remain_global() -> None:
    rdata, token_data, cell_mask = _target_fixture()
    stream_info = {"stream_id": 7}
    time_win = (np.datetime64("2026-01-01"), np.datetime64("2026-01-02"))
    oracle = TokenizerMasking(healpix_level=0, masker=Mock())
    global_coords, global_lens = oracle.get_target_coords(
        stream_info,
        rdata,
        token_data,
        time_win,
        cell_mask,
    )
    global_values = oracle.get_target_values(
        stream_info,
        rdata,
        token_data,
        time_win,
        cell_mask,
    )[0]

    local_coords = []
    for spatial_rank in range(4):
        local = TokenizerMasking(
            healpix_level=0,
            masker=Mock(),
            spatial_shard=SpatialShard(
                healpix_level=0,
                spatial_parallel_size=4,
                spatial_rank=spatial_rank,
            ),
        )
        coords, global_lens_from_local_path = local.get_target_coords(
            stream_info,
            rdata,
            token_data,
            time_win,
            cell_mask,
        )
        local_values = local.get_target_values(
            stream_info,
            rdata,
            token_data,
            time_win,
            cell_mask,
        )[0]

        local_coords.append(coords)
        assert torch.equal(global_lens_from_local_path, global_lens)
        assert torch.equal(local_values, global_values)
        assert len(coords) == len(global_coords) // 4

    torch.testing.assert_close(torch.cat(local_coords), global_coords)


def test_rank_local_target_values_reassemble_to_global_oracle() -> None:
    rdata, token_data, cell_mask = _target_fixture()
    stream_info = {"stream_id": 7}
    time_win = (np.datetime64("2026-01-01"), np.datetime64("2026-01-02"))
    oracle = TokenizerMasking(healpix_level=0, masker=Mock())
    global_values, global_times, global_coords, global_inverse, global_row_ids = (
        oracle.get_target_values(
            stream_info,
            rdata,
            token_data,
            time_win,
            cell_mask,
        )
    )

    local_values, local_times, local_coords, local_row_ids = [], [], [], []
    for spatial_rank in range(4):
        local = TokenizerMasking(
            healpix_level=0,
            masker=Mock(),
            spatial_shard=SpatialShard(0, 4, spatial_rank),
            local_target_values=True,
        )
        values, times, coords, inverse, row_ids = local.get_target_values(
            stream_info,
            rdata,
            token_data,
            time_win,
            cell_mask,
        )
        local_values.append(values)
        local_times.append(times)
        local_coords.append(coords)
        local_row_ids.append(row_ids)
        assert inverse is None

    torch.testing.assert_close(torch.cat(local_values), global_values)
    torch.testing.assert_close(torch.cat(local_coords), global_coords)
    assert np.array_equal(np.concatenate(local_times), global_times)
    assert torch.equal(torch.cat(local_row_ids), global_row_ids)
    assert torch.equal(global_values[global_inverse], rdata.data[global_row_ids[global_inverse]])
    assert global_inverse is not None


def test_tokenizer_rejects_shard_from_another_healpix_level() -> None:
    with pytest.raises(ValueError, match="does not match tokenizer level"):
        TokenizerMasking(
            healpix_level=0,
            masker=Mock(),
            spatial_shard=SpatialShard(healpix_level=1, spatial_parallel_size=1, spatial_rank=0),
        )
