# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from typing import override

import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    ReaderData,
    TimeWindowHandler,
    TIndex,
)
from weathergen.train.utils import Stage
from weathergen.utils.spatial_shard import SpatialShard


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    year, month, day, hour, min, sec = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = year + 1970  # Gregorian Year
    out[..., 1] = (month - year) + 1  # month
    out[..., 2] = (day - month) + 1  # dat
    out[..., 3] = (dt - day).astype("m8[h]")  # hour
    out[..., 4] = (dt - hour).astype("m8[m]")  # minute
    out[..., 5] = (dt - min).astype("m8[s]")  # second
    out[..., 6] = (dt - sec).astype("m8[us]")  # microsecond
    return out


class DataReaderAnemoiOperan(DataReaderAnemoi):
    "Wrapper for Anemoi datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
        spatial_shard: SpatialShard | None = None,
    ) -> None:
        """
        Construct data reader for anemoi dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream
        spatial_shard :
            when set, source reads return only grid rows in this rank's HEALPix
            cell range; targets remain global. Rows with NaN coordinates (e.g.
            off-disk geostationary pixels) belong to no rank and are dropped at
            the reader instead of in the later NaN cleanup.

        Returns
        -------
        None
        """

        super().__init__(tw_handler, filename, stream_info, stage, spatial_shard)

    @override
    def _get(
        self,
        idx: TIndex,
        channels_idx: list[int],
        grid_rows: NDArray[np.int64] | None = None,
    ) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels
        grid_rows : np.array, optional
            When given, only these grid rows (per timestep) are returned

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        t_idxs, dtr = self._get_dataset_idxs(idx)
        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # get additional timestep to ensure we have one valid timestep
        t_idxs = np.insert(t_idxs, 0, t_idxs[0] - 1)

        didx_start = t_idxs[0]
        didx_end = t_idxs[-1] + 1
        datetimes = self.ds.dates[didx_start:didx_end]
        datetimes_split = dt2cal(datetimes)

        # compute corrected datetimes that account for actual availability
        nts = self.stream_info["nominal_time_mapping"]
        deltas = [int(nts[str(hour)]) - int(hour) for hour in datetimes_split[:, 3]]
        datetimes_offset = [
            dt + np.timedelta64(delta, "h") for dt, delta in zip(datetimes, deltas, strict=False)
        ]

        # use latest available sample that is valid w.r.t the input data window
        datetimes_mask = [dt < dtr.end for dt in datetimes_offset]
        if np.array(datetimes_mask).sum() == 0:
            t_idxs = []
        else:
            t_idxs = [t_idxs[datetimes_mask][-1].item()]

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        # End is inclusive
        rd = self._read_window(t_idxs[0], t_idxs[-1] + 1, channels_idx, grid_rows)
        if rd is None:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )
        # The selected timestep may lie before the window start (latest available
        # sample), so the base class's check_reader_data window check is skipped.

        return rd
