import os

import numpy as np
import pytest
import zarr

import weathergen.datasets.data_reader_obs as data_reader_obs
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.data_reader_obs import DataReaderObs


@pytest.fixture
def opened_stores(monkeypatch):
    """Record the stores opened by zarr.open instead of opening them."""
    opened = []

    def fake_open(filename, mode):
        opened.append(filename)
        return {"data": f"data{len(opened)}", "dates": f"dates{len(opened)}"}

    monkeypatch.setattr(data_reader_obs.zarr, "open", fake_open)
    return opened


def _reader(filename) -> DataReaderObs:
    """Reader with an opened store, without the dataset its constructor needs."""
    reader = DataReaderObs.__new__(DataReaderObs)
    reader.filename = filename
    reader._open_store()
    return reader


def test_reopen_remote_store_after_fork(opened_stores):
    reader = _reader("https://example.com/data.zarr")
    # as in a forked data loader worker
    reader._store_pid = os.getpid() + 1

    reader._reopen_store_after_fork()

    assert opened_stores == ["https://example.com/data.zarr"] * 2
    assert reader.data == "data2"
    assert reader.dt == "dates2"
    assert reader._store_pid == os.getpid()

    # only once per process
    reader._reopen_store_after_fork()
    assert len(opened_stores) == 2


def test_no_reopen_local_store_after_fork(opened_stores, tmp_path):
    reader = _reader(tmp_path / "data.zarr")
    reader._store_pid = os.getpid() + 1

    reader._reopen_store_after_fork()

    assert len(opened_stores) == 1
    assert reader.data == "data1"


def test_get_selects_columns():
    t_start = np.datetime64("2023-03-01T00:00")
    reader = DataReaderObs.__new__(DataReaderObs)
    reader.filename = "data.zarr"
    reader._store_pid = os.getpid()
    reader.time_window_handler = TimeWindowHandler(
        t_start, np.datetime64("2023-03-02T00:00"), np.timedelta64(6, "h"), np.timedelta64(6, "h")
    )
    # columns: lat, lon, geoinfo, channel 0, channel 1
    values = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    reader.data = zarr.array(values, chunks=(2, 5))
    reader.dt = zarr.array(np.full((4, 1), t_start + np.timedelta64(1, "h")))
    reader.indices_start, reader.indices_end = np.array([1]), np.array([4])
    reader.coords_idx = [0, 1]
    reader.geoinfo_idx = [2]

    rdata = reader._get(0, np.array([4, 3]))

    np.testing.assert_array_equal(rdata.coords, values[1:4, [0, 1]])
    np.testing.assert_array_equal(rdata.geoinfos, values[1:4, [2]])
    np.testing.assert_array_equal(rdata.data, values[1:4, [4, 3]])
