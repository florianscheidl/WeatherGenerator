import os

import numpy as np
import pytest

import weathergen.datasets.data_reader_anemoi as data_reader_anemoi
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi

T_START = np.datetime64("2023-03-01T00:00")
T_END = np.datetime64("2023-03-02T00:00")


@pytest.fixture
def opened_datasets(monkeypatch):
    """Record the calls to anemoi's open_dataset instead of opening datasets."""
    calls = []

    def fake_open_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return f"dataset{len(calls)}"

    monkeypatch.setattr(data_reader_anemoi.anemoi_datasets, "open_dataset", fake_open_dataset)
    return calls


def _reader(filename) -> DataReaderAnemoi:
    """Reader with a dataset, without the dataset its constructor needs."""
    reader = DataReaderAnemoi.__new__(DataReaderAnemoi)
    reader._open_args = (filename, {"frequency": "6h"}, T_START, T_END)
    reader.ds = "dataset0"
    return reader


def test_reopen_remote_dataset_after_fork(opened_datasets):
    url = "https://example.com/data.zarr"
    reader = _reader(url)
    # as in a forked data loader worker
    reader._ds_pid = os.getpid() + 1

    assert reader.ds == "dataset2"
    # the dataset is reopened with the same arguments as in the constructor
    assert opened_datasets == [
        ((url,), {}),
        (("dataset1",), {"frequency": "6h", "start": T_START, "end": T_END}),
    ]

    # only once per process
    assert reader.ds == "dataset2"
    assert len(opened_datasets) == 2


def test_no_reopen_local_dataset_after_fork(opened_datasets, tmp_path):
    reader = _reader(tmp_path / "data.zarr")
    reader._ds_pid = os.getpid() + 1

    assert reader.ds == "dataset0"
    assert opened_datasets == []


def test_no_reopen_empty_reader_after_fork(opened_datasets):
    reader = _reader("https://example.com/data.zarr")
    reader.ds = None
    reader._ds_pid = os.getpid() + 1

    assert reader.ds is None
    assert opened_datasets == []
