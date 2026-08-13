import numpy as np
import pytest
from anemoi.datasets.data import MissingDateError
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import DTRange


class _Dataset:
    def __init__(self, data: NDArray) -> None:
        self.data = data
        self.indexes = []
        self.dates = np.arange(data.shape[0]).astype("timedelta64[D]") + np.datetime64("2020-01-01")

    def __getitem__(self, index):
        self.indexes.append(index)
        return self.data[index]


class _MissingDataset(_Dataset):
    def __getitem__(self, index):
        raise MissingDateError("missing test date")


def _reader(values: NDArray) -> DataReaderAnemoi:
    reader = object.__new__(DataReaderAnemoi)
    reader.ds = _Dataset(values)
    reader.len = values.shape[0]
    reader.latitudes = np.array([60.0, 50.0, 40.0], dtype=np.float32)
    reader.longitudes = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reader.target_idx = [4, 1]
    reader.geoinfo_idx = [3, 1]
    reader._get_dataset_idxs = lambda _: (
        np.array([0, 1], dtype=np.int64),
        DTRange(np.datetime64("2020-01-01"), np.datetime64("2020-01-03")),
    )
    return reader


@pytest.mark.parametrize("shuffle", [False, True])
def test_early_target_sampling_matches_late_reader_data_sampling(shuffle: bool) -> None:
    values = np.arange(2 * 5 * 1 * 3, dtype=np.float64).reshape(2, 5, 1, 3)
    seed = 7

    legacy_reader = _reader(values)
    legacy = legacy_reader._get(np.int64(0), [4, 1])
    legacy.shuffle(np.random.default_rng(seed), shuffle, 4)
    early_reader = _reader(values)
    actual = early_reader.get_target_sampled(np.int64(0), np.random.default_rng(seed), shuffle, 4)

    for legacy_array, actual_array in zip(
        (legacy.coords, legacy.geoinfos, legacy.data, legacy.datetimes),
        (actual.coords, actual.geoinfos, actual.data, actual.datetimes),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_array, legacy_array)

    # E013 established that a channel-list index fans out into many Zarr reads. Both paths keep
    # the accepted whole-state time-slice access and only project/sample after it returns.
    assert legacy_reader.ds.indexes == [slice(0, 2)]
    assert early_reader.ds.indexes == [slice(0, 2)]


def test_early_target_sampling_missing_date_does_not_advance_rng() -> None:
    values = np.empty((2, 5, 1, 3), dtype=np.float32)
    reader = _reader(values)
    reader.ds = _MissingDataset(values)
    rng = np.random.default_rng(31)
    untouched_rng = np.random.default_rng(31)

    result = reader.get_target_sampled(np.int64(0), rng, False, 4)

    assert result.is_empty()
    assert rng.integers(1_000_000) == untouched_rng.integers(1_000_000)
