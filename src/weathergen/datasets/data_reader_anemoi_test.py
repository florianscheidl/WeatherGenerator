import numpy as np
import pytest
from anemoi.datasets.data import MissingDateError
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi, _read_projected_channels
from weathergen.datasets.data_reader_base import DTRange


class _Dataset:
    def __init__(self, data: NDArray) -> None:
        self.data = data
        self.shape = data.shape
        self.indexes = []
        self.dates = np.arange(data.shape[0]).astype("timedelta64[D]") + np.datetime64("2020-01-01")

    def __getitem__(self, index):
        self.indexes.append(index)
        return self.data[index]


class _MissingDataset(_Dataset):
    def __getitem__(self, index):
        raise MissingDateError("missing test date")


def test_read_projected_channels_selects_before_materializing_and_preserves_order():
    values = np.arange(2 * 5 * 1 * 3, dtype=np.float64).reshape(2, 5, 1, 3)
    dataset = _Dataset(values)

    data, geoinfos, row_indices = _read_projected_channels(dataset, 0, 2, [4, 1], [3, 1])

    assert dataset.indexes == [(slice(0, 2), [4, 1, 3], 0, slice(None))]
    expected = values[:, :, 0].transpose(0, 2, 1).reshape(6, 5)
    np.testing.assert_array_equal(data, expected[:, [4, 1]].astype(np.float32))
    np.testing.assert_array_equal(geoinfos, expected[:, [3, 1]].astype(np.float32))
    assert data.dtype == np.float32
    assert geoinfos.dtype == np.float32
    assert row_indices is None


def test_read_projected_channels_handles_no_selected_fields_without_dataset_access():
    dataset = _Dataset(np.empty((3, 5, 1, 7), dtype=np.float32))

    data, geoinfos, row_indices = _read_projected_channels(dataset, 1, 3, [], [])

    assert dataset.indexes == []
    assert data.shape == (14, 0)
    assert geoinfos.shape == (14, 0)
    assert row_indices is None


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
def test_early_target_sampling_matches_late_reader_data_sampling(shuffle: bool):
    values = np.arange(2 * 5 * 1 * 3, dtype=np.float64).reshape(2, 5, 1, 3)
    seed = 7

    legacy = _reader(values)._get(np.int64(0), [4, 1])
    legacy.shuffle(np.random.default_rng(seed), shuffle, 4)
    actual = _reader(values).get_target_sampled(
        np.int64(0), np.random.default_rng(seed), shuffle, 4
    )

    for legacy_array, actual_array in zip(
        (legacy.coords, legacy.geoinfos, legacy.data, legacy.datetimes),
        (actual.coords, actual.geoinfos, actual.data, actual.datetimes),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_array, legacy_array)


def test_early_target_sampling_missing_date_does_not_advance_rng():
    values = np.empty((2, 5, 1, 3), dtype=np.float32)
    reader = _reader(values)
    reader.ds = _MissingDataset(values)
    rng = np.random.default_rng(31)
    untouched_rng = np.random.default_rng(31)

    result = reader.get_target_sampled(np.int64(0), rng, False, 4)

    assert result.is_empty()
    assert rng.integers(1_000_000) == untouched_rng.integers(1_000_000)
