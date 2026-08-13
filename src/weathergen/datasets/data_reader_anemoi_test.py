import numpy as np
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import _read_projected_channels


class _Dataset:
    def __init__(self, data: NDArray) -> None:
        self.data = data
        self.shape = data.shape
        self.indexes = []

    def __getitem__(self, index):
        self.indexes.append(index)
        return self.data[index]


def test_read_projected_channels_selects_before_materializing_and_preserves_order():
    values = np.arange(2 * 5 * 1 * 3, dtype=np.float64).reshape(2, 5, 1, 3)
    dataset = _Dataset(values)

    data, geoinfos = _read_projected_channels(dataset, 0, 2, [4, 1], [3, 1])

    assert dataset.indexes == [(slice(0, 2), [4, 1, 3], 0, slice(None))]
    expected = values[:, :, 0].transpose(0, 2, 1).reshape(6, 5)
    np.testing.assert_array_equal(data, expected[:, [4, 1]].astype(np.float32))
    np.testing.assert_array_equal(geoinfos, expected[:, [3, 1]].astype(np.float32))
    assert data.dtype == np.float32
    assert geoinfos.dtype == np.float32


def test_read_projected_channels_handles_no_selected_fields_without_dataset_access():
    dataset = _Dataset(np.empty((3, 5, 1, 7), dtype=np.float32))

    data, geoinfos = _read_projected_channels(dataset, 1, 3, [], [])

    assert dataset.indexes == []
    assert data.shape == (14, 0)
    assert geoinfos.shape == (14, 0)
