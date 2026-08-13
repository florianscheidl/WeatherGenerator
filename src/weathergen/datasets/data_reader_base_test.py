# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np
import pytest
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import DataReaderBase, ReaderData, point_selection_indices


def _reader_data(num_rows: int) -> ReaderData:
    values = np.arange(num_rows, dtype=np.float32)
    return ReaderData(
        coords=np.column_stack((values, values + 1)),
        geoinfos=values[:, None],
        data=np.column_stack((values + 2, values + 3)),
        datetimes=np.arange(num_rows).astype("timedelta64[h]") + np.datetime64("2020-01-01"),
    )


@pytest.mark.parametrize(("shuffle", "num_subset"), [(False, 4), (True, 4), (True, -1)])
def test_point_selection_indices_match_reader_data_shuffle(shuffle: bool, num_subset: int) -> None:
    seed = 17
    expected = _reader_data(8).shuffle(np.random.default_rng(seed), shuffle, num_subset)
    original = _reader_data(8)

    indices = point_selection_indices(
        original.len(), np.random.default_rng(seed), shuffle, num_subset
    )

    assert indices is not None
    np.testing.assert_array_equal(original.coords[indices], expected.coords)
    np.testing.assert_array_equal(original.geoinfos[indices], expected.geoinfos)
    np.testing.assert_array_equal(original.data[indices], expected.data)
    np.testing.assert_array_equal(original.datetimes[indices], expected.datetimes)


def test_point_selection_noop_does_not_advance_rng() -> None:
    rng = np.random.default_rng(23)
    untouched_rng = np.random.default_rng(23)

    indices = point_selection_indices(8, rng, False, -1)

    assert indices is None
    assert rng.integers(1_000_000) == untouched_rng.integers(1_000_000)


def test_point_selection_equal_cap_preserves_legacy_random_draw() -> None:
    rng = np.random.default_rng(29)
    legacy_rng = np.random.default_rng(29)

    indices = point_selection_indices(8, rng, False, 8)
    legacy_indices = np.sort(legacy_rng.choice(8, 8, replace=False))

    np.testing.assert_array_equal(indices, legacy_indices)
    assert rng.integers(1_000_000) == legacy_rng.integers(1_000_000)


def _normalize_channel_loop(
    data: NDArray[np.float32],
    idx: list[int],
    mean: NDArray[np.float64],
    stdev: NDArray[np.float64],
) -> NDArray[np.float32]:
    for i, ch in enumerate(idx):
        data[..., i] = (data[..., i] - mean[ch]) / stdev[ch]
    return data


def test_normalize_vectorized_matches_channel_loop_in_place() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=(5, 7, 4)).astype(np.float32)
    data[1, 2, 3] = np.nan
    idx = [6, 1, 8, 3]
    mean = rng.normal(size=10)
    stdev = rng.uniform(0.1, 2.0, size=10)
    expected = _normalize_channel_loop(data.copy(), idx, mean, stdev)

    actual = data.copy()
    result = DataReaderBase._normalize(actual, idx, mean, stdev, "target")

    assert result is actual
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_normalize_supports_empty_channels() -> None:
    data = np.empty((3, 0), dtype=np.float32)

    result = DataReaderBase._normalize(data, [], np.array([]), np.array([]), "target")

    assert result is data
    assert result.shape == (3, 0)


def test_normalize_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="expected 2, got 3"):
        DataReaderBase._normalize(
            np.zeros((4, 3), dtype=np.float32),
            [0, 1],
            np.zeros(2),
            np.ones(2),
            "target",
        )
