# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np
import pytest
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    ReaderData,
    point_selection_indices,
)


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


@pytest.mark.parametrize("shuffle", [False, True])
def test_point_selection_indices_match_reader_data_shuffle(shuffle: bool) -> None:
    seed = 19
    expected = point_selection_indices(12, np.random.default_rng(seed), shuffle, 5)
    assert expected is not None

    data = np.arange(12, dtype=np.float32)[:, None]
    reader_data = ReaderData(
        coords=np.column_stack((data[:, 0], -data[:, 0])),
        geoinfos=data.copy(),
        data=data.copy(),
        datetimes=np.arange(12).astype("datetime64[h]"),
    )

    reader_data.shuffle(np.random.default_rng(seed), shuffle, 5)

    np.testing.assert_array_equal(reader_data.data[:, 0], expected)


@pytest.mark.parametrize(
    ("num_datapoints", "shuffle", "num_subset"),
    [(12, False, -1), (12, False, 13), (0, True, 5)],
)
def test_point_selection_indices_noop_does_not_advance_rng(
    num_datapoints: int, shuffle: bool, num_subset: int
) -> None:
    rng = np.random.default_rng(23)
    untouched_rng = np.random.default_rng(23)

    result = point_selection_indices(num_datapoints, rng, shuffle, num_subset)

    assert result is None
    assert rng.integers(1_000_000) == untouched_rng.integers(1_000_000)


def test_point_selection_indices_equal_cap_retains_legacy_draw() -> None:
    result = point_selection_indices(8, np.random.default_rng(3), False, 8)

    np.testing.assert_array_equal(result, np.arange(8))
