# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import numpy as np
import pytest
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import DataReaderBase


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
