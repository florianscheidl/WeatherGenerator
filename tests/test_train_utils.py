# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

import pytest

from weathergen.train.utils import validation_sample_limit_reached


@pytest.mark.parametrize(
    ("batch_index", "batch_size", "sample_limit", "expected"),
    [
        (0, 1, 1, True),
        (0, 2, 4, False),
        (1, 2, 4, True),
        (1, 3, 4, True),
    ],
)
def test_validation_sample_limit_counts_the_completed_batch(
    batch_index: int, batch_size: int, sample_limit: int, expected: bool
) -> None:
    assert validation_sample_limit_reached(batch_index, batch_size, sample_limit) is expected
