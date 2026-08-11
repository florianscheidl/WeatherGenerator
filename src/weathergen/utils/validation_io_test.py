# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0.

import numpy as np
import pytest
import torch

from weathergen.utils.validation_io import _reassemble_validation_rows


def test_reassemble_validation_rows_uses_stable_global_row_order() -> None:
    predictions = [
        torch.tensor([[[30.0], [10.0]]]),
        torch.tensor([[[40.0], [20.0]]]),
    ]
    targets = [torch.tensor([[3.0], [1.0]]), torch.tensor([[4.0], [2.0]])]
    coords = [torch.tensor([[3.0, 3.0], [1.0, 1.0]]), torch.tensor([[4.0, 4.0], [2.0, 2.0]])]
    times = [torch.tensor([30, 10]), torch.tensor([40, 20])]
    row_ids = [torch.tensor([3, 1]), torch.tensor([4, 2])]

    pred, target, target_coords, target_times = _reassemble_validation_rows(
        predictions, targets, coords, times, row_ids
    )

    assert pred[:, :, 0].tolist() == [[10.0, 20.0, 30.0, 40.0]]
    assert target[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert target_coords[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.array_equal(target_times.astype(np.int64), np.array([10, 20, 30, 40]))


def test_reassemble_validation_rows_rejects_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="duplicate stable row ids"):
        _reassemble_validation_rows(
            [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)],
            [torch.zeros(1, 1), torch.zeros(1, 1)],
            [torch.zeros(1, 2), torch.zeros(1, 2)],
            [torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.int64)],
            [torch.tensor([0]), torch.tensor([0])],
        )
