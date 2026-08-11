# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from weathergen.model.spatial_parallel import (
    ensure_packed_cell_shard,
    zero_gradient_module_dependency,
)


def test_zero_gradient_module_dependency_visits_parameters_without_changing_gradient() -> None:
    module = torch.nn.Linear(2, 3)

    dependency = zero_gradient_module_dependency(module, torch.zeros((1, 2)))
    dependency.backward()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_ensure_packed_cell_shard_selects_global_input() -> None:
    cell_lens = torch.tensor([[1, 0, 2, 1], [0, 2, 1, 0]], dtype=torch.int32)
    global_tokens = torch.arange(cell_lens.sum(), dtype=torch.float32).unsqueeze(1)

    local_tokens, local_lens = ensure_packed_cell_shard(
        global_tokens,
        cell_lens.flatten(),
        num_cells=4,
        cell_start=2,
        cell_end=4,
    )

    assert local_lens.tolist() == [2, 1, 1, 0]
    assert local_tokens.squeeze(1).tolist() == [1, 2, 3, 6]


def test_ensure_packed_cell_shard_preserves_local_input() -> None:
    cell_lens = torch.tensor([[1, 0, 2, 1], [0, 2, 1, 0]], dtype=torch.int32)
    local_tokens = torch.tensor([[10.0], [11.0], [12.0], [13.0]])

    selected, local_lens = ensure_packed_cell_shard(
        local_tokens,
        cell_lens.flatten(),
        num_cells=4,
        cell_start=2,
        cell_end=4,
    )

    assert selected is local_tokens
    assert local_lens.tolist() == [2, 1, 1, 0]


def test_ensure_packed_cell_shard_rejects_unknown_residency() -> None:
    with pytest.raises(ValueError, match="expected either 4 local or 7 global rows"):
        ensure_packed_cell_shard(
            torch.zeros(5, 1),
            torch.tensor([1, 0, 2, 1, 0, 2, 1, 0], dtype=torch.int32),
            num_cells=4,
            cell_start=2,
            cell_end=4,
        )
