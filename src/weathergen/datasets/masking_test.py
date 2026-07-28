# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.datasets.masking import MaskData


def _add(mask_data: MaskData, mask: torch.Tensor) -> None:
    mask_data.add_mask(mask, {}, {}, [], 0, None, None)


def test_restrict_keeps_masks_and_metadata_in_sync():
    """The batch reads masks from .masks, the losses from .metadata[i].mask."""
    mask_data = MaskData()
    _add(mask_data, torch.tensor([True, True, True, True]))

    mask_data.restrict(torch.tensor([True, False, True, False]))

    assert mask_data.masks[0].tolist() == [True, False, True, False]
    assert mask_data.metadata[0].mask.tolist() == [True, False, True, False]


def test_restrict_handles_aliased_masks():
    """An `identity` source mask is the very same tensor object as its target mask, so
    restricting must not mutate in place -- that would apply to both entries at once."""
    shared = torch.tensor([True, True, False, True])
    mask_data = MaskData()
    _add(mask_data, shared)
    _add(mask_data, shared)

    mask_data.restrict(torch.tensor([True, False, True, True]))

    assert shared.tolist() == [True, True, False, True]
    for i in range(2):
        assert mask_data.masks[i].tolist() == [True, False, False, True]


def test_restrict_is_idempotent():
    keep = torch.tensor([True, False, True, False])
    mask_data = MaskData()
    _add(mask_data, torch.tensor([True, True, True, True]))

    mask_data.restrict(keep)
    mask_data.restrict(keep)

    assert mask_data.masks[0].tolist() == [True, False, True, False]
