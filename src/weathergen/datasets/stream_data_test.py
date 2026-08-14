# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for stream data initialization and reproducibility."""

import numpy as np
import torch

from weathergen.datasets.stream_data import StreamData, spoof


def test_empty_targets_use_writer_compatible_defaults():
    """A skipped target step remains a valid empty output for validation writing."""
    stream_data = StreamData(idx=0, input_steps=1, output_steps=2, healpix_cells=12)

    for coords, times, idxs_inv in zip(
        stream_data.target_coords_raw,
        stream_data.target_times_raw,
        stream_data.idxs_inv,
        strict=True,
    ):
        assert isinstance(coords, torch.Tensor)
        assert coords.shape == (0, 2)
        assert times.shape == (0,)
        assert idxs_inv is None


def _spoof(rng: np.random.Generator):
    return spoof(
        healpix_level=2,
        datetime=np.datetime64("2020-01-01T00:00:00"),
        geoinfo_size=3,
        num_channels=4,
        rng=rng,
    )


def test_spoof_is_reproducible_for_equal_rng():
    """The same seeded rng yields identical spoofed tokens (the selected cells)."""
    first = _spoof(np.random.default_rng(123))
    second = _spoof(np.random.default_rng(123))

    assert np.array_equal(first.coords, second.coords)


def test_spoof_uses_passed_rng_not_global():
    """spoof draws from the supplied rng, so the unseeded global RNG cannot affect it."""
    np.random.seed(0)
    seeded = _spoof(np.random.default_rng(123))

    np.random.seed(9999)  # perturb the global RNG that spoof no longer uses
    same = _spoof(np.random.default_rng(123))

    assert np.array_equal(seeded.coords, same.coords)
