from types import SimpleNamespace

import numpy as np
import pytest

from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    check_requested_channels,
)


def _reader(length=10, source=(), target=()):
    return SimpleNamespace(
        length=lambda: length, source_channels=list(source), target_channels=list(target)
    )


def test_check_requested_channels_raises_without_matches():
    with pytest.raises(ValueError, match="source channels"):
        check_requested_channels("ERA5", {"source": ["t_850"]}, _reader(target=["t_850"]), "f")
    with pytest.raises(ValueError, match="target channels"):
        check_requested_channels("ERA5", {"target": ["t_850"]}, _reader(source=["t_850"]), "f")


def test_check_requested_channels_accepts_valid_configs():
    # channels found
    check_requested_channels(
        "ERA5",
        {"source": ["t_850"], "target": ["t_850"]},
        _reader(source=["t_850"], target=["t_850"]),
        "f",
    )
    # nothing requested explicitly, e.g. a diagnostic stream without source channels
    check_requested_channels("ERA5", {}, _reader(target=["t_850"]), "f")
    # stream skipped on purpose (empty reader), e.g. outside the time window
    check_requested_channels("ERA5", {"source": ["t_850"]}, _reader(length=0), "f")


def _sampler(batches_valid):
    """Sampler over 3 samples yielding 2 batches; batches_valid gives validity per _get_batch."""
    sampler = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
    sampler.len = 2
    sampler.batch_size = 1
    sampler.mode_cfg = {"training_mode": ["masking"]}
    sampler.worker_workset = lambda: (0, 2)
    sampler.reset = lambda: (np.arange(3), [0, 0])
    validity = iter(batches_valid)

    def get_batch(idx, num_forecast_steps):
        empty = not next(validity)
        return SimpleNamespace(
            sources_empty=lambda: empty, is_nan=lambda: False, targets_empty=lambda: empty
        )

    sampler._get_batch = get_batch
    return sampler


def test_iteration_skips_empty_batches():
    batches = list(_sampler([False, False, True, True]))
    assert len(batches) == 2


def test_iteration_raises_without_any_valid_batch():
    with pytest.raises(RuntimeError, match="No valid batch"):
        list(_sampler([False] * 10))
