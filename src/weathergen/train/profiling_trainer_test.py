# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

# the trainer imports the model, which needs flash-attn (not in the cpu extra)
pytest.importorskip("flash_attn")

import weathergen.train.trainer as trainer_module  # noqa: E402
from weathergen.train import profiling_trainer  # noqa: E402
from weathergen.train.profiling_trainer import ProfilingTrainer  # noqa: E402
from weathergen.train.trainer import Trainer  # noqa: E402
from weathergen.utils import performance  # noqa: E402
from weathergen.utils.profiling import ProfilingConfig  # noqa: E402


def _validate_and_record(stage: str, profiling_cfg: ProfilingConfig) -> list[str]:
    """Run ProfilingTrainer.validate with Trainer.validate stubbed, return what happened."""
    events = []

    @contextlib.contextmanager
    def fake_snapshot_session(cf):
        events.append("snapshot_start")
        yield
        events.append("snapshot_end")

    trainer = ProfilingTrainer(OmegaConf.create({"metrics": 1}), profiling_cfg)
    trainer.cf = OmegaConf.create({"stage": stage})
    with (
        patch.object(Trainer, "validate", lambda *args: events.append("validate")),
        patch.object(profiling_trainer, "memory_snapshot_session", fake_snapshot_session),
    ):
        trainer.validate(0, None, 1)
    return events


@pytest.mark.parametrize("stop_after_profiling", [False, True])
def test_inference_is_recorded_in_memory_snapshot(stop_after_profiling):
    profiling_cfg = ProfilingConfig(
        enabled=True, memory_snapshot=True, stop_after_profiling=stop_after_profiling
    )

    events = _validate_and_record("inference", profiling_cfg)

    assert events == ["snapshot_start", "validate", "snapshot_end"]


def test_inference_memory_snapshot_needs_profiling_enabled():
    profiling_cfg = ProfilingConfig(enabled=False, memory_snapshot=True)

    assert _validate_and_record("inference", profiling_cfg) == ["validate"]


def test_training_validation_has_no_memory_snapshot():
    profiling_cfg = ProfilingConfig(enabled=True, memory_snapshot=True)

    assert _validate_and_record("train", profiling_cfg) == ["validate"]


def test_profiling_only_training_run_skips_validation():
    profiling_cfg = ProfilingConfig(enabled=True, memory_snapshot=True, stop_after_profiling=True)

    assert _validate_and_record("train", profiling_cfg) == []


class _FakeBatch:
    def to_device(self, device):
        pass

    def to_device_for_chunked_inference(self, device, loss_steps):
        pass

    def to_device_for_output_chunk(self, device, target_steps):
        pass


class _FakeCallable:
    def forward(self, *args):
        pass

    def compute(self, *args):
        pass


def _fake_validation_step(trainer, *args):
    """What Trainer.validate calls per batch, in order, without targets."""
    for _bidx, batch in trainer.validation_batches(iter([_FakeBatch()])):
        batch.to_device_for_chunked_inference(None, [0])
        trainer.target_and_aux_calculators_val["physical"].compute()
        for chunk in ([0], [1]):
            batch.to_device_for_output_chunk(None, chunk)
            trainer.model.forward()
            trainer_module.write_output()


def _annotated_inference_ranges(num_validations: int) -> list[list[str]]:
    """The nvtx ranges opened by each of `num_validations` stubbed inference validations."""
    ranges = []

    @contextlib.contextmanager
    def fake_nvtx_range(name):
        ranges.append(name)
        yield

    def write_output(*args):
        pass

    profiling_cfg = ProfilingConfig(enabled=True, nvtx_annotate=True)
    with (
        patch.object(performance, "nvtx_range", fake_nvtx_range),
        patch.object(trainer_module, "write_output", write_output),
        patch.object(Trainer, "validate", _fake_validation_step),
    ):
        trainer = ProfilingTrainer(OmegaConf.create({"metrics": 1}), profiling_cfg)
        trainer.cf = OmegaConf.create({"stage": "inference"})
        trainer.model = _FakeCallable()
        trainer.target_and_aux_calculators_val = {"physical": _FakeCallable()}

        per_validation = []
        for _ in range(num_validations):
            trainer.validate(0, None, 1)
            per_validation.append(ranges.copy())
            ranges.clear()
            # patched only while validating
            assert trainer_module.write_output is write_output
    return per_validation


def test_inference_validation_steps_are_nvtx_annotated():
    (ranges,) = _annotated_inference_ranges(1)

    chunk = ["to_device_for_output_chunk", "forward", "write_output"]
    assert ranges == [
        "validation",
        "next_batch",
        "batch_0",
        "to_device_for_chunked_inference",
        "target_aux:physical",
        *chunk,
        *chunk,
        "next_batch",  # the exhausted loader
    ]


def test_repeated_validation_is_annotated_once():
    first, second = _annotated_inference_ranges(2)

    assert second == first


def _train_and_record_wall_time(monkeypatch, num_mini_epochs: int) -> list[dict]:
    """Run ProfilingTrainer.train with Trainer.train stubbed, return the logged metrics."""
    logged = []
    clock = iter([110.0, 150.0, 160.0, 175.0])
    trainer = ProfilingTrainer(OmegaConf.create({"metrics": 1}), ProfilingConfig())
    trainer.cf = OmegaConf.create({"general": {"istep": 0}})
    monkeypatch.setattr(trainer, "_log_metrics", lambda stage, m, istep: logged.append(m))
    monkeypatch.setattr(profiling_trainer, "_process_start_time", lambda: 100.0)
    monkeypatch.setattr(profiling_trainer.time, "time", lambda: next(clock))
    with patch.object(Trainer, "train", lambda *args: None):
        for mini_epoch in range(num_mini_epochs):
            trainer.train(mini_epoch)
    return logged


def test_train_logs_startup_once_and_wall_time_per_mini_epoch(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)

    logged = _train_and_record_wall_time(monkeypatch, num_mini_epochs=2)

    assert logged == [
        {"performance.time.startup_s": 10.0},
        {"performance.time.train_s": 40.0, "performance.time.elapsed_s": 50.0},
        {"performance.time.train_s": 15.0, "performance.time.elapsed_s": 75.0},
    ]


def test_train_logs_startup_since_slurm_job_start(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_START_TIME", "70")

    logged = _train_and_record_wall_time(monkeypatch, num_mini_epochs=1)

    assert logged[0] == {
        "performance.time.startup_s": 10.0,
        "performance.time.job_startup_s": 40.0,
    }
