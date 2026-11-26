# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import types
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

# the trainer imports the model, which needs flash-attn (not in the cpu extra)
pytest.importorskip("flash_attn")

import weathergen.train.trainer as trainer_module  # noqa: E402
from weathergen.train import profiling_trainer  # noqa: E402
from weathergen.train.profiling_trainer import ProfilingTrainer  # noqa: E402
from weathergen.train.trainer import Trainer  # noqa: E402
from weathergen.utils import performance  # noqa: E402
from weathergen.utils.nsys_windows import NsysWindows  # noqa: E402
from weathergen.utils.phase_timer import PhaseTimer  # noqa: E402
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


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _phase_timed_trainer(stage: str, clock: _Clock) -> tuple[ProfilingTrainer, PhaseTimer]:
    timer = PhaseTimer("setup/other", clock=clock, sync_fn=lambda: None)
    trainer = ProfilingTrainer(OmegaConf.create({"metrics": 1}), ProfilingConfig())
    trainer.cf = OmegaConf.create({"stage": stage})
    trainer.training_cfg = OmegaConf.create({"num_mini_epochs": 1})
    return trainer, timer


def _seconds(timer: PhaseTimer) -> dict[str, float]:
    return {key: entry["self_s"] for key, entry in timer.report()["phases"].items()}


def test_phase_timer_splits_the_training_loop():
    clock = _Clock()
    trainer, timer = _phase_timed_trainer("train", clock)

    def batches():
        for i in range(2):
            clock.now += 1.0  # loading
            yield i

    def fake_validate(trainer, *args):
        for _bidx, _batch in trainer.validation_batches(iter([0])):
            clock.now += 4.0

    with (
        patch.object(profiling_trainer, "timer", timer),
        patch.object(Trainer, "validate_before_training", lambda self: None),
        patch.object(Trainer, "validate", fake_validate),
        patch.object(Trainer, "save_model", lambda *args: setattr(clock, "now", clock.now + 8.0)),
    ):
        trainer.validate_before_training()
        for _bidx, _batch in trainer.train_batches(batches()):
            clock.now += 2.0  # step
        trainer.validate(0, None, 1)
        trainer.save_model(1)

    assert _seconds(timer) == {
        "setup/other": 0.0,
        "train": 4.0,
        "train/data_wait_first": 1.0,
        "train/data_wait": 1.0,
        "train/validate": 4.0,
        "train/validate/data_wait_first": 0.0,
        "train/validate/data_wait": 0.0,
        "train/checkpoint": 8.0,
        "teardown": 0.0,
    }


def test_phase_timer_bills_model_construction_to_its_own_phase():
    clock = _Clock()
    trainer, timer = _phase_timed_trainer("train", clock)
    original = trainer_module.init_model_and_shard

    def fake_init_model_and_shard(*args):
        clock.now += 3.0
        return "model", "params"

    def fake_run(trainer, *args):
        clock.now += 1.0
        assert trainer_module.init_model_and_shard(None) == ("model", "params")
        clock.now += 0.5

    with (
        patch.object(profiling_trainer, "timer", timer),
        patch.object(trainer_module, "init_model_and_shard", fake_init_model_and_shard),
        patch.object(Trainer, "run", fake_run),
    ):
        trainer.run(None, None)
        # patched only while running
        assert trainer_module.init_model_and_shard is fake_init_model_and_shard
    assert trainer_module.init_model_and_shard is original

    assert _seconds(timer) == {"setup/other": 1.5, "setup/model": 3.0}


def test_phase_timer_makes_inference_a_root_phase():
    clock = _Clock()
    trainer, timer = _phase_timed_trainer("inference", clock)

    with (
        patch.object(profiling_trainer, "timer", timer),
        patch.object(Trainer, "validate", lambda *args: setattr(clock, "now", clock.now + 5.0)),
    ):
        trainer.validate(0, None, 1)

    assert _seconds(timer) == {"setup/other": 0.0, "inference": 5.0, "teardown": 0.0}


def _step_timing_trainer(enabled: bool, clock: _Clock) -> None:
    """A trainer whose loader waits 0.25 s per batch and whose steps take 1 s on `clock`."""
    train_logging = OmegaConf.create({"terminal": 2, "metrics": 3})
    trainer = ProfilingTrainer(train_logging, ProfilingConfig(enabled=True))
    trainer.cf = OmegaConf.create({"rank": 2})
    trainer.step_timing = enabled

    def slow_batches():
        for i in range(4):
            clock.now += 0.25
            yield i

    # the same chain as in train_batches, without the other wrappers
    batches = trainer._timed_wait(slow_batches()) if enabled else slow_batches()
    for _bidx, _batch in trainer._with_step_timing(enumerate(batches)):
        clock.now += 1.0


def test_step_timing_logs_one_line_per_step(caplog, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(profiling_trainer.time, "perf_counter", clock)

    with caplog.at_level("INFO", logger=profiling_trainer.logger.name):
        _step_timing_trainer(True, clock)

    lines = [r.getMessage() for r in caplog.records if "Step time rank" in r.getMessage()]
    assert len(lines) == 4
    assert lines[0].startswith("Step time rank 2 batch 0: wait_ms=250.0 step_ms=1000.0 ")
    # log steps: batch % 2 == 0 or batch % 3 == 0
    assert [" log=1 " in line for line in lines] == [True, False, True, True]
    assert all(" period_ms=1250.0 " in line for line in lines)


def test_step_timing_is_off_by_default(caplog):
    clock = _Clock()

    with caplog.at_level("INFO", logger=profiling_trainer.logger.name):
        _step_timing_trainer(False, clock)

    assert not [r for r in caplog.records if "Step time rank" in r.getMessage()]


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


def test_inference_nsys_windows_follow_the_model_calls():
    events = []

    class RecordingWindows(NsysWindows):
        def _start(self, name):
            if self._open is None and name not in self._done:
                events.append(f"open {name}")
                self._open = name

        def _stop(self, name):
            if self._open == name:
                events.append(f"close {name}")
                self._open = None
                self._done.add(name)

    def fake_rollout(trainer, *args):
        events.append("read batch")
        for step in range(6):
            trainer.model(torch.zeros(1))
            events.append(f"chunk {step}")

    trainer = ProfilingTrainer(OmegaConf.create({"metrics": 1}), ProfilingConfig())
    trainer.cf = OmegaConf.create({"stage": "inference"})
    trainer.model = torch.nn.Identity()
    trainer.nsys_windows = RecordingWindows(warmup_steps=2, steady_steps=3)
    with patch.object(Trainer, "validate", fake_rollout):
        trainer.validate(0, None, 1)

    assert events == [
        "open startup",
        "read batch",
        "close startup",
        "chunk 0",
        "chunk 1",
        "open steady",
        "chunk 2",
        "chunk 3",
        "chunk 4",
        "close steady",
        "chunk 5",
    ]
    assert trainer.nsys_windows.done
    assert not trainer.model._forward_pre_hooks and not trainer.model._forward_hooks


@pytest.mark.parametrize(
    ("output", "with_targets", "decodes"),
    [
        ({"streams": ["latent"], "skip_unwritten_decoders": True}, False, False),
        ({"streams": ["latent"]}, False, True),
        ({"streams": ["latent"], "skip_unwritten_decoders": True}, True, True),
        ({"streams": ["latent", "ERA5"], "skip_unwritten_decoders": True}, False, True),
        ({"streams": None, "skip_unwritten_decoders": True}, False, True),
    ],
)
def test_physical_decoders_are_skipped_only_when_nothing_uses_them(output, with_targets, decodes):
    mode_cfg = OmegaConf.create({"output": output, "with_targets": with_targets})
    assert Trainer._decodes_physical_output(mode_cfg) is decodes


def test_physical_decoding_switch_skips_predict_decoders_and_restores():
    from weathergen.model.model import Model  # noqa: PLC0415

    model = types.SimpleNamespace(decode_physical=True, pred_heads={"ERA5": None})
    output = types.SimpleNamespace(fstep_idx=lambda step: step)
    with Model.physical_decoding(model, False):
        assert Model.predict_decoders(model, None, 0, None, None, output) is output
    assert model.decode_physical is True
