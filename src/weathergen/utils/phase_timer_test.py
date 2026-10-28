import json

import pytest
from omegaconf import OmegaConf

from weathergen.utils.phase_timer import PhaseTimer, format_report, is_enabled


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def _self_seconds(timer: PhaseTimer) -> dict[str, float]:
    return {key: entry["self_s"] for key, entry in timer.report()["phases"].items()}


def test_phases_are_exclusive_and_add_up(clock):
    timer = PhaseTimer("imports", clock=clock)
    clock.tick(1.0)
    timer.switch("setup")
    clock.tick(2.0)
    timer.switch("train")
    clock.tick(0.5)
    with timer.phase("checkpoint"):
        clock.tick(3.0)
        with timer.phase("io"):
            clock.tick(0.25)
    clock.tick(0.5)

    seconds = _self_seconds(timer)
    assert seconds == {
        "imports": 1.0,
        "setup": 2.0,
        "train": 1.0,
        "train/checkpoint": 3.0,
        "train/checkpoint/io": 0.25,
    }
    report = timer.report()
    assert report["total_s"] - (report["launch_s"] or 0.0) == pytest.approx(7.25)


def test_iterate_times_only_the_waits(clock):
    timer = PhaseTimer("train", clock=clock)

    def batches():
        for i in range(3):
            clock.tick(1.0 if i == 0 else 0.1)  # time to produce the batch
            yield i

    for _ in timer.iterate(batches(), "data_wait"):
        clock.tick(2.0)  # step

    seconds = _self_seconds(timer)
    assert seconds["train"] == pytest.approx(6.0)
    assert seconds["train/data_wait_first"] == pytest.approx(1.0)
    assert seconds["train/data_wait"] == pytest.approx(0.2)
    counts = {key: entry["count"] for key, entry in timer.report()["phases"].items()}
    # two waits for items, one for the exhausted iterator
    assert counts["train/data_wait"] == 3


def test_iterate_break_leaves_the_stack_balanced(clock):
    timer = PhaseTimer("validate", clock=clock)
    for i in timer.iterate(range(10), "data_wait"):
        if i == 2:
            break
    timer.switch("teardown")  # asserts that no nested phase is open


def test_sync_runs_at_both_boundaries(clock):
    calls = []
    timer = PhaseTimer("inference", clock=clock, sync_fn=lambda: calls.append(clock.now))
    clock.tick(1.0)
    with timer.phase("write", sync=True):
        clock.tick(1.0)
    assert calls == [1.0, 2.0]
    with timer.phase("other"):
        pass
    assert len(calls) == 2


def test_disabled_timer_is_a_no_op(clock):
    timer = PhaseTimer("imports", clock=clock)
    timer.configure(enabled=False)
    data = [1, 2]
    assert timer.iterate(data, "data_wait") is data
    with timer.phase("write", sync=True):
        clock.tick(1.0)
    timer.switch("train")
    assert timer.report() == {}


def test_exit_report_is_written_for_the_configured_rank(clock, tmp_path):
    timer = PhaseTimer("imports", clock=clock)
    timer.configure(enabled=True, out_dir=tmp_path, rank=3)
    clock.tick(1.0)
    timer._write_at_exit()

    report = json.loads((tmp_path / "phase_timing_rank3.json").read_text())
    assert report["rank"] == 3
    assert report["phases"]["imports"]["self_s"] == pytest.approx(1.0)
    assert "imports" in format_report(report)


def test_enabled_by_performance_logging():
    assert not is_enabled(OmegaConf.create({}))
    assert not is_enabled(OmegaConf.create({"performance_logging": {"throughput": {}}}))
    cf = OmegaConf.create({"performance_logging": {"phase_timing": {"enabled": True}}})
    assert is_enabled(cf)
