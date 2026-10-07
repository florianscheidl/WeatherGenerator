# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from weathergen.utils import nsys_windows
from weathergen.utils.nsys_windows import NsysWindows


@pytest.fixture
def calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(nsys_windows.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(nsys_windows.torch.cuda.profiler, "start", lambda: calls.append("start"))
    monkeypatch.setattr(nsys_windows.torch.cuda.profiler, "stop", lambda: calls.append("stop"))
    return calls


def run_loop(w: NsysWindows, steps: int, log: list[str]) -> None:
    w.before_data_iter()
    for step in range(steps):
        w.before_step(step)
        w.after_forward(step)
        w.after_step(step)
        log.append(f"step{step}:{w._open}")


def test_disabled_without_config(monkeypatch):
    monkeypatch.delenv(nsys_windows.ENV_VAR, raising=False)
    assert NsysWindows.from_config({}) is None
    assert NsysWindows.from_config({"profiling": {"nsys_windows": {"enabled": False}}}) is None


def test_two_windows_in_order(calls):
    log: list[str] = []
    run_loop(NsysWindows(warmup_steps=2, steady_steps=4), 10, log)
    assert calls == ["start", "stop", "start", "stop"]
    # startup closes after the first forward; steady covers steps 2..5
    assert [e for e in log if e.endswith("None")] == [
        "step0:None",
        "step1:None",
        "step5:None",
        "step6:None",
        "step7:None",
        "step8:None",
        "step9:None",
    ]
    assert log[2] == "step2:steady" and log[4] == "step4:steady"


def test_windows_open_once(calls):
    w = NsysWindows(warmup_steps=1, steady_steps=1)
    run_loop(w, 3, [])
    run_loop(w, 3, [])  # second mini-epoch
    assert calls == ["start", "stop", "start", "stop"]


def test_only_steady(calls):
    run_loop(NsysWindows(startup=False, warmup_steps=1, steady_steps=2), 5, [])
    assert calls == ["start", "stop"]


def test_env_overrides_config(monkeypatch):
    cf = {"profiling": {"nsys_windows": {"enabled": False, "startup": True, "warmup_steps": 3}}}
    monkeypatch.setenv(nsys_windows.ENV_VAR, "steady")
    w = NsysWindows.from_config(cf)
    assert w is not None and not w.startup and w.steady and w.warmup_steps == 3
    monkeypatch.setenv(nsys_windows.ENV_VAR, "startup,steady")
    w = NsysWindows.from_config({})
    assert w is not None and w.startup and w.steady


def test_env_unknown_window(monkeypatch):
    monkeypatch.setenv(nsys_windows.ENV_VAR, "startup,bogus")
    with pytest.raises(ValueError):
        NsysWindows.from_config({})
