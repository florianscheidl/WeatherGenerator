"""Tests for scripts/io_timing_to_trace.py (standalone script, loaded by path)."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "io_timing_to_trace", Path(__file__).parents[1] / "scripts" / "io_timing_to_trace.py"
)
trace_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(trace_mod)


def rec(stream, op, t, dt, pid=1, tid=1, path=None):
    return {
        "stream": stream,
        "op": op,
        "path": path or op,
        "t_start": t,
        "dt": dt,
        "pid": pid,
        "tid": tid,
        "host": "h",
    }


RECORDS = [
    rec("nsys-window", "startup open", 100.0, 0.0),
    rec("ERA5", "inner", 101.0, 0.5, pid=2, path="outer > inner"),
    rec("ERA5", "outer", 100.5, 2.0, pid=2),
    rec("nsys-window", "startup close", 103.0, 0.0),
    rec("ERA5", "late", 110.0, 1.0, pid=2),
]


def xs(trace):
    return [e for e in trace["traceEvents"] if e["ph"] == "X"]


def test_convert_events_and_units():
    trace = trace_mod.convert(RECORDS)
    by_name = {e["name"]: e for e in xs(trace)}
    assert by_name["outer"]["ts"] == pytest.approx(0.5e6)  # relative to the first record
    assert by_name["outer"]["dur"] == pytest.approx(2.0e6)
    # nested call stays inside its parent on the same track
    assert by_name["inner"]["pid"] == by_name["outer"]["pid"]
    assert by_name["inner"]["tid"] == by_name["outer"]["tid"]
    assert by_name["nsys window: startup"]["dur"] == pytest.approx(3.0e6)
    json.dumps(trace)


def test_window_filter():
    names = {e["name"] for e in xs(trace_mod.convert(RECORDS, window="startup"))}
    assert names == {"inner", "outer", "nsys window: startup"}


def test_unknown_window_exits():
    with pytest.raises(SystemExit):
        trace_mod.convert(RECORDS, window="steady")


def test_unclosed_window_extends_to_last_event():
    spans = trace_mod.window_spans([rec("nsys-window", "steady open", 5.0, 0.0)])
    assert spans["steady"] == (5.0, 5.0)
