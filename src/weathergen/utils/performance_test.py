# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from contextlib import contextmanager

import pytest

from weathergen.utils import performance
from weathergen.utils.performance import annotate_next, nvtx_context


@pytest.fixture
def ranges(monkeypatch):
    log: list[str] = []
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_push", lambda n: log.append(n))
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_pop", lambda: log.append("pop"))
    return log


def test_nvtx_context_disabled(ranges):
    for cf in ({}, {"profiling": None}, {"profiling": {"nvtx_annotate": False}}):
        with nvtx_context(cf)("x"):
            pass
    assert ranges == []


def test_nvtx_context_enabled(ranges):
    with nvtx_context({"profiling": {"nvtx_annotate": True}})("x"):
        pass
    assert ranges == ["x", "pop"]


def test_annotate_next_wraps_each_fetch():
    log: list[str] = []

    @contextmanager
    def annotate(name):
        log.append(f"enter {name}")
        yield
        log.append(f"exit {name}")

    for item in annotate_next([1, 2], annotate, "next"):
        log.append(f"body {item}")

    # the loop body runs outside the range; the final StopIteration fetch is closed too
    assert log == [
        "enter next",
        "exit next",
        "body 1",
        "enter next",
        "exit next",
        "body 2",
        "enter next",
        "exit next",
    ]
