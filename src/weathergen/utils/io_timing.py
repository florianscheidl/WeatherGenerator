# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Opt-in wall-clock timing of data reading I/O and preprocessing.

Environment variables (read once at import, so set them before the process and its data
loader workers start):

- ``WEATHERGEN_IO_TIMING=1`` enables timing. When disabled, ``io_timer`` is a no-op.
- ``WEATHERGEN_IO_TIMING_DIR`` is the directory for the JSON Lines output (default
  ``./io_timing``). Every process (rank, data loader worker) writes its own file
  ``io_timing_<host>_<pid>.jsonl`` with one record per timed call.
  Each record holds ``stream``, ``op``, ``path`` (the ops of the enclosing timers and this
  one, joined by ``PATH_SEP``), ``t_start`` (epoch seconds), ``dt`` (seconds), ``pid``, ``host``.
- ``WEATHERGEN_IO_TIMING_LOG=0`` suppresses the per-call log lines on the
  ``weathergen.io_timing`` logger (INFO, ``io_timing : <stream> : <op> : <seconds>``).

Analyze the output with ``scripts/analyze_io_timing.py``.
"""

import json
import logging
import os
import pathlib
import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import TextIO

_logger = logging.getLogger("weathergen.io_timing")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


IO_TIMING_ENABLED = _env_flag("WEATHERGEN_IO_TIMING", "0")
_LOG_ENABLED = _env_flag("WEATHERGEN_IO_TIMING_LOG", "1")
_OUT_DIR = pathlib.Path(os.environ.get("WEATHERGEN_IO_TIMING_DIR", "io_timing"))
_HOST = socket.gethostname()

# Separator of the ops of the enclosing timers in a record's ``path``.
PATH_SEP = " > "

# Per-thread stack of the enclosing timers' ops, to record nesting.
_stack = threading.local()
# Output file of the current process; reopened after a fork (data loader workers).
_out: TextIO | None = None
_out_pid: int | None = None
_out_lock = threading.Lock()


def _write(record: dict) -> None:
    global _out, _out_pid
    with _out_lock:
        pid = os.getpid()
        if _out is None or _out_pid != pid:
            _OUT_DIR.mkdir(parents=True, exist_ok=True)
            fname = _OUT_DIR / f"io_timing_{_HOST}_{pid}.jsonl"
            # line buffered: workers can be terminated without running exit handlers
            _out = open(fname, "a", buffering=1)  # noqa: SIM115
            _out_pid = pid
        _out.write(json.dumps(record) + "\n")


@contextmanager
def io_timer(stream: str, op: str) -> Generator[None]:
    """Record the wall-clock time spent in the ``with`` block as ``op`` of ``stream``."""
    if not IO_TIMING_ENABLED:
        yield
        return
    ops = getattr(_stack, "ops", None)
    if ops is None:
        ops = _stack.ops = []
    ops.append(op)
    path = PATH_SEP.join(ops)
    t_wall = time.time()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        ops.pop()
        _write(
            {
                "stream": stream,
                "op": op,
                "path": path,
                "t_start": t_wall,
                "dt": dt,
                "pid": os.getpid(),
                "host": _HOST,
            }
        )
        if _LOG_ENABLED:
            _logger.info(f"io_timing : {stream} : {op} : {dt:.6f}")
