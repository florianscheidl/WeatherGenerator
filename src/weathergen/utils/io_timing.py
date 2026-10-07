# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Opt-in wall-clock timing of data reading I/O and preprocessing.

On the HPC clusters switch it on with ``launch-slurm.py --io-timing`` (and
``--io-timing-memory [SECONDS]`` for memory samples), which sets the variables below for the job
and writes to ``logs/<run-id>/profiling/io_timing/``.

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

- ``WEATHERGEN_IO_TIMING_NVTX=1`` additionally makes every ``io_timer`` open an NVTX range
  ``io:<stream> : <op>`` for Nsight Systems, independent of ``WEATHERGEN_IO_TIMING``. It is
  also on in nsys runs launched with ``launch-slurm.py --nsys-profiling`` (which sets
  ``WEATHERGEN_NSYS_PROFILING=1``). Nested timers give nested ranges.

- ``WEATHERGEN_IO_TIMING_MEMORY=1`` starts a background thread in every process that uses
  ``io_timer`` (rank, data loader workers) which appends host-memory samples to the same JSON
  Lines files (``stream`` = ``memory``); see ``weathergen.utils.memory_sampler``.
  ``WEATHERGEN_IO_TIMING_MEMORY_INTERVAL`` is the sampling period in seconds (default 1).

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
NVTX_ENABLED = _env_flag("WEATHERGEN_IO_TIMING_NVTX", "0") or _env_flag(
    "WEATHERGEN_NSYS_PROFILING", "0"
)
MEMORY_ENABLED = _env_flag("WEATHERGEN_IO_TIMING_MEMORY", "0")
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


def _reinit_after_fork() -> None:
    """Fresh lock and output file in a forked child (data loader worker).

    The memory sampler thread may hold the lock while the parent forks; the child would then
    inherit a lock that nobody can release.
    """
    global _out, _out_lock
    _out_lock = threading.Lock()
    _out = None


os.register_at_fork(after_in_child=_reinit_after_fork)


def write_record(record: dict) -> None:
    """Append ``record`` to this process's JSON Lines file."""
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
    """Record the wall-clock time spent in the ``with`` block as ``op`` of ``stream``.

    Also marks the block with an NVTX range if ``NVTX_ENABLED``.
    """
    if not (IO_TIMING_ENABLED or NVTX_ENABLED or MEMORY_ENABLED):
        yield
        return
    if MEMORY_ENABLED:
        _ensure_memory_sampler()
    if NVTX_ENABLED:
        _nvtx_push(f"io:{stream} : {op}")
    try:
        if IO_TIMING_ENABLED:
            with _timed(stream, op):
                yield
        else:
            yield
    finally:
        if NVTX_ENABLED:
            _nvtx_pop()


def _ensure_memory_sampler() -> None:
    from weathergen.utils import memory_sampler  # noqa: PLC0415 (circular import)

    memory_sampler.ensure_started()


def _nvtx_push(name: str) -> None:
    import torch  # noqa: PLC0415 (lazy: keep this module importable without torch)

    torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    import torch  # noqa: PLC0415

    torch.cuda.nvtx.range_pop()


@contextmanager
def _timed(stream: str, op: str) -> Generator[None]:
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
        write_record(
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


def io_event(stream: str, op: str) -> None:
    """Record an instantaneous marker (``dt`` = 0), e.g. the boundaries of a capture window."""
    if not IO_TIMING_ENABLED:
        return
    write_record(
        {
            "stream": stream,
            "op": op,
            "path": op,
            "t_start": time.time(),
            "dt": 0.0,
            "pid": os.getpid(),
            "host": _HOST,
        }
    )
