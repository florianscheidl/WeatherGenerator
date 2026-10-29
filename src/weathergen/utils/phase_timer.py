# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Wall-clock time per phase of a run (imports, setup, data waits, train/inference loop, writing),
with negligible overhead.

The timer attributes every interval of the main thread's wall time to exactly one phase: the
innermost open one. Phases therefore add up to the total, and a phase measures what the main
thread is blocked on, so data loading that overlaps with compute costs nothing and only shows
up while the loop waits for the next batch (`iterate`).

Phases are either root phases, which follow each other (`switch`), or nested phases opened
inside them (`phase`, `iterate`), keyed by their path, e.g. "train/data_wait". The time of a
key is its self time, excluding nested phases.

Device work is asynchronous: a host timer bills queued GPU work to whichever phase next waits
for it. `sync=True` synchronizes the device at a phase boundary; use it only where the code
waits for the device anyway (writing, logging, checkpoints) or at one-off boundaries.

The timer is created when this module is imported, in the root phase "imports", so it has to
be imported before the heavy imports of the entry point. It records until `configure`, which
either enables it (keeping what was recorded) or turns all calls into no-ops. Enabled by
`performance_logging.phase_timing.enabled` (`is_enabled`); the phases inside a run are timed
by `weathergen.train.profiling_trainer.ProfilingTrainer`. When enabled,
each rank writes `phase_timing_rank<rank>.json` at process exit; the time from process
creation (or $WEATHERGEN_LAUNCH_TIME, epoch seconds, e.g. set by a launch script) to this
module's import is reported as "launch".

Only the main thread may use the timer. It does not import torch, so that it can time it.
"""

import atexit
import contextlib
import json
import logging
import os
import socket
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_NULL_CONTEXT = contextlib.nullcontext()


def _synchronize_device() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.accelerator.is_available():
        torch.accelerator.synchronize()


class PhaseTimer:
    def __init__(
        self,
        root_phase: str,
        clock: Callable[[], float] = time.perf_counter,
        sync_fn: Callable[[], None] = _synchronize_device,
    ) -> None:
        self._clock = clock
        self._sync_fn = sync_fn
        self._enabled = True
        self._wall_start = time.time()
        self._t_start = clock()
        self._t_last = self._t_start
        # keys (paths) of the open phases, root first
        self._stack: list[str] = [root_phase]
        self._seconds: dict[str, float] = {root_phase: 0.0}
        self._counts: dict[str, int] = {root_phase: 1}
        self._out_file: Path | None = None
        self._pid = os.getpid()
        self._rank = 0
        self._written = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _advance(self) -> None:
        now = self._clock()
        self._seconds[self._stack[-1]] += now - self._t_last
        self._t_last = now

    def _start(self, key: str) -> None:
        if key in self._counts:
            self._counts[key] += 1
        else:
            self._seconds[key] = 0.0
            self._counts[key] = 1

    def _push(self, key: str) -> None:
        self._advance()
        self._stack.append(key)
        self._start(key)

    def _pop(self) -> None:
        self._advance()
        self._stack.pop()

    def switch(self, phase: str, sync: bool = False) -> None:
        """End the current root phase and start the root phase `phase`."""
        if not self._enabled:
            return
        assert len(self._stack) == 1, f"switch({phase!r}) inside nested phases {self._stack}"
        if sync:
            self._sync_fn()
        self._advance()
        self._stack[0] = phase
        self._start(phase)

    def phase(self, name: str, sync: bool = False) -> contextlib.AbstractContextManager:
        """Context manager timing a phase nested in the current one."""
        if not self._enabled:
            return _NULL_CONTEXT
        return self._nested(name, sync)

    @contextlib.contextmanager
    def _nested(self, name: str, sync: bool) -> Iterator[None]:
        # synchronizing before entering bills already queued device work to the parent phase
        if sync:
            self._sync_fn()
        self._push(f"{self._stack[-1]}/{name}")
        try:
            yield
        finally:
            if sync:
                self._sync_fn()
            self._pop()

    def iterate(self, iterable: Iterable[T], name: str) -> Iterable[T]:
        """
        Iterate over `iterable`, timing each wait for the next item as the nested phase `name`.

        The wait for the first item is reported separately as `<name>_first`, since it includes
        one-off costs such as starting the data loader workers.
        """
        if not self._enabled:
            return iterable
        return self._iterate(iterable, name)

    def _iterate(self, iterable: Iterable[T], name: str) -> Iterator[T]:
        parent = self._stack[-1]
        key_first = f"{parent}/{name}_first"
        key = f"{parent}/{name}"
        iterator = iter(iterable)
        first = True
        while True:
            self._push(key_first if first else key)
            first = False
            try:
                item = next(iterator)
            except StopIteration:
                return
            finally:
                self._pop()
            yield item

    def configure(self, enabled: bool, out_dir: Path | None = None, rank: int = 0) -> None:
        """
        Enable the timer and write the report of this rank to `out_dir` at process exit, or
        disable it, discarding what was recorded.
        """
        self._enabled = enabled
        if not enabled:
            self._seconds.clear()
            self._counts.clear()
            return
        self._rank = rank
        self._pid = os.getpid()
        if out_dir is not None:
            self._out_file = Path(out_dir) / f"phase_timing_rank{rank}.json"

    def report(self) -> dict:
        """Times recorded so far, per phase and in total, in seconds."""
        if not self._enabled:
            return {}
        self._advance()
        launch = _launch_seconds(self._wall_start)
        measured = self._t_last - self._t_start
        return {
            "rank": self._rank,
            "host": socket.gethostname(),
            "pid": self._pid,
            "wall_start": self._wall_start,
            "launch_s": launch,
            "total_s": (launch or 0.0) + measured,
            "phases": {
                key: {"self_s": self._seconds[key], "count": self._counts[key]}
                for key in self._seconds
            },
        }

    def write_report(self) -> None:
        """
        Write this rank's report, once. Runs at process exit; call it directly before an exit
        that skips the exit handlers (`os._exit`).
        """
        # forked or spawned workers inherit or re-import the module, but only report for the
        # configured process
        if not self._enabled or self._written or self._out_file is None or os.getpid() != self._pid:
            return
        self._written = True
        report = self.report()
        try:
            self._out_file.parent.mkdir(parents=True, exist_ok=True)
            self._out_file.write_text(json.dumps(report, indent=1))
        except OSError as e:
            logger.warning(f"Could not write phase timing to {self._out_file}: {e}")
        if self._rank == 0:
            logger.info(f"Phase timing (written to {self._out_file}):\n" + format_report(report))


def is_enabled(cf) -> bool:
    """Whether the run config asks for phase timing (off unless set)."""
    section = (cf.get("performance_logging") or {}).get("phase_timing") or {}
    return section.get("enabled", False)


def _launch_seconds(wall_start: float) -> float | None:
    """Seconds from process creation (or $WEATHERGEN_LAUNCH_TIME) to `wall_start`."""
    launch_time = os.environ.get("WEATHERGEN_LAUNCH_TIME")
    if launch_time is not None:
        return wall_start - float(launch_time)
    try:
        import psutil

        return wall_start - psutil.Process().create_time()
    except (ImportError, OSError):
        return None


def format_report(report: dict) -> str:
    """Table of self and inclusive time per phase, in the order the phases first ran."""
    total = report["total_s"]
    phases = report["phases"]
    rows = []
    if report["launch_s"] is not None:
        rows.append(("launch", report["launch_s"], report["launch_s"], 1))
    for key, entry in phases.items():
        inclusive = sum(
            other["self_s"]
            for other_key, other in phases.items()
            if other_key == key or other_key.startswith(key + "/")
        )
        rows.append((key, entry["self_s"], inclusive, entry["count"]))
    width = max(len(row[0]) for row in rows + [("total", 0, 0, 0)])
    lines = [f"{'phase':<{width}}  {'self [s]':>10}  {'incl [s]':>10}  {'%':>5}  {'count':>7}"]
    for key, self_s, inclusive, count in rows:
        percent = 100.0 * inclusive / total if total > 0 else 0.0
        lines.append(
            f"{key:<{width}}  {self_s:>10.3f}  {inclusive:>10.3f}  {percent:>5.1f}  {count:>7d}"
        )
    lines.append(f"{'total':<{width}}  {total:>10.3f}  {total:>10.3f}  {100.0:>5.1f}")
    return "\n".join(lines)


timer = PhaseTimer("imports")
atexit.register(timer.write_report)
