# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Diagnostic cgroup-v2 host-memory timeline collection."""

import faulthandler
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "diagnostic.cgroup_memory"
_PROCESS_METRIC_PREFIX = "diagnostic.process_memory"
_REQUIRED_STAT_FIELDS = ("anon", "file", "shmem")
_REQUIRED_EVENT_FIELDS = ("high", "max", "oom", "oom_kill")
_PROCESS_STAT_FIELDS = {
    "Pss": "pss_bytes",
    "Pss_Anon": "pss_anon_bytes",
    "Pss_File": "pss_file_bytes",
    "Pss_Shmem": "pss_shmem_bytes",
    "Rss": "rss_bytes",
    "Private_Dirty": "private_dirty_bytes",
}


def resolve_cgroup_v2_path(
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_mount: Path = Path("/sys/fs/cgroup"),
    slurm_job_id: str | None = None,
) -> Path:
    """Resolve the cgroup-v2 directory that best represents the current Slurm job."""
    unified_path: str | None = None
    for line in proc_cgroup_path.read_text().splitlines():
        hierarchy_id, controllers, relative_path = line.split(":", maxsplit=2)
        if hierarchy_id == "0" and controllers == "":
            unified_path = relative_path
            break

    if unified_path is None:
        raise RuntimeError("The current process does not expose a cgroup-v2 hierarchy")

    relative_path = unified_path.lstrip("/")
    current_path = cgroup_mount / relative_path
    if not (current_path / "memory.current").is_file():
        # A cgroup namespace can expose its root as '/', while the mount itself is
        # already rooted at the process's delegated cgroup.
        if relative_path or not (cgroup_mount / "memory.current").is_file():
            raise RuntimeError(f"Cannot read cgroup-v2 memory metrics from {current_path}")
        current_path = cgroup_mount

    job_id = slurm_job_id if slurm_job_id is not None else os.environ.get("SLURM_JOB_ID")
    if not job_id or current_path == cgroup_mount:
        return current_path

    job_dir_names = {f"job_{job_id}", f"job-{job_id}.scope"}
    for candidate in (current_path, *current_path.parents):
        if candidate == cgroup_mount.parent:
            break
        if candidate.name in job_dir_names and (candidate / "memory.current").is_file():
            return candidate
        if candidate == cgroup_mount:
            break

    logger.warning(
        "Could not identify a job-level cgroup for SLURM_JOB_ID=%s; sampling %s",
        job_id,
        current_path,
    )
    return current_path


def _read_key_value_file(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split(maxsplit=1)
        values[key] = int(value)
    return values


def read_cgroup_memory_snapshot(cgroup_path: Path) -> dict[str, float]:
    """Read the small set of cgroup-v2 counters used by the diagnostic timeline."""
    memory_stat = _read_key_value_file(cgroup_path / "memory.stat")
    memory_events = _read_key_value_file(cgroup_path / "memory.events")

    missing_stat = set(_REQUIRED_STAT_FIELDS) - memory_stat.keys()
    missing_events = set(_REQUIRED_EVENT_FIELDS) - memory_events.keys()
    if missing_stat or missing_events:
        raise RuntimeError(
            f"Incomplete cgroup memory counters: stat={sorted(missing_stat)}, "
            f"events={sorted(missing_events)}"
        )

    snapshot = {
        f"{_METRIC_PREFIX}.current_bytes": float(
            (cgroup_path / "memory.current").read_text().strip()
        ),
        f"{_METRIC_PREFIX}.anon_bytes": float(memory_stat["anon"]),
        f"{_METRIC_PREFIX}.file_bytes": float(memory_stat["file"]),
        f"{_METRIC_PREFIX}.shmem_bytes": float(memory_stat["shmem"]),
        f"{_METRIC_PREFIX}.events.high": float(memory_events["high"]),
        f"{_METRIC_PREFIX}.events.max": float(memory_events["max"]),
        f"{_METRIC_PREFIX}.events.oom": float(memory_events["oom"]),
        f"{_METRIC_PREFIX}.events.oom_kill": float(memory_events["oom_kill"]),
    }
    peak_path = cgroup_path / "memory.peak"
    if peak_path.is_file():
        snapshot[f"{_METRIC_PREFIX}.peak_bytes"] = float(peak_path.read_text().strip())
    return snapshot


def read_process_memory_snapshot(pid: int, proc_root: Path = Path("/proc")) -> dict[str, float]:
    """Read selected byte counters from a Linux process's ``smaps_rollup`` file."""
    values: dict[str, int] = {}
    for line in (proc_root / str(pid) / "smaps_rollup").read_text().splitlines():
        key, *rest = line.split()
        key = key.removesuffix(":")
        if key not in _PROCESS_STAT_FIELDS:
            continue
        if len(rest) != 2 or rest[1] != "kB":
            raise RuntimeError(f"Unexpected smaps_rollup value for {key}: {line}")
        values[key] = int(rest[0]) * 1024

    missing = set(_PROCESS_STAT_FIELDS) - values.keys()
    if missing:
        raise RuntimeError(f"Incomplete smaps_rollup counters for pid {pid}: {sorted(missing)}")

    return {
        f"{_PROCESS_METRIC_PREFIX}.{metric_name}": float(values[field_name])
        for field_name, metric_name in _PROCESS_STAT_FIELDS.items()
    }


class CgroupMemoryTimeline:
    """Sample job host memory and asynchronously log trainer stage markers."""

    def __init__(
        self,
        log_fn: Callable[[dict[str, float]], None],
        sampling_interval_ms: int = 100,
        cgroup_path: Path | None = None,
        *,
        sample_cgroup: bool = True,
        process_sampling_interval_ms: int | None = None,
        rank: int = 0,
        proc_root: Path = Path("/proc"),
        trainer_process: psutil.Process | None = None,
        watchdog_timeout_seconds: float | None = None,
        watchdog_output_path: Path | None = None,
    ) -> None:
        if sampling_interval_ms <= 0:
            raise ValueError("sampling_interval_ms must be greater than zero")
        if process_sampling_interval_ms is not None and process_sampling_interval_ms <= 0:
            raise ValueError("process_sampling_interval_ms must be greater than zero")
        if watchdog_timeout_seconds is not None and watchdog_timeout_seconds <= 0:
            raise ValueError("watchdog_timeout_seconds must be greater than zero")
        if (watchdog_timeout_seconds is None) != (watchdog_output_path is None):
            raise ValueError(
                "watchdog_timeout_seconds and watchdog_output_path must be configured together"
            )

        self._log_fn = log_fn
        self._sampling_interval_s = sampling_interval_ms / 1_000
        self._process_sampling_interval_s = (
            process_sampling_interval_ms / 1_000
            if process_sampling_interval_ms is not None
            else None
        )
        self._sample_cgroup = sample_cgroup
        self._cgroup_path = cgroup_path or resolve_cgroup_v2_path() if sample_cgroup else None
        self._rank = rank
        self._proc_root = proc_root
        self._trainer_process = trainer_process or psutil.Process()
        self._worker_processes: dict[tuple[int, float], tuple[int, int]] = {}
        self._worker_slot_generations: dict[int, int] = {}
        self._pending_events: queue.SimpleQueue[dict[str, float]] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._watchdog_timeout_ns = (
            None
            if watchdog_timeout_seconds is None
            else int(watchdog_timeout_seconds * 1_000_000_000)
        )
        self._watchdog_output_path = watchdog_output_path
        self._last_progress = (time.monotonic_ns(), "timeline_initialized")
        self._watchdog_dump_progress_ns: int | None = None

        # Validate the source before starting a background thread, so configuration
        # errors fail visibly in the trainer process.
        if self._cgroup_path is not None:
            read_cgroup_memory_snapshot(self._cgroup_path)
        if self._process_sampling_interval_s is not None:
            read_process_memory_snapshot(self._trainer_process.pid, self._proc_root)

    @property
    def cgroup_path(self) -> Path | None:
        return self._cgroup_path

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._last_progress = (time.monotonic_ns(), "timeline_started")
        self._watchdog_dump_progress_ns = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"memory-timeline-rank-{self._rank}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        intervals = [self._sampling_interval_s]
        if self._process_sampling_interval_s is not None:
            intervals.append(self._process_sampling_interval_s)
        self._thread.join(timeout=max(1.0, 2 * max(intervals)))
        self._thread = None

    def record_stage(
        self,
        name: str,
        *,
        monotonic_ns: int | None = None,
        **values: int | float,
    ) -> None:
        """Queue a timestamped marker without performing file I/O in the training path."""
        progress_ns = time.monotonic_ns()
        self._last_progress = (progress_ns, name)
        marker = {
            "diagnostic.timeline.monotonic_ns": float(
                progress_ns if monotonic_ns is None else monotonic_ns
            ),
            "diagnostic.timeline.rank": float(self._rank),
            f"diagnostic.timeline.stage.{name}": 1.0,
        }
        marker.update({f"diagnostic.timeline.{key}": float(value) for key, value in values.items()})
        self._pending_events.put(marker)

    def _run(self) -> None:
        try:
            next_process_sample = time.monotonic()
            while True:
                monotonic_ns = time.monotonic_ns()
                if self._cgroup_path is not None:
                    sample = read_cgroup_memory_snapshot(self._cgroup_path)
                    sample["diagnostic.timeline.monotonic_ns"] = float(monotonic_ns)
                    sample["diagnostic.timeline.rank"] = float(self._rank)
                    sample["diagnostic.timeline.sample"] = 1.0
                    self._log_fn(sample)

                if (
                    self._process_sampling_interval_s is not None
                    and time.monotonic() >= next_process_sample
                ):
                    self._sample_process_memory(monotonic_ns)
                    next_process_sample += self._process_sampling_interval_s
                    while next_process_sample <= time.monotonic():
                        next_process_sample += self._process_sampling_interval_s

                self._drain_pending_events()
                self._maybe_dump_watchdog(monotonic_ns)
                if self._stop_event.is_set():
                    break
                wait_s = self._sampling_interval_s
                if self._cgroup_path is None and self._process_sampling_interval_s is not None:
                    wait_s = max(0.0, next_process_sample - time.monotonic())
                self._stop_event.wait(wait_s)
            self._drain_pending_events()
        except Exception:
            logger.exception("Memory timeline stopped after an unexpected error")

    def _maybe_dump_watchdog(self, monotonic_ns: int) -> None:
        if self._watchdog_timeout_ns is None or self._watchdog_output_path is None:
            return

        progress_ns, last_stage = self._last_progress
        if monotonic_ns - progress_ns < self._watchdog_timeout_ns:
            return
        if self._watchdog_dump_progress_ns == progress_ns:
            return

        stalled_seconds = (monotonic_ns - progress_ns) / 1_000_000_000
        try:
            with self._watchdog_output_path.open("ab", buffering=0) as output:
                output.write(
                    (
                        f"\n=== trainer watchdog rank={self._rank} "
                        f"last_stage={last_stage} stalled_seconds={stalled_seconds:.1f} ===\n"
                    ).encode()
                )
                faulthandler.dump_traceback(file=output, all_threads=True)
        except (OSError, RuntimeError):
            logger.exception("Could not write trainer watchdog stack dump")
        finally:
            self._watchdog_dump_progress_ns = progress_ns

        self._log_fn(
            {
                "diagnostic.timeline.monotonic_ns": float(monotonic_ns),
                "diagnostic.timeline.rank": float(self._rank),
                "diagnostic.timeline.watchdog_dump": 1.0,
                "diagnostic.timeline.watchdog_stalled_seconds": stalled_seconds,
                f"diagnostic.timeline.watchdog_last_stage.{last_stage}": 1.0,
            }
        )

    def _sample_process_memory(self, monotonic_ns: int) -> None:
        try:
            children = self._trainer_process.children(recursive=True)
        except psutil.Error:
            logger.debug("Could not enumerate trainer child processes", exc_info=True)
            children = []

        child_identities: list[tuple[psutil.Process, tuple[int, float]]] = []
        for process in children:
            try:
                child_identities.append((process, (process.pid, process.create_time())))
            except psutil.Error:
                continue

        worker_metadata = self._update_worker_metadata(
            {identity for _process, identity in child_identities}
        )
        processes = [(self._trainer_process, "trainer", -1, 0)] + [
            (process, "worker", *worker_metadata[identity])
            for process, identity in child_identities
        ]

        for process, role, worker_slot, worker_generation in processes:
            try:
                sample = read_process_memory_snapshot(process.pid, self._proc_root)
            except (OSError, RuntimeError, psutil.Error):
                logger.debug("Could not sample process memory for pid %d", process.pid)
                continue
            sample.update(
                {
                    "diagnostic.timeline.monotonic_ns": float(monotonic_ns),
                    "diagnostic.timeline.rank": float(self._rank),
                    f"{_PROCESS_METRIC_PREFIX}.sample": 1.0,
                    f"{_PROCESS_METRIC_PREFIX}.pid": float(process.pid),
                    f"{_PROCESS_METRIC_PREFIX}.role.{role}": 1.0,
                    f"{_PROCESS_METRIC_PREFIX}.worker_slot": float(worker_slot),
                    f"{_PROCESS_METRIC_PREFIX}.worker_generation": float(worker_generation),
                }
            )
            self._log_fn(sample)

    def _update_worker_metadata(
        self, identities: set[tuple[int, float]]
    ) -> dict[tuple[int, float], tuple[int, int]]:
        self._worker_processes = {
            identity: metadata
            for identity, metadata in self._worker_processes.items()
            if identity in identities
        }
        used_slots = {slot for slot, _generation in self._worker_processes.values()}

        for identity in sorted(identities, key=lambda item: (item[1], item[0])):
            if identity in self._worker_processes:
                continue
            worker_slot = 0
            while worker_slot in used_slots:
                worker_slot += 1
            generation = self._worker_slot_generations.get(worker_slot, -1) + 1
            self._worker_slot_generations[worker_slot] = generation
            self._worker_processes[identity] = (worker_slot, generation)
            used_slots.add(worker_slot)

        return self._worker_processes

    def _drain_pending_events(self) -> None:
        while True:
            try:
                marker = self._pending_events.get_nowait()
            except queue.Empty:
                return
            self._log_fn(marker)
