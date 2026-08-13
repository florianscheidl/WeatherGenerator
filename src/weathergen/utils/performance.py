# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for measuring training throughput metrics."""

import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, TextIO

import torch

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


class ThroughputTracker:
    """Tracks training throughput metrics.

    Accumulates per-batch sample and source-byte counts across ranks, with the warmup
    / accumulation logic required to produce stable global throughput metrics.
    """

    def __init__(
        self,
        device: torch.device,
        warmup_steps: int,
        batch_size_per_gpu: int,
    ) -> None:
        self._device = device
        self._warmup_steps = warmup_steps
        self.batch_size_per_gpu = batch_size_per_gpu
        self._t0: float | None = None
        self._warmup_done: bool = False
        self._total_batches: int = 0
        self._total_samples: int = 0
        self._total_mb: float = 0.0
        self._synced_elapsed: float | None = None
        self._synced_global_batches: int = 0
        self._synced_global_samples: int = 0
        self._synced_global_mb: float = 0.0

    def step(
        self,
        batch,
        istep: int,
        log_fn: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        """Record one training step and optionally log metrics.

        Wrapper around ``update`` and ``compute_metrics`` that also computes
        source bytes from the batch on the fly. When metrics are available and
        the current rank is root, ``log_fn`` is called with the metrics dict.

        Args:
            batch: The current training batch (must expose ``get_source_samples()``).
            batch_size_per_gpu: Number of samples processed on this rank.
            istep: Global training step index.
            log_fn: Called with the metrics dict on the root rank once warmup is
                    complete. Typically ``lambda m: logger.log_metrics(stage, m, step=istep)``.
        """
        source_mb = compute_source_bytes(batch.get_source_samples()) / 1e6
        self.update(istep, source_mb)
        self._sync()  # collective: all ranks must participate
        if log_fn is not None and is_root():
            metrics = self.compute_metrics()
            if metrics is not None:
                log_fn(metrics)

    def update(self, istep: int, source_mb: float) -> None:
        """Record one training step, handling warmup internally.

        Args:
            batch_size_per_gpu: Number of samples processed on this rank.
            istep: Global training step index (used for warmup countdown).
            source_mb: Source tensor megabytes for this batch. Should be computed
                       fresh each step via ``compute_source_bytes`` as batch sizes
                       can vary across samples.
        """
        if not self._warmup_done:
            if istep >= self._warmup_steps - 1:
                self._t0 = time.time()
                self._warmup_done = True
        else:
            torch.cuda.synchronize()
            self._total_batches += 1
            self._total_samples += self.batch_size_per_gpu
            self._total_mb += source_mb

    def _sync(self) -> None:
        """Collective: reduce per-rank counters across all ranks and cache the result.

        Must be called on every rank at the same point in the training loop.
        The cached values are later read by ``compute_metrics()`` on the root rank.
        """
        if self._total_batches == 0 or self._t0 is None:
            return

        elapsed = time.time() - self._t0

        global_batches = torch.tensor(self._total_batches, dtype=torch.int64, device=self._device)
        global_samples = torch.tensor(self._total_samples, dtype=torch.int64, device=self._device)
        global_total_mb = torch.tensor(self._total_mb, dtype=torch.float32, device=self._device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            elapsed_tensor = torch.tensor(elapsed, dtype=torch.float32, device=self._device)
            torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.AVG)
            elapsed = elapsed_tensor.item()

            torch.distributed.all_reduce(global_batches)
            torch.distributed.all_reduce(global_samples)
            torch.distributed.all_reduce(global_total_mb)

        self._synced_elapsed = elapsed
        self._synced_global_batches = int(global_batches.item())
        self._synced_global_samples = int(global_samples.item())
        self._synced_global_mb = global_total_mb.item()

    def compute_metrics(self) -> dict[str, float] | None:
        """Return performance metrics dict, or None if warmup is not yet complete.

        Returns:
            Dict of ``"performance.<key>": value`` pairs, or None if no data yet.
        """
        if self._total_batches == 0 or self._t0 is None:
            return None
        elapsed = time.time() - self._t0

        if elapsed <= 0 or self._synced_elapsed is None or self._synced_elapsed <= 0:
            return None

        metrics: dict[str, float] = {}

        # Device-level throughput (this rank only).
        metrics["performance.throughput.device.batches_per_sec"] = self._total_batches / elapsed
        metrics["performance.throughput.device.samples_per_sec"] = self._total_samples / elapsed
        metrics["performance.throughput.device.mb_per_sec"] = self._total_mb / elapsed

        # Global throughput: use values already reduced across all ranks by _sync().
        synced_elapsed = self._synced_elapsed
        metrics["performance.throughput.global.batches_per_sec"] = (
            self._synced_global_batches / synced_elapsed
        )
        metrics["performance.throughput.global.samples_per_sec"] = (
            self._synced_global_samples / synced_elapsed
        )
        metrics["performance.throughput.global.mb_per_sec"] = (
            self._synced_global_mb / synced_elapsed
        )

        return metrics


class NullThroughputTracker:
    """No-op throughput tracker used when performance tracking is disabled.

    Implements the same interface as ``ThroughputTracker`` so call sites in the
    training loop need no ``if`` guards.
    """

    def step(self, batch, istep: int, log_fn=None) -> None:
        pass


def compute_source_bytes(source_samples) -> int:
    """Count total bytes of all source token tensors in a batch.

    Args:
        source_samples: Result of sample_batch.get_source_samples(), containing
                        a list of samples each with per-stream source token cells.

    Returns:
        Total byte count across all streams and cells in the batch.
    """
    total = 0
    for sample in source_samples.samples:
        for stream_data in sample.streams_data.values():
            for t in stream_data.source_tokens_cells:
                total += t.nbytes
    return total


@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class NvtxAnnotator:
    """Emit nested NVTX ranges under a shared name prefix, or nothing when disabled.

    Unlike ``InferencePhaseProfiler`` this keeps no records: it is meant for code that also
    runs in DataLoader worker processes, where no run directory is available to write to and
    the trace is the only sink. An annotator is created before the workers start and is copied
    into each of them together with the dataset, so one flag covers parent and workers.
    """

    def __init__(self, enabled: bool, prefix: str = "") -> None:
        self.enabled = enabled
        self.prefix = prefix

    def child(self, prefix: str) -> "NvtxAnnotator":
        """Return an annotator that prepends ``prefix`` to this one's range names."""
        return NvtxAnnotator(self.enabled, f"{self.prefix}{prefix}")

    def range(self, name: str) -> AbstractContextManager[None]:
        """Context manager for one NVTX range; a no-op context when annotation is off."""
        if not self.enabled:
            return nullcontext()
        return nvtx_range(f"{self.prefix}{name}")


class _LoaderPhaseWriter:
    """Process-local JSONL writer shared by child profiler prefixes."""

    def __init__(
        self,
        enabled: bool,
        output_dir: Path | None,
        rank: int,
        run_id: str,
        stage: str,
    ) -> None:
        if enabled and output_dir is None:
            raise ValueError("loader phase timing requires an output directory")
        self.enabled = enabled
        self.output_dir = output_dir
        self.rank = rank
        self.run_id = run_id
        self.stage = stage
        self.stack: list[dict[str, Any]] = []
        self._file: TextIO | None = None
        self._pid: int | None = None
        self._write_failed = False

    def _worker_id(self) -> int | None:
        worker_info = torch.utils.data.get_worker_info()
        return None if worker_info is None else worker_info.id

    def _get_file(self) -> tuple[TextIO, int | None, int]:
        pid = os.getpid()
        worker_id = self._worker_id()
        if self._file is not None and self._pid != pid:
            # Do not retain a file object inherited from a process that happened to write
            # before the DataLoader forked. Normal multi-worker use opens only after fork.
            self._file.close()
            self._file = None
            self.stack = []

        if self._file is None:
            assert self.output_dir is not None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            worker_label = "main" if worker_id is None else f"{worker_id:02d}"
            output_path = self.output_dir / (
                f"loader_phase_timing_rank{self.rank:04d}_worker{worker_label}.jsonl"
            )
            self._file = output_path.open("a", encoding="utf-8", buffering=1)
            self._pid = pid

        return self._file, worker_id, pid

    def write(
        self,
        name: str,
        metadata: dict[str, Any],
        context: list[dict[str, Any]],
        start_ns: int,
        end_ns: int,
    ) -> None:
        if not self.enabled or self._write_failed:
            return

        try:
            output, worker_id, pid = self._get_file()
            record = {
                "schema_version": 1,
                "clock": "time.perf_counter_ns",
                "rank": self.rank,
                "worker": worker_id,
                "pid": pid,
                "run_id": self.run_id,
                "stage": self.stage,
                "name": name,
                "context": context,
                "metadata": metadata,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
        except (OSError, TypeError, ValueError):
            # Diagnostics must never take down a loader worker. Disable subsequent writes
            # after the first failure so a broken filesystem does not flood the logs.
            logger.exception("Disabling loader phase timing after a write failure")
            self._write_failed = True


class LoaderPhaseProfiler:
    """Record nested DataLoader phases without relying on tracing forked workers.

    The object is constructed before DataLoader workers fork. It holds no open file until a
    worker completes its first range, then writes to a rank/worker-specific JSONL file. Child
    profilers share a process-local context stack so flat records retain their enclosing batch
    and retry ranges. Optional NVTX emission preserves the existing timeline annotations.
    """

    def __init__(
        self,
        timing_enabled: bool = False,
        nvtx_enabled: bool = False,
        output_dir: Path | None = None,
        rank: int = 0,
        run_id: str = "",
        stage: str = "",
        prefix: str = "",
        writer: _LoaderPhaseWriter | None = None,
    ) -> None:
        self.timing_enabled = timing_enabled
        self.nvtx_enabled = nvtx_enabled
        self.prefix = prefix
        self._writer = writer or _LoaderPhaseWriter(timing_enabled, output_dir, rank, run_id, stage)

    def child(self, prefix: str) -> "LoaderPhaseProfiler":
        """Return a profiler with an additional name prefix and shared writer state."""
        return LoaderPhaseProfiler(
            timing_enabled=self.timing_enabled,
            nvtx_enabled=self.nvtx_enabled,
            prefix=f"{self.prefix}{prefix}",
            writer=self._writer,
        )

    def range(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> AbstractContextManager[None]:
        """Record one range and its active parent context, including on exceptions."""
        if not self.timing_enabled and not self.nvtx_enabled:
            return nullcontext()
        return self._range(f"{self.prefix}{name}", metadata or {})

    @contextmanager
    def _range(self, name: str, metadata: dict[str, Any]):
        if self.nvtx_enabled:
            torch.cuda.nvtx.range_push(name)

        start_ns = time.perf_counter_ns()
        context = [dict(item) for item in self._writer.stack]
        self._writer.stack.append({"name": name, "metadata": metadata})
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            self._writer.stack.pop()
            if self.nvtx_enabled:
                torch.cuda.nvtx.range_pop()
            self._writer.write(name, metadata, context, start_ns, end_ns)


class InferencePhaseProfiler:
    """Record inference phase timings and emit matching NVTX ranges."""

    def __init__(
        self,
        output_path: Path | None,
        rank: int,
        world_size: int,
        run_id: str,
        nvtx_annotate: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_path = output_path
        self.nvtx_annotate = nvtx_annotate
        self.started_ns = time.perf_counter_ns()
        self.payload: dict[str, Any] = {
            "schema_version": 2,
            "clock": "time.perf_counter_ns",
            "started_ns": self.started_ns,
            "rank": rank,
            "world_size": world_size,
            "run_id": run_id,
            "metadata": metadata or {},
            "phases": [],
        }

    @contextmanager
    def phase(self, name: str, batch_idx: int | None = None):
        """Measure one host phase without introducing a CUDA synchronization."""
        range_name = f"inference.{name}"
        if batch_idx is not None:
            range_name = f"{range_name}.batch_{batch_idx}"

        if self.nvtx_annotate:
            torch.cuda.nvtx.range_push(range_name)
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            if self.nvtx_annotate:
                torch.cuda.nvtx.range_pop()
            if self.output_path is not None:
                self.payload["phases"].append(
                    {
                        "name": name,
                        "batch_idx": batch_idx,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": end_ns - start_ns,
                        "start_s": (start_ns - self.started_ns) / 1e9,
                        "duration_s": (end_ns - start_ns) / 1e9,
                    }
                )
                self.write()

    def write(self) -> None:
        """Atomically write this rank's accumulated timing records."""
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
        temporary_path.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.output_path)


def _read_key_value_file(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        values[key] = int(value)
    return values


def _read_process_io(path: Path = Path("/proc/self/io")) -> dict[str, int]:
    """Read Linux per-process I/O accounting counters."""
    return _read_key_value_file(path)


def _resolve_cgroup_v2_path(
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_mount: Path = Path("/sys/fs/cgroup"),
) -> Path:
    """Resolve the cgroup-v2 directory visible to the current process."""
    for line in proc_cgroup_path.read_text(encoding="utf-8").splitlines():
        hierarchy_id, controllers, relative_path = line.split(":", maxsplit=2)
        if hierarchy_id == "0" and controllers == "":
            path = cgroup_mount / relative_path.lstrip("/")
            if (path / "memory.current").is_file():
                return path
            if relative_path == "/" and (cgroup_mount / "memory.current").is_file():
                return cgroup_mount
            raise RuntimeError(f"Cannot read cgroup-v2 memory metrics from {path}")
    raise RuntimeError("The current process does not expose a cgroup-v2 hierarchy")


def _read_cgroup_memory(path: Path) -> dict[str, int]:
    memory_stat = _read_key_value_file(path / "memory.stat")
    fields = ("anon", "file", "shmem", "file_dirty", "file_writeback")
    values = {field: memory_stat[field] for field in fields if field in memory_stat}
    values["current"] = int((path / "memory.current").read_text(encoding="utf-8").strip())
    peak_path = path / "memory.peak"
    if peak_path.is_file():
        values["peak"] = int(peak_path.read_text(encoding="utf-8").strip())
    return values


def _counter_delta(start: dict[str, int], end: dict[str, int]) -> dict[str, int]:
    return {key: end[key] - start[key] for key in start.keys() & end.keys()}


class OutputWriterProfiler:
    """Incrementally record nested inference-output writer operations.

    Each completed range is flushed as one JSONL record. Process CPU and Linux process-I/O
    counters are captured for every range; cgroup memory snapshots are limited to explicitly
    requested coarse ranges to avoid perturbing each small metadata operation.
    """

    def __init__(
        self,
        enabled: bool,
        output_path: Path | None,
        rank: int,
        world_size: int,
        run_id: str,
        nvtx_annotate: bool,
        context: dict[str, Any] | None = None,
        *,
        cgroup_memory: bool = True,
        _root: "OutputWriterProfiler | None" = None,
    ) -> None:
        if enabled and output_path is None:
            raise ValueError("output writer timing requires an output path")
        self.enabled = enabled
        self.output_path = output_path
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.nvtx_annotate = nvtx_annotate
        self.context = dict(context or {})
        self._root = _root or self
        if _root is None:
            self._file: TextIO | None = None
            self._write_failed = False
            self._cgroup_path: Path | None = None
            if enabled and cgroup_memory:
                try:
                    self._cgroup_path = _resolve_cgroup_v2_path()
                except (OSError, RuntimeError, ValueError):
                    logger.warning("Cgroup-v2 writer snapshots are unavailable", exc_info=True)

    def child(self, **context: Any) -> "OutputWriterProfiler":
        """Return a profiler carrying additional immutable record context."""
        return OutputWriterProfiler(
            enabled=self.enabled,
            output_path=self.output_path,
            rank=self.rank,
            world_size=self.world_size,
            run_id=self.run_id,
            nvtx_annotate=self.nvtx_annotate,
            context={**self.context, **context},
            cgroup_memory=False,
            _root=self._root,
        )

    @contextmanager
    def range(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        *,
        capture_cgroup: bool = False,
    ):
        """Measure one writer operation without synchronizing CUDA."""
        if not self.enabled:
            yield
            return

        range_name = f"output_writer.{name}"
        if self.nvtx_annotate:
            torch.cuda.nvtx.range_push(range_name)
        record_metadata = metadata if metadata is not None else {}
        start_ns = time.perf_counter_ns()
        process_cpu_start_ns = time.process_time_ns()
        process_io_start = self._optional_process_io()
        cgroup_start = self._optional_cgroup_memory() if capture_cgroup else None
        completed = False
        try:
            yield
            completed = True
        finally:
            end_ns = time.perf_counter_ns()
            process_cpu_end_ns = time.process_time_ns()
            process_io_end = self._optional_process_io()
            cgroup_end = self._optional_cgroup_memory() if capture_cgroup else None
            if self.nvtx_annotate:
                torch.cuda.nvtx.range_pop()
            self._write(
                name=name,
                metadata=record_metadata,
                start_ns=start_ns,
                end_ns=end_ns,
                process_cpu_start_ns=process_cpu_start_ns,
                process_cpu_end_ns=process_cpu_end_ns,
                process_io_start=process_io_start,
                process_io_end=process_io_end,
                cgroup_start=cgroup_start,
                cgroup_end=cgroup_end,
                completed=completed,
            )

    def close(self) -> None:
        """Close the incremental JSONL file if this profiler opened it."""
        root = self._root
        if root._file is not None:
            try:
                root._file.close()
            except OSError:
                logger.exception("Failed to close output writer timing file")
            root._file = None

    def _optional_process_io(self) -> dict[str, int] | None:
        try:
            return _read_process_io()
        except (OSError, RuntimeError, ValueError):
            return None

    def _optional_cgroup_memory(self) -> dict[str, int] | None:
        cgroup_path = self._root._cgroup_path
        if cgroup_path is None:
            return None
        try:
            return _read_cgroup_memory(cgroup_path)
        except (OSError, RuntimeError, ValueError):
            return None

    def _get_file(self) -> TextIO:
        root = self._root
        if root._file is None:
            assert self.output_path is not None
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            root._file = self.output_path.open("a", encoding="utf-8", buffering=1)
        return root._file

    def _write(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
        start_ns: int,
        end_ns: int,
        process_cpu_start_ns: int,
        process_cpu_end_ns: int,
        process_io_start: dict[str, int] | None,
        process_io_end: dict[str, int] | None,
        cgroup_start: dict[str, int] | None,
        cgroup_end: dict[str, int] | None,
        completed: bool,
    ) -> None:
        root = self._root
        if root._write_failed:
            return
        record: dict[str, Any] = {
            "schema_version": 1,
            "clock": "time.perf_counter_ns",
            "rank": self.rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "run_id": self.run_id,
            "name": name,
            "completed": completed,
            "context": self.context,
            "metadata": metadata,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": end_ns - start_ns,
            "process_cpu_start_ns": process_cpu_start_ns,
            "process_cpu_end_ns": process_cpu_end_ns,
            "process_cpu_duration_ns": process_cpu_end_ns - process_cpu_start_ns,
        }
        if process_io_start is not None and process_io_end is not None:
            record["process_io_start"] = process_io_start
            record["process_io_end"] = process_io_end
            record["process_io_delta"] = _counter_delta(process_io_start, process_io_end)
        if cgroup_start is not None and cgroup_end is not None:
            record["cgroup_memory_start"] = cgroup_start
            record["cgroup_memory_end"] = cgroup_end
        try:
            output = self._get_file()
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
        except (OSError, TypeError, ValueError):
            logger.exception("Disabling output writer timing after a write failure")
            root._write_failed = True


def _nvtx_push(name: str):
    torch.cuda.nvtx.range_push(name)


def _nvtx_pop():
    torch.cuda.nvtx.range_pop()


def register_nvtx_hooks(model, scope: str = "global"):
    torch.nn.modules.module.register_module_forward_pre_hook(
        lambda m, args: _nvtx_push(f"{m.__class__.__name__}.forward")
    )
    torch.nn.modules.module.register_module_forward_hook(
        lambda m, input, output: _nvtx_pop(), always_call=True
    )
    torch.nn.modules.module.register_module_full_backward_pre_hook(
        lambda m, args: _nvtx_push(f"{m.__class__.__name__}.backward")
    )
    torch.nn.modules.module.register_module_full_backward_hook(lambda m, input, output: _nvtx_pop())
