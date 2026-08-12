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
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

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
            "schema_version": 1,
            "clock": "time.perf_counter_ns",
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
                        "start_s": (start_ns - self.started_ns) / 1e9,
                        "duration_s": (end_ns - start_ns) / 1e9,
                    }
                )

    def write(self) -> None:
        """Atomically write this rank's accumulated timing records."""
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
        temporary_path.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.output_path)


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
