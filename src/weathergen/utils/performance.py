# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for measuring training throughput metrics."""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager

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

    def step(self, batch, istep: int) -> None:
        """Accumulate one training step's counts. No synchronization or collectives.

        Call on every step from the training loop. Metrics are emitted separately
        via ``log`` at the logging interval, so the hot path stays free of device
        syncs and cross-rank collectives.

        Args:
            batch: The current training batch (must expose ``get_source_samples()``).
            istep: Global training step index (used for the warmup countdown).
        """
        source_mb = compute_source_bytes(batch.get_source_samples()) / 1e6
        self.update(istep, source_mb)

    def update(self, istep: int, source_mb: float) -> None:
        """Record one training step, handling warmup internally.

        Purely local bookkeeping: no device synchronization. The cumulative
        counts are turned into throughput (and reduced across ranks) only when
        ``compute_metrics`` runs.

        Args:
            istep: Global training step index (used for warmup countdown).
            source_mb: Source tensor megabytes for this batch. Should be computed
                       fresh each step via ``compute_source_bytes`` as batch sizes
                       can vary across samples.
        """
        self._total_batches += 1
        self._total_samples += self.batch_size_per_gpu
        self._total_mb += source_mb

    def log(self, log_fn: Callable[[dict[str, float]], None] | None = None) -> None:
        """Collective: reduce throughput across ranks and log it on the root rank.

        Must be called on every rank at the same point in the training loop (the
        all-reduce lives in ``compute_metrics``). Intended to be called once per
        logging interval rather than every step, so the only cross-rank
        synchronization for throughput happens here.

        Args:
            log_fn: Called with the metrics dict on the root rank once warmup is
                    complete. Typically ``lambda m: logger.log_metrics(stage, m, step=istep)``.
        """
        metrics = self.compute_metrics()
        if metrics is not None and log_fn is not None and is_root():
            log_fn(metrics)

    def compute_metrics(self) -> dict[str, float] | None:
        """Return throughput metrics dict, or None if warmup is not yet complete.

        Collective: performs a single SUM all-reduce of the per-device throughput
        to obtain the global throughput, so it must be called on every rank at the
        same point in the training loop. The returned dict is identical on all
        ranks. Global throughput is the sum of the per-device rates across ranks.

        Returns:
            Dict of ``"performance.<key>": value`` pairs, or None if no data yet.
        """
        if self._total_batches == 0 or self._t0 is None:
            return None


        device_batches = self._total_batches
        device_samples = self._total_samples
        device_mb = self._total_mb

        # Global throughput: sum the per-device rates across ranks with a single
        # reduce, done here (once per logging interval) rather than per step.
        global_batches, global_samples, global_mb = device_batches, device_samples, device_mb
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rates = torch.tensor(
                [device_batches, device_samples, device_mb],
                dtype=torch.float64,
                device=self._device,
            )
            torch.distributed.all_reduce(rates, op=torch.distributed.ReduceOp.SUM)
            global_batches, global_samples, global_mb = rates.tolist()

        return {
            "performance.throughput.global.batches": global_batches,
            "performance.throughput.global.samples": global_samples,
            "performance.throughput.global.mb": global_mb,
        }


class NullThroughputTracker:
    """No-op throughput tracker used when performance tracking is disabled.

    Implements the same interface as ``ThroughputTracker`` so call sites in the
    training loop need no ``if`` guards.
    """

    def step(self, batch, istep: int) -> None:
        pass

    def log(self, log_fn=None) -> None:
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
