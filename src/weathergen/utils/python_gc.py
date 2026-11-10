# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Python garbage collector (GC) in training runs: make its pauses visible, and control when they
happen.

CPython's cycle collector runs whenever enough container objects were allocated, on whichever
thread allocates. A gen-2 (full) collection walks every tracked object of the process; with a
large heap it stops the training step for hundreds of milliseconds, at a random point, on one
rank at a time, and the other ranks then wait for it in the next collective.

Config section `python_gc` (absent from `config/default_config.yml`; the defaults below leave
CPython's behaviour unchanged):

    python_gc:
      nvtx_ranges: False      # nvtx range `gc:gen<n>` around every collection, for nsys
      log_stats: False        # log collections, time and longest pause per generation per step
      freeze: False           # gc.collect() and gc.freeze() once, before the first step
      collect_every: 0        # N > 0: no automatic gen-2 collections, a full one every N steps
      break_nvtx_cycles: False  # profiled runs: drop the per-step nvtx wrappers after the step

- `nvtx_ranges` needs CUDA (on other devices it is skipped with a warning). The callback is
  registered before the data loader starts, so forked DataLoader workers inherit it; their
  ranges show up on the worker rows when nsys traces forks (`--nsys-trace-fork`).
- `freeze` moves everything alive before the first step (model, data sampler, imported
  modules) into the permanent generation, which collections skip. Frozen objects are never
  collected, even if they become cyclic garbage later.
- `collect_every` raises the gen-2 threshold out of reach (gen 0 and gen 1 still run
  automatically) and runs `gc.collect()` after every N-th training step, on every rank at the
  same point of the step: in a profiled run inside the step's `batch_<i>` range, after the
  optimizer step and logging. Cyclic garbage that reached gen 2 lives until then. Only CPython
  3.12's generational collector is supported (`requires-python` is `<3.13`).
- `break_nvtx_cycles` (`ProfilingTrainer`, `profiling.nvtx_annotate`): the nvtx wrappers that
  the profiling trainer puts on each batch (`to_device`, `pin_memory`) and on each scaled loss
  (`backward`) form reference cycles (instance -> wrapper -> bound method -> instance), so the
  batch, its device tensors and the loss with its autograd graph are freed only by a gen-2
  collection. This option removes the wrappers after use, so refcounting frees them.

None of this changes the arithmetic of a training step: a collection only frees unreachable
objects.
"""

import dataclasses
import gc
import logging
import time
from collections.abc import Iterator
from typing import Any, TypeVar

import torch

from weathergen.common.config import Config
from weathergen.utils.performance import _nvtx_pop, _nvtx_push

logger = logging.getLogger(__name__)

T = TypeVar("T")

NUM_GENERATIONS = 3
# gen-2 collections run when gen 1 has been collected this many times; set_threshold takes ints
_UNREACHABLE_THRESHOLD = 2**31 - 1


@dataclasses.dataclass(frozen=True)
class GcConfig:
    """The `python_gc` section of a run config, see the module docstring."""

    nvtx_ranges: bool = False
    log_stats: bool = False
    freeze: bool = False
    collect_every: int = 0
    break_nvtx_cycles: bool = False

    @classmethod
    def from_config(cls, cf: Config) -> "GcConfig":
        section = cf.get("python_gc") or {}
        defaults = cls()
        return cls(
            **{
                f.name: section.get(f.name, getattr(defaults, f.name))
                for f in dataclasses.fields(cls)
            }
        )


class GcStats:
    """Collections, time and collected objects per generation, from a `gc.callbacks` entry."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.reset()

    def reset(self) -> None:
        self.count = [0] * NUM_GENERATIONS
        self.seconds = [0.0] * NUM_GENERATIONS
        self.longest = [0.0] * NUM_GENERATIONS
        self.collected = [0] * NUM_GENERATIONS

    def callback(self, phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            self._start = time.perf_counter()
            return
        if self._start is None:
            return
        elapsed = time.perf_counter() - self._start
        self._start = None
        gen = info["generation"]
        self.count[gen] += 1
        self.seconds[gen] += elapsed
        self.longest[gen] = max(self.longest[gen], elapsed)
        self.collected[gen] += info["collected"]

    def format(self) -> str:
        return ", ".join(
            f"gen{g} {self.count[g]}x {1e3 * self.seconds[g]:.1f} ms "
            f"(max {1e3 * self.longest[g]:.1f} ms, {self.collected[g]} freed)"
            for g in range(NUM_GENERATIONS)
        )


def _nvtx_callback(phase: str, info: dict[str, Any]) -> None:
    """Open an nvtx range `gc:gen<n>` for the duration of each collection."""
    if phase == "start":
        _nvtx_push(f"gc:gen{info['generation']}")
    else:
        _nvtx_pop()


class GcControl:
    """
    Applies a `GcConfig` to the training loop: `install` at setup (before the data loader
    forks its workers), `before_training` at the start of `Trainer.train`, and `steps` around
    the loop over the training batches.
    """

    def __init__(self, cfg: GcConfig, rank: int = 0) -> None:
        self.cfg = cfg
        self.rank = rank
        self.stats: GcStats | None = None
        self.installed = False
        self.frozen = False
        self.steps_done = 0

    @classmethod
    def from_config(cls, cf: Config) -> "GcControl":
        return cls(GcConfig.from_config(cf), rank=cf.get("rank", 0) or 0)

    def install(self) -> None:
        """Register the gc callbacks (once per process)."""
        if self.installed:
            return
        self.installed = True
        cfg = self.cfg
        if cfg != GcConfig():
            logger.info(f"Python GC: {cfg}, thresholds {gc.get_threshold()}")
        if cfg.nvtx_ranges:
            if torch.cuda.is_available():
                gc.callbacks.append(_nvtx_callback)
            else:
                logger.warning("python_gc.nvtx_ranges needs CUDA (nvtx), skipped.")
        if cfg.log_stats:
            self.stats = GcStats()
            gc.callbacks.append(self.stats.callback)

    def before_training(self) -> None:
        """Freeze the heap and take gen-2 collections off automatic, once per run."""
        if self.frozen:
            return
        self.frozen = True
        cfg = self.cfg
        if cfg.freeze:
            start = time.perf_counter()
            gc.collect()
            gc.freeze()
            logger.info(
                f"Python GC rank {self.rank}: froze {gc.get_freeze_count()} objects "
                f"in {1e3 * (time.perf_counter() - start):.0f} ms"
            )
        if cfg.collect_every > 0:
            threshold0, threshold1, _ = gc.get_threshold()
            gc.set_threshold(threshold0, threshold1, _UNREACHABLE_THRESHOLD)
            logger.info(
                f"Python GC rank {self.rank}: automatic gen-2 collections off, thresholds "
                f"{gc.get_threshold()}; gc.collect() every {cfg.collect_every} steps"
            )
        if self.stats is not None:
            self.stats.reset()

    def after_step(self, bidx: int) -> None:
        """The fixed point at the end of a training step: manual collection, then stats."""
        self.steps_done += 1
        manual = ""
        if self.cfg.collect_every > 0 and self.steps_done % self.cfg.collect_every == 0:
            start = time.perf_counter()
            collected = gc.collect()
            manual = f"; manual gc.collect() {1e3 * (time.perf_counter() - start):.1f} ms, "
            manual += f"{collected} freed"
        if self.stats is not None:
            logger.info(f"Python GC rank {self.rank} batch {bidx}: {self.stats.format()}{manual}")
            self.stats.reset()

    def steps(self, batches: Iterator[tuple[int, T]]) -> Iterator[tuple[int, T]]:
        """Yield the training batches, calling `after_step` once each step is done."""
        if not (self.cfg.collect_every > 0 or self.stats is not None):
            yield from batches
            return
        for bidx, batch in batches:
            yield bidx, batch
            self.after_step(bidx)
