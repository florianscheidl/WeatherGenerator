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
      debug_cycles_steps: []  # batch indices: log what the step-end collection frees (slow)

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

- `debug_cycles_steps` (a list of batch indices, e.g. `[3, 4]`; a single int is one step):
  at the end of these steps, the full collection runs with `gc.DEBUG_SAVEALL`, and the log
  gets one block per rank and step on what it found unreachable: object count by type,
  sample objects and the types that refer to them within the garbage, a few reference cycles
  as type chains, and a census of the unfrozen heap (top types, change since the previous
  listed step). Every line of the block starts with `[gc-debug r<rank> b<batch>]`. Analysing
  ~10,000 objects and the heap takes tens to hundreds of ms, so the listed steps (and the next
  one, which starts later) are not timing-clean. This replaces the step's `collect_every`
  collection. The saved garbage is released and collected again before the next step.

None of this changes the arithmetic of a training step: a collection only frees unreachable
objects.
"""

import collections
import dataclasses
import gc
import itertools
import logging
import time
import types
import weakref
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
    debug_cycles_steps: tuple[int, ...] = ()

    @classmethod
    def from_config(cls, cf: Config) -> "GcConfig":
        section = cf.get("python_gc") or {}
        defaults = cls()
        values = {
            f.name: section.get(f.name, getattr(defaults, f.name)) for f in dataclasses.fields(cls)
        }
        steps = values["debug_cycles_steps"]
        steps = [steps] if isinstance(steps, int) else (steps or [])
        values["debug_cycles_steps"] = tuple(int(step) for step in steps)
        return cls(**values)


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
        # type name -> count in the unfrozen heap at the previous `debug_cycles_steps` step
        self.last_census: collections.Counter[str] | None = None

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
        if bidx in self.cfg.debug_cycles_steps:
            manual = self.debug_cycles(bidx)
        elif self.cfg.collect_every > 0 and self.steps_done % self.cfg.collect_every == 0:
            start = time.perf_counter()
            collected = gc.collect()
            manual = f"; manual gc.collect() {1e3 * (time.perf_counter() - start):.1f} ms, "
            manual += f"{collected} freed"
        if self.stats is not None:
            logger.info(f"Python GC rank {self.rank} batch {bidx}: {self.stats.format()}{manual}")
            self.stats.reset()

    def steps(self, batches: Iterator[tuple[int, T]]) -> Iterator[tuple[int, T]]:
        """Yield the training batches, calling `after_step` once each step is done."""
        cfg = self.cfg
        if not (cfg.collect_every > 0 or self.stats is not None or cfg.debug_cycles_steps):
            yield from batches
            return
        for bidx, batch in batches:
            yield bidx, batch
            self.after_step(bidx)

    def debug_cycles(self, bidx: int) -> str:
        """
        Full collection with `gc.DEBUG_SAVEALL`, then log what it found unreachable and what
        the unfrozen heap holds. Returns a summary for the stats line.
        """
        before = len(gc.garbage)
        start = time.perf_counter()
        gc.set_debug(gc.DEBUG_SAVEALL)
        try:
            collected = gc.collect()
        finally:
            gc.set_debug(0)
        collect_ms = 1e3 * (time.perf_counter() - start)
        # take this collection's objects out of gc.garbage; the list keeps them alive meanwhile
        garbage = gc.garbage[before:]
        del gc.garbage[before:]

        start = time.perf_counter()
        prefix = f"[gc-debug r{self.rank} b{bidx}]"
        lines = [
            f"{prefix} gc.collect() with DEBUG_SAVEALL: {collect_ms:.1f} ms, {collected} "
            f"unreachable, {len(garbage)} in gc.garbage"
        ]
        lines += garbage_report(garbage)
        del garbage
        # the saved objects are cyclic garbage again: free them now, not at a later step
        start_free = time.perf_counter()
        freed = gc.collect()
        free_ms = 1e3 * (time.perf_counter() - start_free)
        lines.append(f"freed the saved garbage: gc.collect() {free_ms:.1f} ms, {freed} freed")
        by_type = collections.Counter(map(type, gc.get_objects()))
        census: collections.Counter[str] = collections.Counter()
        for t, n in by_type.items():
            census[type_name(t)] += n
        del by_type
        lines += heap_report(census, self.last_census)
        self.last_census = census
        lines.append(f"analysis took {1e3 * (time.perf_counter() - start):.0f} ms")
        logger.info(f"\n{prefix} ".join(lines))
        return f"; debug gc.collect() {collect_ms:.1f} ms, {collected} freed (DEBUG_SAVEALL)"


# What the `debug_cycles_steps` block lists
_TOP_TYPES = 40
_SAMPLE_TYPES = 8
_SAMPLES_PER_TYPE = 3
_REFERRER_TYPES = 6
_CYCLE_TYPES = 4
_CYCLE_DEPTH = 8
_CYCLE_NODES = 20_000
_HEAP_TYPES = 15
_REPR_CHARS = 120


def type_name(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


def _truncate(text: str, limit: int = _REPR_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe(obj: Any) -> str:
    """
    A short description of `obj` that never calls an arbitrary `__repr__` (a tensor's repr
    copies its data to the host, a container's repr calls its items' reprs).
    """
    try:
        if isinstance(obj, torch.Tensor):
            grad_fn = obj.grad_fn
            return (
                f"{type(obj).__name__}(shape={tuple(obj.shape)}, dtype={obj.dtype}, "
                f"device={obj.device}, requires_grad={obj.requires_grad}, "
                f"grad_fn={type(grad_fn).__name__ if grad_fn is not None else None})"
            )
        if isinstance(obj, torch.autograd.graph.Node):
            return f"autograd node {obj.name()}"
        if isinstance(obj, str | bytes | int | float | complex | None):
            return _truncate(repr(obj))
        if isinstance(obj, dict):
            keys = [_truncate(repr(k), 40) for k in itertools.islice(obj, 6)]
            more = ", ..." if len(obj) > 6 else ""
            return f"dict(len={len(obj)}, keys=[{', '.join(keys)}{more}])"
        if isinstance(obj, list | tuple | set | frozenset | collections.deque):
            items = [type(i).__name__ for i in itertools.islice(obj, 4)]
            more = ", ..." if len(obj) > 4 else ""
            return f"{type(obj).__name__}(len={len(obj)}, items=[{', '.join(items)}{more}])"
        if isinstance(obj, types.FunctionType):
            return f"function {obj.__module__}.{obj.__qualname__}"
        if isinstance(obj, types.MethodType):
            return f"method {type(obj.__self__).__name__}.{obj.__func__.__qualname__}"
        if isinstance(obj, types.BuiltinMethodType):
            owner = type(obj.__self__).__name__ if obj.__self__ is not None else ""
            return f"builtin {owner}.{obj.__name__}"
        if isinstance(obj, types.CodeType):
            return f"code {obj.co_qualname} {obj.co_filename}:{obj.co_firstlineno}"
        if isinstance(obj, types.FrameType):
            code = obj.f_code
            return f"frame {code.co_qualname} {code.co_filename}:{obj.f_lineno}"
        if isinstance(obj, types.CellType):
            try:
                return f"cell -> {type_name(type(obj.cell_contents))}"
            except ValueError:
                return "cell (empty)"
        if isinstance(obj, weakref.ReferenceType):
            target = obj()
            return f"weakref -> {type_name(type(target)) if target is not None else 'dead'}"
        if isinstance(obj, type):
            return f"class {type_name(obj)}"
        if isinstance(obj, types.ModuleType):
            return f"module {obj.__name__}"
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict) and attrs:
            names = [str(k) for k in itertools.islice(attrs, 8)]
            more = ", ..." if len(attrs) > 8 else ""
            return f"<{type_name(type(obj))} attrs {', '.join(names)}{more}>"
    except Exception as e:  # a description must never fail the step
        return f"<{type_name(type(obj))}: describe failed: {type(e).__name__}>"
    return f"<{type_name(type(obj))}>"


def _step(obj: Any, nxt: Any) -> str:
    """The edge obj -> nxt in a cycle path: the key of a dict, the attribute of an object."""
    try:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if value is nxt:
                    return f"[{_truncate(repr(key), 40)}]" if isinstance(key, str | int) else "[.]"
            return ""
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict) and attrs is not nxt:
            for key, value in attrs.items():
                if value is nxt:
                    return f".{key}"
    except Exception:  # a label must never fail the step
        pass
    return ""


def _node(obj: Any) -> str:
    """A node of a cycle path: the type, plus the name for functions, methods, frames, code."""
    name = type_name(type(obj))
    if isinstance(obj, types.FunctionType):
        return f"{name}({obj.__qualname__})"
    if isinstance(obj, types.CodeType):
        return f"{name}({obj.co_qualname})"
    if isinstance(obj, types.FrameType):
        return f"{name}({obj.f_code.co_qualname})"
    if isinstance(obj, types.MethodType):
        return f"{name}({obj.__func__.__qualname__})"
    if isinstance(obj, torch.autograd.graph.Node):
        return f"{name}({obj.name()})"
    return name


def find_cycle(start: Any, members: set[int]) -> list[Any] | None:
    """Shortest path start -> ... -> start along gc.get_referents, within `members` (ids)."""
    parents: dict[int, Any] = {id(start): None}
    frontier = [start]
    seen = 1
    for _ in range(_CYCLE_DEPTH):
        next_frontier = []
        for obj in frontier:
            for ref in gc.get_referents(obj):
                if ref is start:
                    path = [obj]
                    while (parent := parents[id(path[-1])]) is not None:
                        path.append(parent)
                    return [*reversed(path), start]
                if id(ref) in members and id(ref) not in parents:
                    parents[id(ref)] = obj
                    next_frontier.append(ref)
                    seen += 1
                    if seen > _CYCLE_NODES:
                        return None
        frontier = next_frontier
    return None


def garbage_report(garbage: list[Any]) -> list[str]:
    """Lines on a list of unreachable objects (gc.garbage after a DEBUG_SAVEALL collection)."""
    if not garbage:
        return ["no unreachable objects"]
    members = {id(obj) for obj in garbage}
    by_type = collections.Counter(type(obj) for obj in garbage)
    lines = [f"by type ({len(by_type)} types, top {_TOP_TYPES}):"]
    lines += [f"  {n:7d} {type_name(t)}" for t, n in by_type.most_common(_TOP_TYPES)]

    # who refers to what, by type, within the garbage
    top = [t for t, _ in by_type.most_common(_SAMPLE_TYPES)]
    referrers: dict[type, collections.Counter[type]] = {t: collections.Counter() for t in top}
    for obj in garbage:
        for ref in gc.get_referents(obj):
            if id(ref) in members and type(ref) in referrers:
                referrers[type(ref)][type(obj)] += 1
    samples: dict[type, list[Any]] = {t: [] for t in top}
    for obj in garbage:
        bucket = samples.get(type(obj))
        if bucket is not None and len(bucket) < _SAMPLES_PER_TYPE:
            bucket.append(obj)
    # the cycles first: describing an object can create its __dict__, which is no member
    cycles = {
        t: next(filter(None, (find_cycle(obj, members) for obj in samples[t])), None)
        for t in top[:_CYCLE_TYPES]
    }
    sample_ids = {id(obj) for bucket in samples.values() for obj in bucket}
    sample_referrers: dict[int, list[str]] = collections.defaultdict(list)
    for obj in garbage:
        for ref in gc.get_referents(obj):
            if id(ref) in sample_ids:
                sample_referrers[id(ref)].append(type(obj).__name__)
    lines.append(f"top {len(top)} types: samples, and referrer types within the garbage:")
    for t in top:
        refs = ", ".join(
            f"{type_name(rt)} {n}" for rt, n in referrers[t].most_common(_REFERRER_TYPES)
        )
        lines.append(f"  {type_name(t)} ({by_type[t]}): referred to by {refs or 'nothing'}")
        for obj in samples[t]:
            names = collections.Counter(sample_referrers[id(obj)])
            held_by = ", ".join(f"{name} {n}" for name, n in names.most_common(4))
            lines.append(f"    {describe(obj)}  <- {held_by or '-'}")

    lines.append("reference cycles (shortest, through a sample object; [key], .attribute):")
    for t, path in cycles.items():
        if path is None:
            lines.append(f"  {type_name(t)}: samples on no cycle of <= {_CYCLE_DEPTH} steps")
            continue
        chain = _node(path[0])
        for obj, nxt in itertools.pairwise(path):
            chain += f" -{_step(obj, nxt)}> {_node(nxt)}"
        lines.append(f"  {chain}")
    return lines


def heap_report(
    census: collections.Counter[str], previous: collections.Counter[str] | None
) -> list[str]:
    """Lines on the tracked objects outside the frozen generation, by type."""
    lines = [
        f"unfrozen heap: {sum(census.values())} tracked objects "
        f"({gc.get_freeze_count()} frozen), top {_HEAP_TYPES}:"
    ]
    lines += [f"  {n:8d} {name}" for name, n in census.most_common(_HEAP_TYPES)]
    if previous is not None:
        change = census.copy()
        change.subtract(previous)
        total = sum(census.values()) - sum(previous.values())
        lines.append(f"change since the previous listed step: {total:+d}, largest increases:")
        grown = [(name, n) for name, n in change.most_common(_HEAP_TYPES) if n > 0]
        lines += [f"  {n:+8d} {name}" for name, n in grown] or ["  none"]
    return lines
