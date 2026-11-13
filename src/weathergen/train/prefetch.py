# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Prefetch the next training batch to the device in a background thread.

Without prefetching, the training loop's main thread receives the batch from a DataLoader
worker (`next`) and then runs `batch.to_device(device)` while the GPU has nothing queued. The
batch lives in pageable host memory, so every `cudaMemcpyAsync` stages through the driver and
only returns when the staging is done, and (on the compute stream) first waits for the previous
step's work. `DevicePrefetcher` moves both onto a background thread and a dedicated copy stream:

    thread:  next(loader_iter) -> wait for a GPU slot -> to_device on the copy stream
             -> record an event -> queue
    main:    queue.get -> compute_stream.wait_event(event) -> record_stream(compute_stream)

The batches are not pinned (`pin_memory` is only called if asked for, see `pin`).

GPU memory. At most ONE batch is copied to the device but not yet taken by the training loop,
in addition to the batch in use (about one extra batch of GPU memory, ~6 GB for the production
batch). A semaphore(1) is acquired before the copy and released when the main thread takes a
batch, so the thread may already hold the following batch on the CPU, but copies it only after
the previous prefetched batch has been taken. The batch the loop just finished with is dropped
by the loop right after it receives the next one; its memory is reusable once the compute
stream is done with it (stream-ordered, see `record_stream` below), so for a short moment the
copy of the next batch can overlap with that tail.

Streams. The tensors are allocated on the copy stream and freed on the compute stream. Without
`Tensor.record_stream(compute_stream)` the caching allocator would hand a freed block back to
the copy stream while kernels of the compute stream may still read it. `_record_streams` walks
every tensor of the batch (not only the ones `to_device` moves) and calls it.

Only the background thread touches the DataLoader iterator. Everything the DataLoader hands
over (the unpickled batch) is therefore created on the background thread; Python's GC is
process-wide, so `python_gc` (freeze, thresholds, manual collection at step end) is unaffected,
only the thread on which an automatic collection happens can now differ.

On a CPU (or any non-CUDA) device the same thread and queue are used, without streams, events
or `record_stream`.
"""

import contextlib
import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# blocking waits poll the stop event at this interval
_POLL_S = 0.1

_LEAVES = (str, bytes, int, float, bool, complex, type(None), type(np.empty(0)), np.generic)

_SENTINEL = object()


def iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor reachable from `obj` through containers and instance attributes.

    Generic on purpose (ModelBatch -> BatchSamples -> Sample -> StreamData/SampleMetaData and
    whatever `to_device` is extended with), so a new tensor field cannot be missed. Each object
    is visited once.
    """
    seen: set[int] = set()
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, _LEAVES) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            yield item
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend(item)
        elif isinstance(item, torch.nn.Module | type) or callable(item):
            # functions, bound methods (e.g. nvtx wrappers patched onto a batch), classes
            continue
        elif hasattr(item, "__dict__"):
            stack.extend(vars(item).values())


def _record_streams(batch: Any, stream: "torch.cuda.Stream") -> None:
    """Tell the allocator that `stream` uses every CUDA tensor of `batch`."""
    for tensor in iter_tensors(batch):
        if tensor.is_cuda:
            tensor.record_stream(stream)


class DevicePrefetcher:
    """Iterator that yields the batches of `loader_iter`, already on `device`.

    Parameters
    ----------
    loader_iter:
        Iterator of batches (e.g. `iter(DataLoader)`); only the background thread uses it.
    device:
        Target device. Batches need `to_device(device)`.
    pin:
        Call `batch.pin_memory()` in the thread before the copy (the `memory_pinning` option).
    annotate:
        Context-manager factory `annotate(name)` for nvtx ranges in the thread (the ranges are
        `prefetch_next_batch` and `prefetch_to_device`); no-op by default.
    wait_timeout_s:
        `close` waits this long for the thread, which cannot be interrupted inside a blocking
        `next(loader_iter)`; it is a daemon thread, so it never keeps the process alive.
    """

    def __init__(
        self,
        loader_iter: Iterator,
        device: torch.device | str,
        pin: bool = False,
        annotate: Callable[[str], contextlib.AbstractContextManager] | None = None,
        wait_timeout_s: float = 30.0,
    ) -> None:
        self._loader_iter = loader_iter
        self._device = torch.device(device)
        self._pin = pin
        self._annotate = (
            annotate if annotate is not None else (lambda name: contextlib.nullcontext())
        )
        self._wait_timeout_s = wait_timeout_s
        self._cuda = self._device.type == "cuda"
        self._copy_stream = torch.cuda.Stream(self._device) if self._cuda else None

        self._queue: queue.Queue = queue.Queue()
        # one GPU slot: a batch that is copied but not yet taken by the consumer
        self._slot = threading.Semaphore(1)
        self._stop = threading.Event()
        self._finished = False
        self._thread = threading.Thread(target=self._run, name="batch-prefetch", daemon=True)
        self._thread.start()

    # producer (background thread)

    def _acquire_slot(self) -> bool:
        while not self._stop.is_set():
            if self._slot.acquire(timeout=_POLL_S):
                return True
        return False

    def _put(self, item: Any) -> None:
        self._queue.put(item)

    def _run(self) -> None:
        try:
            if self._cuda:
                torch.cuda.set_device(self._device)
            while not self._stop.is_set():
                with self._annotate("prefetch_next_batch"):
                    batch = next(self._loader_iter)
                if not self._acquire_slot():
                    return
                if self._pin:
                    batch = batch.pin_memory()
                event = None
                with self._annotate("prefetch_to_device"):
                    if self._cuda:
                        with torch.cuda.stream(self._copy_stream):
                            batch.to_device(self._device)
                        event = torch.cuda.Event()
                        event.record(self._copy_stream)
                    else:
                        batch.to_device(self._device)
                self._put((batch, event))
                # do not keep the batch alive in this frame while waiting for the next one
                del batch, event
        except StopIteration:
            self._put((_SENTINEL, None))
        except BaseException as exc:  # re-raised in the consumer
            self._put((exc, None))

    # consumer (main thread)

    def __iter__(self) -> "DevicePrefetcher":
        return self

    def __next__(self) -> Any:
        if self._finished:
            raise StopIteration
        while True:
            try:
                item, event = self._queue.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    self._finished = True
                    raise RuntimeError("batch-prefetch thread died without a result") from None
        if item is _SENTINEL:
            self._finished = True
            raise StopIteration
        if isinstance(item, BaseException):
            self._finished = True
            raise item
        if self._cuda:
            stream = torch.cuda.current_stream(self._device)
            stream.wait_event(event)
            _record_streams(item, stream)
        # the copied batch has been taken: the thread may copy the next one
        self._slot.release()
        return item

    def close(self) -> None:
        """Stop the thread, wait for it (bounded) and drop batches still in the queue."""
        self._finished = True
        self._stop.set()
        self._thread.join(timeout=self._wait_timeout_s)
        if self._thread.is_alive():
            logger.warning(
                "batch-prefetch thread still busy in the data loader after "
                f"{self._wait_timeout_s:.0f} s; it is a daemon thread and exits afterwards."
            )
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
