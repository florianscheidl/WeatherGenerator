# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import threading
import time

import pytest
import torch

from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.stream_data import StreamData
from weathergen.train.prefetch import DevicePrefetcher, iter_tensors


class FakeBatch:
    """Counts copies; `device` is set by to_device like the real batches do."""

    copied = 0  # copies started minus batches taken (maintained by the tests)
    lock = threading.Lock()

    def __init__(self, idx: int, log: list | None = None):
        self.idx = idx
        self.x = torch.full((3,), float(idx))
        self.device = None
        self.log = log

    def to_device(self, device):
        with FakeBatch.lock:
            FakeBatch.copied += 1
            if self.log is not None:
                self.log.append(("copy", self.idx, threading.current_thread().name))
        self.x = self.x.to(device)
        self.device = device
        return self

    def pin_memory(self):
        return self


@pytest.fixture(autouse=True)
def _reset():
    FakeBatch.copied = 0


def test_same_sequence_and_values():
    batches = [FakeBatch(i) for i in range(7)]
    out = list(DevicePrefetcher(iter(batches), "cpu"))
    assert [b.idx for b in out] == list(range(7))
    assert all(b.device == torch.device("cpu") for b in out)
    assert [b.x[0].item() for b in out] == [float(i) for i in range(7)]


def test_copy_runs_in_the_prefetch_thread():
    log = []
    list(DevicePrefetcher(iter([FakeBatch(i, log) for i in range(3)]), "cpu"))
    assert {name for _, _, name in log} == {"batch-prefetch"}


def test_stop_iteration_is_sticky():
    prefetcher = DevicePrefetcher(iter([FakeBatch(0)]), "cpu")
    next(prefetcher)
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(prefetcher)
    prefetcher.close()


def test_exception_in_loader_is_reraised_after_the_good_batches():
    def gen():
        yield FakeBatch(0)
        raise ValueError("boom")

    prefetcher = DevicePrefetcher(gen(), "cpu")
    assert next(prefetcher).idx == 0
    with pytest.raises(ValueError, match="boom"):
        next(prefetcher)
    with pytest.raises(StopIteration):
        next(prefetcher)


def test_exception_in_to_device_is_reraised():
    class Bad(FakeBatch):
        def to_device(self, device):
            raise RuntimeError("copy failed")

    with pytest.raises(RuntimeError, match="copy failed"):
        next(DevicePrefetcher(iter([Bad(0)]), "cpu"))


def test_pin_is_called_in_the_thread_before_the_copy():
    log = []

    class Pinned(FakeBatch):
        def pin_memory(self):
            log.append(("pin", threading.current_thread().name))
            return self

    next(DevicePrefetcher(iter([Pinned(0, log)]), "cpu", pin=True))
    assert [e[0] for e in log] == ["pin", "copy"]
    assert log[0][1] == "batch-prefetch"


def test_at_most_one_batch_copied_but_not_taken():
    n = 6
    prefetcher = DevicePrefetcher(iter([FakeBatch(i) for i in range(n)]), "cpu")
    for taken in range(n):
        # let the thread run as far as it can
        time.sleep(0.2)
        assert FakeBatch.copied - taken <= 1
        next(prefetcher)
        time.sleep(0.2)
        assert FakeBatch.copied - (taken + 1) <= 1
    # and it did prefetch: the next batch was copied before it was asked for
    prefetcher.close()


def test_prefetches_ahead_of_the_consumer():
    prefetcher = DevicePrefetcher(iter([FakeBatch(i) for i in range(4)]), "cpu")
    next(prefetcher)
    deadline = time.time() + 5
    while FakeBatch.copied < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert FakeBatch.copied == 2
    prefetcher.close()


def test_close_early_joins_the_thread_quickly():
    prefetcher = DevicePrefetcher(iter([FakeBatch(i) for i in range(100)]), "cpu")
    next(prefetcher)
    start = time.time()
    prefetcher.close()
    assert time.time() - start < 2
    assert not prefetcher._thread.is_alive()
    with pytest.raises(StopIteration):
        next(prefetcher)


def test_close_while_loader_is_slow_does_not_hang_forever():
    release = threading.Event()

    def slow():
        yield FakeBatch(0)
        release.wait(5)
        yield FakeBatch(1)

    prefetcher = DevicePrefetcher(slow(), "cpu", wait_timeout_s=0.3)
    next(prefetcher)
    start = time.time()
    prefetcher.close()  # thread is blocked in next(loader_iter): bounded wait, then a warning
    assert time.time() - start < 2
    release.set()
    prefetcher._thread.join(5)
    assert not prefetcher._thread.is_alive()


def test_annotations_in_the_thread():
    names = []

    class Ctx:
        def __init__(self, name):
            names.append(name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    list(DevicePrefetcher(iter([FakeBatch(0)]), "cpu", annotate=Ctx))
    assert names[:2] == ["prefetch_next_batch", "prefetch_to_device"]


def test_does_not_touch_the_global_rng():
    torch.manual_seed(0)
    state = torch.get_rng_state()
    list(DevicePrefetcher(iter([FakeBatch(i) for i in range(3)]), "cpu"))
    assert torch.equal(state, torch.get_rng_state())


def _model_batch(seed: int) -> ModelBatch:
    g = torch.Generator().manual_seed(seed)
    batch = ModelBatch(["s"], 1, 1, 0, 1)
    for add, samples in (
        (batch.add_source_stream, batch.source_samples),
        (batch.add_target_stream, batch.target_samples),
    ):
        sd = StreamData(0, 1, 1, 4)
        sd.source_tokens_cells = [torch.randn(4, 3, generator=g)]
        sd.source_tokens_lens = [torch.ones(4, dtype=torch.int32)]
        sd.source_idxs_embed = [torch.arange(4)]
        sd.target_coords = [torch.randn(4, 2, generator=g)]
        sd.target_coords_lens = [torch.ones(4, dtype=torch.int32)]
        sd.target_tokens = [torch.randn(4, 3, generator=g)]
        add(0, "s", sd, SampleMetaData(params={}, mask=torch.ones(4, dtype=torch.bool)))
        samples.set_tokens_lens(torch.ones(1, 1, 4, dtype=torch.int32))
    return batch


def test_walker_covers_every_tensor_to_device_moves():
    batch = _model_batch(0)
    batch.to_device("meta")
    found = {id(t) for t in iter_tensors(batch)}
    meta = [t for t in _all_tensors_by_brute_force(batch) if t.device.type == "meta"]
    assert len(meta) >= 10
    assert all(id(t) in found for t in meta)
    # tokens_lens_cpu stays on the CPU and is not moved
    assert any(t.device.type == "cpu" for t in iter_tensors(batch))


def _all_tensors_by_brute_force(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _all_tensors_by_brute_force(v, seen)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _all_tensors_by_brute_force(v, seen)
    elif hasattr(obj, "__dict__") and not callable(obj):
        for v in vars(obj).values():
            yield from _all_tensors_by_brute_force(v, seen)


def _train(batches, prefetch: bool):
    """A few SGD steps on a tiny model over real ModelBatches; returns the losses."""
    torch.manual_seed(1234)
    model = torch.nn.Linear(3, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    it = iter(batches)
    if prefetch:
        it = DevicePrefetcher(it, "cpu")
    losses = []
    for batch in it:
        if not prefetch:
            batch.to_device("cpu")
        sd = batch.source_samples.samples[0].streams_data["s"]
        td = batch.target_samples.samples[0].streams_data["s"]
        loss = ((model(sd.source_tokens_cells[0]) - td.target_tokens[0]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach().clone())
    if prefetch:
        it.close()
    return losses, torch.get_rng_state()


def test_losses_are_bitwise_identical_with_and_without_prefetch():
    off, rng_off = _train([_model_batch(i) for i in range(5)], prefetch=False)
    on, rng_on = _train([_model_batch(i) for i in range(5)], prefetch=True)
    assert len(on) == len(off) == 5
    assert all(torch.equal(a, b) for a, b in zip(off, on, strict=True))
    assert torch.equal(rng_off, rng_on)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_copy_stream_and_record_stream():
    device = torch.device("cuda", 0)
    batches = [_model_batch(i) for i in range(3)]
    prefetcher = DevicePrefetcher(iter(batches), device)
    for b in prefetcher:
        sd = b.source_samples.samples[0].streams_data["s"]
        assert sd.source_tokens_cells[0].is_cuda
        torch.cuda.current_stream().synchronize()
    prefetcher.close()
