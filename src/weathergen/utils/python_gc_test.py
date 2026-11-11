# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import gc
import logging
import weakref
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.utils.performance import nvtx_wrap
from weathergen.utils.python_gc import GcConfig, GcControl, describe


@pytest.fixture(autouse=True)
def restore_gc():
    callbacks, threshold = list(gc.callbacks), gc.get_threshold()
    yield
    gc.callbacks[:] = callbacks
    gc.set_threshold(*threshold)
    gc.unfreeze()


def test_defaults_change_nothing():
    cfg = GcConfig.from_config(OmegaConf.create({}))
    assert cfg == GcConfig()
    callbacks, threshold = list(gc.callbacks), gc.get_threshold()
    frozen = gc.get_freeze_count()
    control = GcControl(cfg)
    control.install()
    control.before_training()
    assert list(control.steps(iter([(0, "a"), (1, "b")]))) == [(0, "a"), (1, "b")]
    assert gc.callbacks == callbacks
    assert gc.get_threshold() == threshold
    assert gc.get_freeze_count() == frozen


def test_section_is_read():
    cf = OmegaConf.create({"python_gc": {"freeze": True, "collect_every": 5}, "rank": 3})
    control = GcControl.from_config(cf)
    assert control.cfg == GcConfig(freeze=True, collect_every=5)
    assert control.rank == 3


def test_stats_count_collections_per_step():
    control = GcControl(GcConfig(log_stats=True))
    control.install()
    control.before_training()
    seen = []
    for _bidx, _ in control.steps(iter([(0, None), (1, None)])):
        gc.collect()
        seen.append((control.stats.count[2], control.stats.collected[2] >= 0))
    assert seen == [(1, True), (1, True)], "one full collection per step, reset after each"


def test_collect_every_takes_gen2_off_automatic_and_collects_at_the_step_end():
    control = GcControl(GcConfig(freeze=True, collect_every=2))
    control.install()
    frozen = gc.get_freeze_count()
    control.before_training()
    assert gc.get_freeze_count() > frozen
    assert gc.get_threshold()[2] == 2**31 - 1
    with patch("gc.collect", wraps=gc.collect) as collect:
        list(control.steps(iter([(i, None) for i in range(5)])))
    assert collect.call_count == 2, "after steps 2 and 4"


def test_nvtx_ranges_are_skipped_without_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        GcControl(GcConfig(nvtx_ranges=True)).install()
    assert all(getattr(cb, "__name__", "") != "_nvtx_callback" for cb in gc.callbacks)


class _Batch:
    def to_device(self):
        return "moved"


@pytest.mark.parametrize("once", [False, True])
def test_nvtx_wrap_once_breaks_the_reference_cycle(once):
    gc.disable()
    try:
        with patch("torch.cuda.nvtx.range_push"), patch("torch.cuda.nvtx.range_pop"):
            batch = _Batch()
            nvtx_wrap(batch, "to_device", "to_device", once=once)
            assert batch.to_device() == "moved"
            ref = weakref.ref(batch)
            del batch
            assert (ref() is None) == once, "without once, only the collector frees it"
    finally:
        gc.enable()


@pytest.mark.parametrize(("value", "steps"), [([3, 4], (3, 4)), (2, (2,)), (None, ())])
def test_debug_cycles_steps_are_read(value, steps):
    cf = OmegaConf.create({"python_gc": {"debug_cycles_steps": value}})
    assert GcConfig.from_config(cf).debug_cycles_steps == steps


class _Cyclic:
    def __init__(self):
        self.me = self
        self.payload = torch.ones(2, 3)


def test_debug_cycles_names_the_cycle_and_frees_it(caplog):
    control = GcControl(GcConfig(debug_cycles_steps=(1,)), rank=2)
    control.install()
    control.before_training()
    gc.collect()
    refs = []
    with caplog.at_level(logging.INFO, logger="weathergen.utils.python_gc"):
        for bidx, _ in control.steps(iter([(0, None), (1, None), (2, None)])):
            if bidx == 1:
                cyclic = _Cyclic()
                refs.append(weakref.ref(cyclic))
                del cyclic
    blocks = [r.getMessage() for r in caplog.records if "[gc-debug" in r.getMessage()]
    assert len(blocks) == 1, "only on the listed step"
    block = blocks[0]
    assert all(line.startswith("[gc-debug r2 b1]") for line in block.splitlines())
    assert "_Cyclic" in block, "the type is listed"
    assert "Tensor(shape=(2, 3), dtype=torch.float32" in block, "tensors are described"
    assert "_Cyclic -" in block and "> python_gc_test._Cyclic" in block.replace(
        "weathergen.utils.", ""
    ), "the cycle path"
    assert "unfrozen heap:" in block
    assert refs[0]() is None, "the saved garbage is freed in the same step"
    assert gc.get_debug() == 0
    assert not gc.garbage


def test_debug_cycles_heap_change_between_listed_steps(caplog):
    control = GcControl(GcConfig(debug_cycles_steps=(0, 1)))
    kept = []
    with caplog.at_level(logging.INFO, logger="weathergen.utils.python_gc"):
        for bidx, _ in control.steps(iter([(0, None), (1, None)])):
            if bidx == 1:
                kept.extend(_Cyclic() for _ in range(50))
    blocks = [r.getMessage() for r in caplog.records if "[gc-debug" in r.getMessage()]
    assert len(blocks) == 2
    assert "change since the previous listed step" not in blocks[0]
    grown = blocks[1].split("change since the previous listed step")[1]
    assert "+50" in grown and "_Cyclic" in grown, "the kept objects show as growth"


def test_describe_never_calls_repr_on_tensor_data():
    t = torch.zeros(4, requires_grad=True) * 2
    assert describe(t) == (
        "Tensor(shape=(4,), dtype=torch.float32, device=cpu, requires_grad=True, "
        "grad_fn=MulBackward0)"
    )
    assert describe({"a": t}) == "dict(len=1, keys=['a'])"
    assert describe([t, 1]) == "list(len=2, items=[Tensor, int])"
