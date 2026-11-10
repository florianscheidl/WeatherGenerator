# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import gc
import weakref
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from weathergen.utils.performance import nvtx_wrap
from weathergen.utils.python_gc import GcConfig, GcControl


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
