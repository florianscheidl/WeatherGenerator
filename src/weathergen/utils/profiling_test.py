# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from weathergen.common.config import _DEFAULT_CONFIG_PTH
from weathergen.utils import profiling


def test_profiling_config_defaults_when_section_is_absent():
    assert profiling.ProfilingConfig.from_config(OmegaConf.create({})) == (
        profiling.ProfilingConfig()
    )


def test_default_config_disables_memory_snapshot():
    cfg = OmegaConf.load(_DEFAULT_CONFIG_PTH)

    assert not profiling.ProfilingConfig.from_config(cfg).records_memory_snapshot


def test_memory_snapshot_needs_parent_and_collector_flags():
    parent_disabled = OmegaConf.create(
        {"profiling": {"enabled": False, "memory_snapshot": {"enabled": True}}}
    )
    collector_disabled = OmegaConf.create(
        {"profiling": {"enabled": True, "memory_snapshot": {"enabled": False}}}
    )
    enabled = OmegaConf.create(
        {"profiling": {"enabled": True, "memory_snapshot": {"enabled": True}}}
    )

    assert not profiling.ProfilingConfig.from_config(parent_disabled).records_memory_snapshot
    assert not profiling.ProfilingConfig.from_config(collector_disabled).records_memory_snapshot
    assert profiling.ProfilingConfig.from_config(enabled).records_memory_snapshot


def test_memory_snapshot_session_records_and_dumps_on_exception(monkeypatch, tmp_path):
    calls = []
    cfg = OmegaConf.create({"profiling": {"enabled": True, "memory_snapshot": {"enabled": True}}})
    monkeypatch.setattr(profiling, "is_root", lambda: True)
    monkeypatch.setattr(profiling, "get_rank", lambda: 0)
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profiling.config, "get_path_profiling_traces", lambda unused_cf: tmp_path)
    monkeypatch.setattr(
        profiling.torch.cuda.memory,
        "_record_memory_history",
        lambda **kwargs: calls.append(("record", kwargs)),
    )
    monkeypatch.setattr(
        profiling.torch.cuda.memory,
        "_dump_snapshot",
        lambda path: calls.append(("dump", Path(path))),
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        with profiling.memory_snapshot_session(cfg):
            raise RuntimeError("inference failed")

    assert calls[0] == ("record", {"max_entries": 100_000})
    assert calls[1][0] == "dump"
    assert calls[1][1].parent == tmp_path
    assert calls[1][1].name.endswith("_rank_0_memory_snapshot.pickle")
    assert calls[2] == ("record", {"enabled": None})


def test_memory_snapshot_session_is_noop_when_disabled(monkeypatch):
    cfg = OmegaConf.create({"profiling": {"enabled": False, "memory_snapshot": {"enabled": True}}})
    monkeypatch.setattr(
        profiling.torch.cuda.memory,
        "_record_memory_history",
        lambda **kwargs: pytest.fail(f"unexpected memory-history call: {kwargs}"),
    )

    with profiling.memory_snapshot_session(cfg):
        pass


def test_memory_snapshot_session_is_noop_off_root(monkeypatch):
    cfg = OmegaConf.create({"profiling": {"enabled": True, "memory_snapshot": {"enabled": True}}})
    monkeypatch.setattr(profiling, "is_root", lambda: False)
    monkeypatch.setattr(
        profiling.torch.cuda.memory,
        "_record_memory_history",
        lambda **kwargs: pytest.fail(f"unexpected memory-history call: {kwargs}"),
    )

    with profiling.memory_snapshot_session(cfg):
        pass
