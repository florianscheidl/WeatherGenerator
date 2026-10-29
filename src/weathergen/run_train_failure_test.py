# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from weathergen import run_train


class _ExitCalledError(Exception):
    pass


def _fail(cf, env: dict[str, str]) -> list:
    """Call _handle_failure inside an except block; return what it ran."""
    calls = []

    def fake_exit(code):
        calls.append(("exit", code))
        raise _ExitCalledError

    with (
        patch.dict(run_train.os.environ, env, clear=True),
        patch.object(run_train.os, "_exit", fake_exit),
        patch.object(run_train.shutil, "which", lambda name: f"/usr/bin/{name}"),
        patch.object(run_train.subprocess, "run", lambda cmd, check: calls.append(cmd)),
        patch.object(run_train.pdb, "post_mortem", lambda tb: calls.append("pdb")),
        patch.object(run_train.phase_timer.timer, "write_report", lambda: calls.append("report")),
    ):
        try:
            raise RuntimeError("DataLoader worker is killed by signal: Bus error")
        except RuntimeError:
            with contextlib.suppress(_ExitCalledError):
                run_train._handle_failure(cf)
    return calls


def test_failed_rank_cancels_its_slurm_step_and_exits_nonzero():
    cf = OmegaConf.create({"world_size": 4, "rank": 2})

    calls = _fail(cf, {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0"})

    assert calls == ["report", ["scancel", "123.0"], ("exit", 1)]


def test_failed_rank_outside_slurm_exits_nonzero():
    cf = OmegaConf.create({"world_size": 4, "rank": 2})

    assert _fail(cf, {}) == ["report", ("exit", 1)]


@pytest.mark.parametrize("cf", [{"world_size": 1}, {}])
def test_single_process_failure_is_debugged(cf):
    assert _fail(OmegaConf.create(cf), {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0"}) == ["pdb"]
