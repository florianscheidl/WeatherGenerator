# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from weathergen.utils import cuda_module_loading
from weathergen.utils.cuda_module_loading import ENV_VAR

_real_driver_check = cuda_module_loading.cuda_driver_initialized


@pytest.fixture(autouse=True)
def no_cuda_state():
    """Run as on a fresh process without a CUDA driver, whatever machine runs the tests."""
    with (
        patch("torch.cuda.is_initialized", return_value=False),
        patch.object(cuda_module_loading, "cuda_driver_initialized", return_value=None),
    ):
        yield


def _cf(**kwargs):
    return OmegaConf.create(kwargs)


def test_unset_leaves_environment_alone(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    result = cuda_module_loading.apply(_cf())
    assert ENV_VAR not in os.environ
    assert result.requested is None and result.effective is None

    monkeypatch.setenv(ENV_VAR, "LAZY")
    result = cuda_module_loading.apply(_cf(cuda_module_loading=None))
    assert os.environ[ENV_VAR] == "LAZY"
    assert result.requested is None and result.effective == "LAZY"


@pytest.mark.parametrize("value", ["EAGER", "eager", "LAZY"])
def test_value_is_set(monkeypatch, value):
    monkeypatch.delenv(ENV_VAR, raising=False)
    result = cuda_module_loading.apply(_cf(cuda_module_loading=value))
    assert os.environ[ENV_VAR] == value.upper()
    assert result.effective == value.upper()
    assert result.previous is None
    assert not result.too_late


def test_config_replaces_existing_value(monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, "LAZY")
    result = cuda_module_loading.apply(_cf(cuda_module_loading="EAGER"))
    assert os.environ[ENV_VAR] == "EAGER"
    with caplog.at_level(logging.INFO, logger=cuda_module_loading.__name__):
        result.log()
    assert "CUDA_MODULE_LOADING=EAGER" in caplog.text
    assert "replaced CUDA_MODULE_LOADING=LAZY" in caplog.text


def test_invalid_value_raises(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="cuda_module_loading"):
        cuda_module_loading.apply(_cf(cuda_module_loading="SOMETIMES"))
    assert ENV_VAR not in os.environ


@pytest.mark.parametrize("torch_init, driver_init", [(True, None), (False, True)])
def test_warns_when_cuda_already_initialised(monkeypatch, caplog, torch_init, driver_init):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with (
        patch("torch.cuda.is_initialized", return_value=torch_init),
        patch.object(cuda_module_loading, "cuda_driver_initialized", return_value=driver_init),
    ):
        result = cuda_module_loading.apply(_cf(cuda_module_loading="EAGER"))
    assert result.too_late
    with caplog.at_level(logging.INFO, logger=cuda_module_loading.__name__):
        result.log()
    assert "may have no effect" in caplog.text


def test_driver_check_without_cuda():
    # the real check, unpatched: a torch build without CUDA -> None, no driver call
    with patch("torch.version.cuda", None):
        assert _real_driver_check() is None
