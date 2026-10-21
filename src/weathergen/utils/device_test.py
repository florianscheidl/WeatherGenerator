import pytest
import torch

from weathergen.train.trainer_base import parse_slurm_tasks_per_node
from weathergen.utils.device import (
    get_device,
    get_device_type,
    non_blocking_copy,
    supports_memory_pinning,
    synchronize,
)


def test_get_device_without_accelerator(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    assert get_device_type() == "cpu"
    assert get_device(3) == torch.device("cpu")


def test_get_device_with_accelerator(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("mps"))
    assert get_device(0) == torch.device("mps", 0)


@pytest.mark.parametrize(
    "device, expected",
    [("cuda:1", True), ("xpu", True), ("mps", False), ("cpu", False)],
)
def test_async_copy_device_types(device, expected):
    assert supports_memory_pinning(device) == expected
    assert non_blocking_copy(torch.device(device)) == expected


def test_synchronize_cpu_is_noop():
    synchronize(torch.device("cpu"))


@pytest.mark.parametrize(
    "tasks_per_node, expected",
    [("1", 1), ("4", 4), ("16", 16), ("8(x4)", 8), ("12(x2),4", 12)],
)
def test_parse_slurm_tasks_per_node(tasks_per_node, expected):
    assert parse_slurm_tasks_per_node(tasks_per_node) == expected


def test_parse_slurm_tasks_per_node_invalid():
    with pytest.raises(ValueError):
        parse_slurm_tasks_per_node("")
