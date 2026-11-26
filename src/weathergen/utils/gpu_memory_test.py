# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import sys
import types

import pytest
import torch

from weathergen.utils import gpu_memory
from weathergen.utils.gpu_memory import ProcessGpuMemory, format_fields

_GIB = 1024**3


def _fake_pynvml(processes: list[types.SimpleNamespace]) -> types.SimpleNamespace:
    """A pynvml whose GPU runs `processes`; counts the process queries."""
    fake = types.SimpleNamespace(queries=0, handles=[])

    def get_processes(handle):
        fake.queries += 1
        return processes

    def get_handle(index):
        fake.handles.append(index)
        return f"handle{index}"

    fake.nvmlInit = lambda: None
    fake.nvmlDeviceGetHandleByIndex = get_handle
    fake.nvmlDeviceGetComputeRunningProcesses = get_processes
    return fake


@pytest.fixture
def cuda_device(monkeypatch):
    """A CUDA device index mapping and allocator reservation of 10 GiB, without a GPU."""
    monkeypatch.setattr(torch.cuda, "_get_nvml_device_index", lambda device: 3, raising=False)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 10 * _GIB)
    return torch.device("cuda", 1)


def test_fields_are_this_process_and_its_memory_outside_pytorch(monkeypatch, cuda_device):
    other = types.SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=50 * _GIB)
    this = types.SimpleNamespace(pid=os.getpid(), usedGpuMemory=int(11.5 * _GIB))
    fake = _fake_pynvml([other, this])
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    fields = ProcessGpuMemory(cuda_device).fields()

    assert fields == {"process_gib": 11.5, "non_pytorch_gib": 1.5}
    assert fake.handles == [3], "the NVML index, not the CUDA one"


def test_an_unlisted_process_warns_once_and_is_not_queried_again(monkeypatch, cuda_device, caplog):
    other = types.SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=_GIB)
    fake = _fake_pynvml([other])
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    reader = ProcessGpuMemory(cuda_device)

    assert reader.fields() == {}
    assert reader.fields() == {}
    assert fake.queries == 1
    assert len([r for r in caplog.records if "NVML" in r.getMessage()]) == 1


def test_without_pynvml_nothing_is_logged(monkeypatch, cuda_device):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError

    assert ProcessGpuMemory(cuda_device).fields() == {}


def test_one_reader_per_device():
    gpu_memory.process_gpu_memory.cache_clear()
    device = torch.device("cuda", 0)

    assert gpu_memory.process_gpu_memory(device) is gpu_memory.process_gpu_memory(device)


def test_format_fields():
    assert format_fields({"process_gib": 11.5, "non_pytorch_gib": 1.567}) == (
        "process_gib=11.50 non_pytorch_gib=1.57"
    )
