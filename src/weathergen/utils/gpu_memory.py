# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Device memory of this process as the driver sees it, i.e. what `nvidia-smi` lists per process.

The CUDA caching allocator's counters (`torch.cuda.memory_reserved` and friends) only see the
memory PyTorch allocated. The process holds more: the CUDA context and the loaded kernels, NCCL
buffers, workspaces allocated directly with cudaMalloc, profiler buffers. NVML (`nvidia-ml-py`,
imported as `pynvml`) reports all of it, so `process - reserved` is the memory outside PyTorch.

PyTorch only returns cached memory to the device in `torch.cuda.empty_cache()` (or when an
allocation fails and it retries), so between those the reservation only grows: a reading right
after a step is the peak of that step, plus what is held outside PyTorch.

NVML is queried from the host and does not wait for the device, so reading it is cheap enough
for every training step. Without `pynvml`, or if the driver does not list this process, the
reader warns once and returns None from then on.
"""

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_GIB = 1024**3


class ProcessGpuMemory:
    """Reads the device memory this process uses on one GPU, via NVML."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._handle = None
        self._unavailable = False

    def used_bytes(self) -> int | None:
        """Device memory of this process on the GPU, or None if NVML cannot tell."""
        if self._unavailable:
            return None
        try:
            if self._handle is None:
                self._handle = _nvml_handle(self._device)
            return _process_used_bytes(self._handle, os.getpid())
        except Exception as e:  # no pynvml or driver, process not listed, ...
            self._unavailable = True
            logger.warning(f"Process GPU memory (NVML) unavailable, not logged: {e!r}")
            return None

    def fields(self) -> dict[str, float]:
        """
        `process_gib` and `non_pytorch_gib` (process minus the allocator's current
        reservation), or nothing if NVML cannot tell.
        """
        used = self.used_bytes()
        if used is None:
            return {}
        reserved = torch.cuda.memory_reserved(self._device)
        return {"process_gib": used / _GIB, "non_pytorch_gib": (used - reserved) / _GIB}


@functools.cache
def process_gpu_memory(device: torch.device) -> ProcessGpuMemory:
    """The reader of `device`, shared by all users in this process."""
    return ProcessGpuMemory(device)


def format_fields(fields: dict[str, float]) -> str:
    """`key=value` pairs with two decimals, separated by spaces."""
    return " ".join(f"{key}={value:.2f}" for key, value in fields.items())


def _nvml_handle(device: torch.device):
    """NVML handle of the torch device (NVML counts all GPUs, CUDA only the visible ones)."""
    import pynvml  # noqa: PLC0415 (optional: only needed when the memory is logged)

    pynvml.nvmlInit()
    # the mapping torch's own NVML helpers use (torch.cuda.device_memory_used)
    index = torch.cuda._get_nvml_device_index(device)
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def _process_used_bytes(handle, pid: int) -> int:
    """Device memory of process `pid` on the GPU of `handle`."""
    import pynvml  # noqa: PLC0415

    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if process.pid == pid:
            if process.usedGpuMemory is None:
                raise RuntimeError(f"NVML reports no memory use for process {pid}")
            return process.usedGpuMemory
    raise RuntimeError(f"NVML does not list process {pid} on this GPU")
