# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Device selection independent of the accelerator vendor.

ROCm builds of torch report AMD GPUs as "cuda", so CUDA and ROCm share one code path;
Apple Silicon reports "mps", Intel GPUs "xpu". Without an accelerator, torch runs on "cpu".
The config key `device` ("auto" by default) selects a device type explicitly, e.g. "cpu" on a
machine with a GPU; it is applied with set_device_type before any device is chosen.

NOTE: the "xpu" (Intel GPU) code paths are implemented but untested, for lack of hardware.
"""

import torch

DEVICE_TYPES = ("cpu", "cuda", "mps", "xpu")

# Device types for which pinned host memory speeds up host-to-device copies and for which
# non_blocking host-to-device copies are safe. MPS uses unified memory, so pinning is
# pointless there, and a non_blocking copy to MPS returns wrong data when the host tensor is
# freed before the copy completes (torch 2.9.1), as happens with temporaries.
_ASYNC_COPY_DEVICE_TYPES = ("cuda", "xpu")

# set by set_device_type; None selects the available accelerator
_device_type_override: str | None = None


def _available_device_type() -> str:
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def set_device_type(device_type: str) -> None:
    """
    Select the device type: "auto" for the available accelerator (or the CPU), or one of
    DEVICE_TYPES. A requested accelerator must be the one that is available.
    """
    global _device_type_override
    if device_type == "auto":
        _device_type_override = None
        return
    if device_type not in DEVICE_TYPES:
        raise ValueError(f"device must be 'auto' or one of {DEVICE_TYPES}, got {device_type!r}")
    available = _available_device_type()
    if device_type != "cpu" and device_type != available:
        raise ValueError(
            f"device {device_type!r} requested, but the available one is {available!r}"
        )
    _device_type_override = device_type


def get_device_type() -> str:
    """Type of the selected device ("cuda", "mps", "xpu", ...), or "cpu"."""
    if _device_type_override is not None:
        return _device_type_override
    return _available_device_type()


def get_device(local_rank: int = 0) -> torch.device:
    """Device for the given node-local rank: the accelerator with that index, or the CPU."""
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, local_rank)


def supports_memory_pinning(device: torch.device | str) -> bool:
    """Whether pinning host memory is useful for copies to `device`."""
    return torch.device(device).type in _ASYNC_COPY_DEVICE_TYPES


def non_blocking_copy(device: torch.device | str) -> bool:
    """Whether host-to-device copies to `device` may use non_blocking=True."""
    return torch.device(device).type in _ASYNC_COPY_DEVICE_TYPES


def synchronize(device: torch.device | str) -> None:
    """Wait for all queued work on `device`; no-op on the CPU."""
    device = torch.device(device)
    if device.type != "cpu":
        torch.accelerator.synchronize(device)
