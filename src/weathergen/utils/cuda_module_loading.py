# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
CUDA module loading mode (`CUDA_MODULE_LOADING`) from the run config.

With lazy loading (the driver default since CUDA 12.2) the driver loads a kernel the first
time it is launched. For cuBLAS this shows up as `cuKernelSetAttribute` calls of tens to
hundreds of milliseconds before the first GEMM of each new shape, on whichever thread
launches it. `EAGER` loads all kernels of a module when the module is loaded, which moves
that cost to startup (and uses more device memory).

The driver takes the variable from the environment when it is initialised (`cuInit`), so it
must be set before that. A variable set in the launching shell does not always reach
the training processes (e.g. sbatch with an explicit `--export=` list), hence this option.
PyTorch 2.9 no longer sets the variable itself.

Config key (absent from `config/default_config.yml`; the default leaves the environment
unchanged):

    cuda_module_loading: null   # EAGER or LAZY: set CUDA_MODULE_LOADING to this value

The training entry points apply it right after loading the config, before
`Trainer.init_torch` calls `torch.cuda.is_available()`, which is the first call that
initialises the CUDA driver (`cudaGetDeviceCount` -> `cuInit`). Importing the model code,
flash-attn included, does not initialise CUDA. The result is logged once the loggers are set
up, with the state of the driver at the time the variable was set.
"""

import ctypes
import dataclasses
import logging
import os

import torch

from weathergen.common.config import Config

logger = logging.getLogger(__name__)

ENV_VAR = "CUDA_MODULE_LOADING"
VALUES = ("EAGER", "LAZY")

_CUDA_SUCCESS = 0
_CUDA_ERROR_NOT_INITIALIZED = 3


def cuda_driver_initialized() -> bool | None:
    """Whether `cuInit` has run in this process; None if there is no CUDA driver to ask.

    `cuDeviceGetCount` returns CUDA_ERROR_NOT_INITIALIZED before `cuInit` and has no side
    effects. `torch.cuda.is_initialized()` is not enough: `torch.cuda.is_available()` calls
    `cuInit` without initialising torch's CUDA state.
    """
    if torch.version.cuda is None:
        return None
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    count = ctypes.c_int()
    status = libcuda.cuDeviceGetCount(ctypes.byref(count))
    if status == _CUDA_SUCCESS:
        return True
    if status == _CUDA_ERROR_NOT_INITIALIZED:
        return False
    return None


@dataclasses.dataclass(frozen=True)
class CudaModuleLoading:
    """What `apply` did: requested value, environment before/after, CUDA state at that time."""

    requested: str | None
    previous: str | None
    effective: str | None
    torch_cuda_initialized: bool
    driver_initialized: bool | None

    @property
    def too_late(self) -> bool:
        return self.requested is not None and (
            self.torch_cuda_initialized or bool(self.driver_initialized)
        )

    def log(self) -> None:
        """Log the effective value; warn if it was set after CUDA was initialised."""
        logger.info(
            f"{ENV_VAR}={self.effective or '<unset>'} "
            f"(config cuda_module_loading={self.requested}, previous "
            f"{self.previous or '<unset>'}); when set: "
            f"torch.cuda.is_initialized()={self.torch_cuda_initialized}, "
            f"CUDA driver initialized={self.driver_initialized}"
        )
        if self.requested is not None and self.previous not in (None, self.requested):
            logger.warning(f"cuda_module_loading replaced {ENV_VAR}={self.previous}.")
        if self.too_late:
            logger.warning(
                f"{ENV_VAR}={self.requested} was set after CUDA was initialised; "
                "it may have no effect."
            )


def apply(cf: Config) -> CudaModuleLoading:
    """Set CUDA_MODULE_LOADING from `cf.cuda_module_loading`; leave it alone if that is unset.

    Call before anything initialises CUDA. Call `.log()` on the result once logging is set up.
    """
    value = cf.get("cuda_module_loading", None)
    previous = os.environ.get(ENV_VAR)
    torch_cuda_initialized = torch.cuda.is_initialized()
    driver_initialized = cuda_driver_initialized()
    requested = None
    if value is not None:
        requested = str(value).upper()
        if requested not in VALUES:
            raise ValueError(f"cuda_module_loading must be one of {VALUES} or null, got {value!r}")
        os.environ[ENV_VAR] = requested
    return CudaModuleLoading(
        requested=requested,
        previous=previous,
        effective=os.environ.get(ENV_VAR),
        torch_cuda_initialized=torch_cuda_initialized,
        driver_initialized=driver_initialized,
    )
