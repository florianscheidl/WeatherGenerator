# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Capture several separate Nsight Systems reports in a single training run.

Each window is one ``cudaProfilerStart``/``cudaProfilerStop`` pair. Launch the run with

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:2 -o trace_%n ...

so that nsys writes one report per window (``repeat:N`` must equal the number of windows).
Windows, in order of occurrence (first mini-epoch only):

- ``startup``: from before the data loader is iterated until the first model forward pass is
  done, i.e. data reading, preprocessing and tokenization up to the first forward.
- ``steady``: ``steady_steps`` full training steps, after ``warmup_steps`` untraced steps.

Config (all optional, disabled unless ``profiling.nsys_windows.enabled``)::

    profiling:
      nsys_windows:
        enabled: true
        startup: true
        steady: true
        warmup_steps: 2
        steady_steps: 4
        stop_after_capture: false

The environment variable ``WEATHERGEN_NSYS_WINDOWS`` (comma-separated subset of ``startup``,
``steady``; set by ``launch-slurm.py --nsys-windows``) enables exactly the listed windows and
takes precedence over ``startup``/``steady`` in the config, so that the number of windows
matches the ``repeat:N`` of the nsys command line; ``warmup_steps``/``steady_steps`` still come
from the config.

With ``stop_after_capture`` (or ``WEATHERGEN_NSYS_STOP_AFTER_CAPTURE=1``, set by
``launch-slurm.py --nsys-stop-after-capture``) the hook that closes the last enabled window
raises ``NsysCaptureDone``, which ``Trainer.run`` catches to return without validation or
checkpointing, so the nsys reports are finalized and the job ends. All ranks reach the same
step, so they leave together.

Each window boundary is also written as an ``nsys-window`` event to the io_timing records
(``WEATHERGEN_IO_TIMING=1``), so that these can be split per window afterwards.
"""

import logging
import os

import torch

from weathergen.utils.io_timing import io_event

_logger = logging.getLogger(__name__)

IO_EVENT_STREAM = "nsys-window"
STARTUP = "startup"
STEADY = "steady"
ENV_VAR = "WEATHERGEN_NSYS_WINDOWS"
ENV_VAR_STOP = "WEATHERGEN_NSYS_STOP_AFTER_CAPTURE"


class NsysCaptureDone(Exception):  # noqa: N818 (control flow, not an error)
    """Raised after the last enabled window closed, if the run is to end there."""


class NsysWindows:
    """Opens and closes the capture windows from the hooks called by the training loop."""

    def __init__(
        self,
        startup: bool = True,
        steady: bool = True,
        warmup_steps: int = 2,
        steady_steps: int = 4,
        stop_after_capture: bool = False,
    ) -> None:
        self.startup = startup
        self.steady = steady
        self.warmup_steps = warmup_steps
        self.steady_steps = steady_steps
        self.stop_after_capture = stop_after_capture
        self._open: str | None = None
        self._done: set[str] = set()

    @classmethod
    def from_config(cls, cf) -> "NsysWindows | None":
        """Return None unless ``profiling.nsys_windows.enabled`` or the environment enables it."""
        wcf = (cf.get("profiling") or {}).get("nsys_windows") or {}
        startup = wcf.get("startup", True)
        steady = wcf.get("steady", True)
        env = os.environ.get(ENV_VAR, "")
        if env:
            names = {n.strip() for n in env.split(",") if n.strip()}
            unknown = names - {STARTUP, STEADY}
            if unknown:
                raise ValueError(f"{ENV_VAR}: unknown window(s) {sorted(unknown)}")
            startup, steady = STARTUP in names, STEADY in names
        elif not wcf.get("enabled", False):
            return None
        return cls(
            startup=startup,
            steady=steady,
            warmup_steps=wcf.get("warmup_steps", 2),
            steady_steps=wcf.get("steady_steps", 4),
            stop_after_capture=wcf.get("stop_after_capture", False)
            or os.environ.get(ENV_VAR_STOP, "") in ("1", "true"),
        )

    def _start(self, name: str) -> None:
        if self._open is not None or name in self._done:
            return
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        io_event(IO_EVENT_STREAM, f"{name} open")
        _logger.info(f"nsys window '{name}' opened")
        self._open = name

    def _stop(self, name: str) -> None:
        if self._open != name:
            return
        # include the queued GPU work of the window in the capture
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        io_event(IO_EVENT_STREAM, f"{name} close")
        _logger.info(f"nsys window '{name}' closed")
        self._open = None
        self._done.add(name)
        enabled = {n for n, on in ((STARTUP, self.startup), (STEADY, self.steady)) if on}
        if self.stop_after_capture and enabled <= self._done:
            raise NsysCaptureDone

    def before_data_iter(self) -> None:
        """Call right before the data loader is iterated (workers start prefetching there)."""
        if self.startup:
            self._start(STARTUP)

    def after_forward(self, step: int) -> None:
        """Call after the model forward pass of training step ``step`` (0-based)."""
        if step == 0:
            self._stop(STARTUP)

    def before_step(self, step: int) -> None:
        if self.steady and step == self.warmup_steps:
            self._start(STEADY)

    def after_step(self, step: int) -> None:
        """Call at the end of training step ``step``, after the optimizer step."""
        if step == self.warmup_steps + self.steady_steps - 1:
            self._stop(STEADY)
