# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Opt-in wall-clock timing of data reading I/O and preprocessing.

Enable with the environment variable ``WEATHERGEN_IO_TIMING=1``. Timings are logged on the
``weathergen.io_timing`` logger at INFO as ``io_timing : <label> : <seconds>``, one line per
timed call. The variable is read once at import, so it has to be set before the process
(and its data loader workers) start. When disabled, ``io_timer`` is a no-op.
"""

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager

_logger = logging.getLogger("weathergen.io_timing")

IO_TIMING_ENABLED = os.environ.get("WEATHERGEN_IO_TIMING", "0").lower() in ("1", "true", "yes")


@contextmanager
def io_timer(label: str) -> Generator[None]:
    """Log the wall-clock time spent in the ``with`` block under ``label``."""
    if not IO_TIMING_ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _logger.info(f"io_timing : {label} : {time.perf_counter() - t0:.6f}")
