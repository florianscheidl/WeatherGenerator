# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Writing prepared validation/inference output (`validation_io.prepare_output`) to its store.

Kept apart from `validation_io`, which needs the model, so it imports without the GPU deps.
"""

import collections
from concurrent.futures import Future, ThreadPoolExecutor

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import ZarrWriteConfig, zarrio_writer


def write_output_store(store_path, write_config: ZarrWriteConfig, data: io.OutputBatchData) -> None:
    """Write prepared output (`validation_io.prepare_output`) to the store at `store_path`."""
    with zarrio_writer(store_path, write_config) as zio:
        for subset in data.items():
            zio.write_zarr(subset)


class OutputWriter:
    """
    Write prepared output (`validation_io.prepare_output`) of one batch to the run's results store.

    Synchronous by default. With `asynchronous`, the writes run on one background thread in
    submission order, so the next forecast step is computed while the previous one is
    written. At most `max_pending` writes are then outstanding: `submit` waits for the oldest
    one beyond that, which bounds the host memory held by queued output and re-raises a
    failed write on the caller's thread. A single thread, since a store must not be written
    concurrently (ZipStore takes no concurrent writers).
    """

    def __init__(self, cf, mini_epoch, asynchronous: bool = False, max_pending: int = 1):
        assert max_pending >= 1, f"max_pending must be >= 1, got {max_pending}."
        # resolved here, so the writer thread does not read the config
        self._store_path = config.get_path_results(cf, mini_epoch)
        self._write_config = ZarrWriteConfig.from_config(cf)
        self._max_pending = max_pending
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="output-writer")
            if asynchronous
            else None
        )
        self._pending: collections.deque[Future] = collections.deque()

    def submit(self, data: io.OutputBatchData) -> None:
        """Write `data`, or queue it when asynchronous."""
        # write_output_store is looked up per call, so a patched (nvtx-wrapped) one is used
        if self._executor is None:
            write_output_store(self._store_path, self._write_config, data)
            return
        while len(self._pending) >= self._max_pending:
            self._pending.popleft().result()
        self._pending.append(
            self._executor.submit(write_output_store, self._store_path, self._write_config, data)
        )

    def close(self) -> None:
        """Wait for the outstanding writes, re-raising the first failure, and stop the thread."""
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
