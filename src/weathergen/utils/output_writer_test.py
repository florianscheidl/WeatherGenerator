# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import threading

import numpy as np
import pytest
from numpy.typing import NDArray
from omegaconf import OmegaConf

from weathergen.common import config
from weathergen.common.io import IOReaderData, OutputBatchData, TimeRange, zarrio_reader
from weathergen.utils import output_writer
from weathergen.utils.output_writer import OutputWriter

N_POINTS, N_CHANNELS = 7, 3


@pytest.fixture
def store_path(tmp_path, monkeypatch):
    path = tmp_path / "validation_chkpt00000_rank0000.zip"
    monkeypatch.setattr(config, "get_path_results", lambda cf, mini_epoch: path)
    return path


@pytest.fixture
def recorded_writes(monkeypatch):
    """Replace the store write by one that records (data, thread name)."""
    writes = []

    def write(store_path, write_config, data):
        writes.append((data, threading.current_thread().name))

    monkeypatch.setattr(output_writer, "write_output_store", write)
    return writes


def _writer(asynchronous: bool, max_pending: int = 1) -> OutputWriter:
    return OutputWriter(OmegaConf.create({}), 0, asynchronous, max_pending)


def test_synchronous_writer_writes_on_the_caller_thread(store_path, recorded_writes):
    writer = _writer(asynchronous=False)
    writer.submit("step1")
    assert recorded_writes == [("step1", threading.current_thread().name)]
    writer.close()


def test_asynchronous_writer_writes_in_order_on_its_own_thread(store_path, recorded_writes):
    writer = _writer(asynchronous=True, max_pending=2)
    for step in range(5):
        writer.submit(step)
    writer.close()

    assert [data for data, _ in recorded_writes] == list(range(5))
    assert all(name.startswith("output-writer") for _, name in recorded_writes)


def test_submit_waits_while_max_pending_writes_are_outstanding(store_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(output_writer, "write_output_store", lambda *args: release.wait())
    writer = _writer(asynchronous=True, max_pending=1)

    writer.submit("step1")  # outstanding, blocked in the write
    second = threading.Thread(target=writer.submit, args=("step2",))
    second.start()
    second.join(timeout=0.2)
    assert second.is_alive(), "submit returned while max_pending writes were outstanding"

    release.set()
    second.join(timeout=5)
    assert not second.is_alive()
    writer.close()


def test_failed_write_is_raised_on_the_caller_thread(store_path, monkeypatch):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(output_writer, "write_output_store", fail)
    writer = _writer(asynchronous=True)
    writer.submit("step1")
    with pytest.raises(OSError, match="disk full"):
        writer.close()


def _step_without_targets(forecast_step: int) -> dict[str, NDArray]:
    """Predictions at points without values; no points at the source step 0."""
    n_points = 0 if forecast_step == 0 else N_POINTS
    rng = np.random.default_rng(forecast_step)
    return {
        "targets": np.zeros((0, N_CHANNELS), dtype=np.float32),
        "predictions": rng.random((1, n_points, N_CHANNELS), dtype=np.float32),
        "coords": rng.random((n_points, 2), dtype=np.float32),
        "times": np.repeat(np.datetime64("2020-01-01T12:00", "ns"), n_points),
        "lens": [n_points],
    }


def _output_without_targets(forecast_steps: list[int]) -> OutputBatchData:
    """One sample and stream, as `prepare_output` collects a chunk of `forecast_steps`."""
    steps = [_step_without_targets(step) for step in forecast_steps]
    source = IOReaderData(
        coords=np.zeros((N_POINTS, 2), dtype=np.float32),
        geoinfos=np.zeros((N_POINTS, 0), dtype=np.float32),
        data=np.ones((N_POINTS, N_CHANNELS), dtype=np.float32),
        datetimes=np.repeat(np.datetime64("2020-01-01T06:00", "ns"), N_POINTS),
    )
    return OutputBatchData(
        sources=[[source]],
        source_intervals=[TimeRange("2020-01-01T00:00", "2020-01-01T06:00")],
        targets=[[step["targets"]] for step in steps],
        predictions=[[step["predictions"]] for step in steps],
        targets_coords=[[step["coords"]] for step in steps],
        targets_times=[[step["times"]] for step in steps],
        targets_lens=[[step["lens"]] for step in steps],
        streams={"ERA5": 0},
        target_channels=[[f"c{i}" for i in range(N_CHANNELS)]],
        source_channels=[[f"c{i}" for i in range(N_CHANNELS)]],
        geoinfo_channels=[[]],
        latents=None,
        sample_start=0,
        forecast_offset=1,
        forecast_steps=forecast_steps,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rollout_without_targets_round_trips(store_path, asynchronous):
    # chunk_size 1: the first chunk carries the source step 0
    chunks = [[0, 1], [2], [3]]
    writer = _writer(asynchronous)
    for chunk in chunks:
        writer.submit(_output_without_targets(chunk))
    writer.close()

    with zarrio_reader(store_path) as zio:
        assert zio.forecast_offset == 1
        for step in [1, 2, 3]:
            expected = _step_without_targets(step)
            item = zio.get_data(0, "ERA5", step)

            prediction = item.prediction
            np.testing.assert_array_equal(
                np.asarray(prediction.data), expected["predictions"].transpose(1, 2, 0)
            )
            np.testing.assert_array_equal(np.asarray(prediction.coords), expected["coords"])
            np.testing.assert_array_equal(np.asarray(prediction.times), expected["times"])

            # the target has no values and so no points: coords and times are written once
            assert np.asarray(item.target.data).shape == (0, N_CHANNELS)
            assert np.asarray(item.target.coords).shape == (0, 2)
            assert np.asarray(item.target.times).shape == (0,)
