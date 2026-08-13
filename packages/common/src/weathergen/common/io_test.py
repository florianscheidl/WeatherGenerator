# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence 2.0.

import contextlib

import numpy as np
import pytest

from weathergen.common.io import (
    ItemKey,
    OutputDataset,
    OutputItem,
    TimeRange,
    zarrio_reader,
    zarrio_writer,
)


@pytest.mark.parametrize("suffix", ["zarr", "zip"])
def test_zarr_writer_profiles_store_item_groups_and_arrays(tmp_path, suffix):
    records: list[tuple[str, dict]] = []

    @contextlib.contextmanager
    def profile(name, metadata):
        yield
        records.append((name, dict(metadata)))

    key = ItemKey(sample=3, stream="ERA5", forecast_step=0)
    dataset = OutputDataset(
        name="target",
        item_key=key,
        source_interval=TimeRange("2020-01-01", "2020-01-02"),
        data=np.ones((2, 1), dtype=np.float32),
        times=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        coords=np.ones((2, 2), dtype=np.float32),
        geoinfo=np.empty((2, 0), dtype=np.float32),
        channels=["t"],
        geoinfo_channels=[],
    )
    store_path = tmp_path / f"output.{suffix}"

    with zarrio_writer(store_path, profile_context=profile) as writer:
        writer.write_zarr(OutputItem(key, forecast_offset=0, target=dataset))

    names = [name for name, _ in records]
    assert names.count("store_open") == 1
    assert names.count("store_close") == 1
    assert names.count("item_write") == 1
    assert names.count("item_group_create") == 1
    assert names.count("dataset_group_create") == 1
    assert names.count("array_create") == 4
    data_record = next(
        metadata
        for name, metadata in records
        if name == "array_create" and metadata["array"] == "data"
    )
    assert data_record["sample"] == 3
    assert data_record["stream"] == "ERA5"
    assert data_record["forecast_step"] == 0
    assert data_record["dataset"] == "target"
    assert data_record["shape"] == [2, 1]
    assert data_record["dtype"] == "float32"
    assert data_record["logical_bytes"] == 8
    assert "store_size_after_bytes" in data_record

    with zarrio_reader(store_path) as reader:
        written = reader.get_data(sample=3, stream="ERA5", forecast_step=0)
        assert written.target is not None
        np.testing.assert_array_equal(written.target.data[:], dataset.data)
