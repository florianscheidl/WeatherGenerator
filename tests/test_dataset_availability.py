# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the dataset-availability analysis scripts in packages/science/."""

import csv
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
import zarr

_SCIENCE_DIR = pathlib.Path(__file__).parent.parent / "packages" / "science"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCIENCE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cda = _load_module("compute_dataset_availability")
pda = _load_module("plot_dataset_availability")


@pytest.fixture
def obs_zarr(tmp_path: pathlib.Path) -> pathlib.Path:
    """Observation-style zarr: hourly reports for two days with a 6-hour gap on day one.

    Channel obsvalue_a is always valid, obsvalue_b is NaN in every second report.
    """
    n_per_hour = 4
    hours = [h for h in range(48) if not 6 <= h < 12]
    dates = np.concatenate(
        [
            np.full(n_per_hour, np.datetime64("2020-01-01T00:00:00") + np.timedelta64(h, "h"))
            for h in hours
        ]
    )
    n = len(dates)
    data = np.ones((n, 4), dtype=np.float32)
    data[::2, 3] = np.nan  # obsvalue_b missing in every second row

    path = tmp_path / "obs.zarr"
    g = zarr.open_group(str(path), mode="w")
    arr = g.create_array("data", shape=data.shape, dtype=data.dtype, chunks=(16, 4))
    arr[:] = data
    arr.attrs["colnames"] = ["lat", "lon", "obsvalue_a", "obsvalue_b"]
    d = g.create_array("dates", shape=(n, 1), dtype="datetime64[s]", chunks=(16, 1))
    d[:] = dates.astype("datetime64[s]").reshape(-1, 1)
    return path


def test_accumulate_counts_sorted_and_unsorted():
    for order in (np.arange(6), np.array([3, 0, 4, 1, 5, 2])):
        n_present = np.zeros((3, 2), dtype=np.int32)
        bin_idx = np.array([0, 0, 1, 1, 2, 2])[order]
        finite = np.array([[1, 0], [1, 1], [0, 0], [1, 0], [1, 1], [1, 1]], dtype=bool)[order]
        cda.accumulate_counts(n_present, bin_idx, finite)
        assert n_present.tolist() == [[2, 1], [1, 0], [2, 2]]


def test_analyze_obs_gaps_and_completeness(obs_zarr: pathlib.Path):
    r = cda.analyze_obs(obs_zarr, "SYNOP", bin_seconds=3600, start=None, end=None)

    assert r.channels == ["obsvalue_a", "obsvalue_b"]
    assert r.total_rows == 42 * 4
    # the 6h gap has zero values; every other hourly bin has 4 rows (obsvalue_b half NaN)
    gap = slice(6, 12)
    assert (r.n_present[gap] == 0).all()
    assert (r.n_expected[gap] == 0).all()
    present_outside = np.delete(r.n_present, np.arange(6, 12), axis=0)
    assert (present_outside == [4, 2]).all()
    assert (np.delete(r.n_expected, np.arange(6, 12)) == 4).all()
    # obsvalue_a fully valid, obsvalue_b valid in half the rows
    np.testing.assert_allclose(r.completeness, [1.0, 0.5])
    assert r.native_frequency_seconds is None
    assert r.delta_stats_seconds["median"] == 3600.0


def test_analyze_obs_time_restriction(obs_zarr: pathlib.Path):
    r = cda.analyze_obs(
        obs_zarr,
        "SYNOP",
        bin_seconds=3600,
        start=np.datetime64("2020-01-02T00:00:00"),
        end=None,
    )
    assert r.time_min == np.datetime64("2020-01-02T00:00:00")
    assert r.total_rows == 24 * 4


def test_analyze_regular_dates_missing_and_bin_coarsening():
    dates = np.datetime64("2020-01-01T00:00:00") + np.arange(8) * np.timedelta64(6, "h")
    presence = np.ones((8, 3), dtype=bool)
    presence[2, :] = False  # a missing date
    presence[:, 2] = False  # a dead channel
    r = cda.analyze_regular_dates(
        dates,
        presence,
        ["a", "b", "c"],
        stream="ERA5",
        dataset="era5",
        path="era5.zarr",
        reader_type="anemoi",
        bin_seconds=900,
        native_frequency_seconds=6 * 3600,
        completeness=presence.mean(axis=0),
        analysis_mode="metadata",
    )
    # bin size is widened to the native frequency (multiple of the requested bin)
    assert r.bin_seconds == 6 * 3600
    assert len(r.time_bins) == 8
    assert (r.n_expected == 1).all()
    assert r.n_present[2].tolist() == [0, 0, 0]
    assert (r.n_present[[0, 1, 3, 4, 5, 6, 7]][:, :2] == 1).all()
    assert (r.n_present[:, 2] == 0).all()
    assert r.delta_stats_seconds["median"] == 6 * 3600.0


def test_store_roundtrip_and_plots(obs_zarr: pathlib.Path, tmp_path: pathlib.Path):
    obs = cda.analyze_obs(obs_zarr, "SYNOP", bin_seconds=3600, start=None, end=None)
    dates = np.datetime64("2020-01-01T00:00:00") + np.arange(8) * np.timedelta64(6, "h")
    regular = cda.analyze_regular_dates(
        dates,
        np.ones((8, 2), dtype=bool),
        ["t2m", "u10"],
        stream="ERA5",
        dataset="era5",
        path="era5.zarr",
        reader_type="anemoi",
        bin_seconds=3600,
        native_frequency_seconds=6 * 3600,
        completeness=np.ones(2),
        analysis_mode="metadata",
    )

    store = tmp_path / "availability.zarr"
    cda.write_store(store, [obs, regular], [], "test_config", 3600)

    manifest = pda.load_manifest(store)
    assert manifest["label"] == "test_config"
    assert [g["stream"] for g in manifest["groups"]] == ["SYNOP", "ERA5"]

    ds = pda.open_group(store, manifest["groups"][0]["group"])
    assert ds["n_present"].shape == (48, 2)
    np.testing.assert_allclose(ds["completeness"].values, [1.0, 0.5])

    overview = tmp_path / "overview.html"
    pda.plot_overview(store, overview, "coverage", 100, None)
    assert overview.exists() and overview.stat().st_size > 0
    overview_text = overview.read_text()
    assert "plotly" in overview_text.lower()
    assert "Value completeness: %{z:.2f}%" in overview_text

    per_channel = tmp_path / "per_channel.html"
    pda.plot_per_channel(store, "SYNOP/obs", per_channel, "coverage", 100, None)
    assert per_channel.exists() and per_channel.stat().st_size > 0
    assert "Value completeness: %{z:.2f}%" in per_channel.read_text()


def test_summary_html_and_csv_exports(tmp_path: pathlib.Path):
    summary = {
        "label": "test_config",
        "datasets": [
            {
                "stream": "SYNOP",
                "dataset": "a-very-long-observation-dataset-name-for-hovering",
                "reader_type": "obs",
                "analysis_mode": "full",
                "time_min": "2020-01-01T00:00:00",
                "time_max": "2020-01-02T00:00:00",
                "n_channels": 2,
                "total_rows": 100,
                "native_frequency_seconds": None,
                "timestamp_spacing_seconds": {"median": 3600.0},
                "completeness_min": 0.5,
                "completeness_median": 0.75,
                "completeness_max": 1.0,
                "completeness_per_channel": {"a": 1.0, "b": 0.5},
            }
        ],
    }
    summary_path = tmp_path / "availability.summary.json"
    summary_path.write_text(json.dumps(summary))
    html_path = tmp_path / "summary.html"
    csv_path = tmp_path / "summary.csv"

    pda.write_summary_exports(summary_path, html_path, csv_path)

    html_text = html_path.read_text()
    assert "Dataset availability summary — test_config" in html_text
    assert 'title="a-very-long-observation-dataset-name-for-hovering"' in html_text
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "a-very-long-observation-dataset-name-for-hovering"
    assert float(rows[0]["completeness_median_pct"]) == 75.0
    assert "fraction_bins_with_data_pct" not in rows[0]


def test_aggregate_to_display_coverage_vs_raw():
    dates = np.datetime64("2020-01-01T00:00:00") + np.arange(8) * np.timedelta64(6, "h")
    r = cda.analyze_regular_dates(
        dates,
        np.ones((8, 1), dtype=bool),
        ["t2m"],
        stream="ERA5",
        dataset="era5",
        path="era5.zarr",
        reader_type="anemoi",
        bin_seconds=900,
        native_frequency_seconds=6 * 3600,
        completeness=np.ones(1),
        analysis_mode="metadata",
    )
    ds = cda.result_to_xarray(r)
    edges = pda.display_edges(dates[0], dates[-1] + np.timedelta64(6, "h"), 2)
    coverage = pda.aggregate_to_display(ds, edges, "coverage")
    # a complete regular dataset has 100% availability in coverage mode
    np.testing.assert_allclose(coverage[np.isfinite(coverage)], 1.0)


def test_aggregate_to_display_obs_value_completeness(obs_zarr: pathlib.Path):
    r = cda.analyze_obs(obs_zarr, "SYNOP", bin_seconds=3600, start=None, end=None)
    ds = cda.result_to_xarray(r)
    # hourly display columns aligned with the fine bins
    edges = pda.display_edges(r.time_bins[0], r.time_bins[-1] + np.timedelta64(1, "h"), 48)

    coverage = pda.aggregate_to_display(ds, edges, "coverage")
    # per occupied hour: obsvalue_a 4/4 + obsvalue_b 2/4 of 2*4 slots = 75%; the gap is white
    np.testing.assert_allclose(coverage[:6], 0.75)
    assert np.isnan(coverage[6:12]).all()
    np.testing.assert_allclose(coverage[12:48], 0.75)

    raw = pda.aggregate_to_display(ds, edges, "raw")
    # duty cycle: both channels have values in every occupied bin, none in the gap
    np.testing.assert_allclose(raw[:6], 1.0)
    np.testing.assert_allclose(raw[6:12], 0.0)
    np.testing.assert_allclose(raw[12:48], 1.0)


def test_load_manifest_rejects_old_store_format(tmp_path: pathlib.Path):
    store = tmp_path / "old.zarr"
    root = zarr.open_group(str(store), mode="w")
    root.attrs["wg_availability"] = {"label": "x", "bin_seconds": 900, "groups": [], "skipped": []}
    with pytest.raises(ValueError, match="older"):
        pda.load_manifest(store)


def test_default_output_path():
    base = pathlib.Path("results/dataset_availability")
    assert cda.default_output_path("my_config", None, None) == base / "my_config.zarr"
    assert (
        cda.default_output_path(
            "my_config",
            np.datetime64("2023-01-01T00:00:00"),
            np.datetime64("2023-02-01T06:30:00"),
        )
        == base / "my_config_2023-01-01_2023-02-01T063000.zarr"
    )
    assert (
        cda.default_output_path("my_config", np.datetime64("2023-01-01"), None)
        == base / "my_config_2023-01-01.zarr"
    )


def test_resolve_dataset_path(tmp_path: pathlib.Path):
    (tmp_path / "data.zarr").mkdir()
    assert cda.resolve_dataset_path("data.zarr", [str(tmp_path)]) == tmp_path / "data.zarr"
    with pytest.raises(FileNotFoundError):
        cda.resolve_dataset_path("nope.zarr", [str(tmp_path)])
