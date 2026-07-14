#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plot dataset availability heatmaps from a stats store built by
``compute_dataset_availability.py``.

Main figure: one heatmap row per dataset, grouped by stream, over a common time axis.
The cell value is the percentage of channels that have at least one value in the time
interval. Because the fine bins (5/15 min) of a multi-year archive are far more numerous
than any figure can show, the time axis is aggregated to ``--display-bins`` columns with
an explicit mean — never by implicit image downsampling. ``--zoom`` renders a subrange
at (up to) native bin resolution; ``--per-channel`` renders a channel-by-time heatmap
for one dataset.

Aggregation modes:

- ``coverage`` (default): the mean is taken only over fine bins where data is *expected*
  (native timestamps for regular datasets, the whole data span for irregular obs).
  A complete 6-hourly dataset shows 100%, gaps show as dips.
- ``raw``: plain duty cycle over all fine bins; a complete 6-hourly dataset at 15-min
  bins shows ~4%. Honest about absolute temporal density, hard to compare across
  frequencies.

Example (run from the repository root):

    uv run python packages/science/plot_dataset_availability.py \\
        --stats results/dataset_availability/era5_georing_avhrr.zarr \\
        [--zoom 2021-01-01 2021-01-08] [--per-channel STREAM/DATASET]
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Sequential single-hue ramp (blue, steps 100->700) and neutral tones; lightest step
# means "0% available", the neutral gray marks bins outside a dataset's time span.
_SEQ_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# Bins where no data is expected blend into the page so absence reads as "nothing
# there"; 0% availability keeps the lightest ramp step and stays visible against it.
_NO_DATA_COLOR = "white"
_TEXT_PRIMARY = "#3a3a37"
_TEXT_MUTED = "#6f6e6a"


def availability_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list("availability", _SEQ_RAMP)
    cmap.set_bad(_NO_DATA_COLOR)
    return cmap


def _shorten(name: str, max_chars: int) -> str:
    """Truncate from the left: the tail of dataset filenames carries the distinguishing part."""
    return name if len(name) <= max_chars else "…" + name[-(max_chars - 1) :]


# ---------------------------------------------------------------------------
# Loading & aggregation
# ---------------------------------------------------------------------------


def load_manifest(stats_path: pathlib.Path) -> dict:
    root = zarr.open_group(str(stats_path), mode="r")
    manifest = root.attrs.get("wg_availability")
    if manifest is None:
        raise ValueError(f"{stats_path} is not a dataset-availability store.")
    return dict(manifest)


def open_group(stats_path: pathlib.Path, group: str) -> xr.Dataset:
    return xr.open_zarr(stats_path, group=group, consolidated=False, chunks=None)


def aggregate_to_display(
    ds: xr.Dataset,
    edges: NDArray[np.datetime64],
    mode: str,
) -> NDArray[np.float64]:
    """Aggregate one dataset's fine-bin channel availability onto display bin edges.

    Returns, per display bin, the mean fraction of channels with at least one value;
    NaN where the dataset has no expected data (rendered in the no-data color).
    """
    time = ds["time"].values
    frac_present = (ds["n_present"].values > 0).mean(axis=1)  # [t] fraction of channels
    expected = ds["n_expected"].values > 0
    if mode == "raw":
        expected = (time >= time[0]) & (time <= time[-1])

    idx = np.searchsorted(edges, time, side="right") - 1
    valid = (idx >= 0) & (idx < len(edges) - 1) & expected
    n_display = len(edges) - 1
    sums = np.bincount(idx[valid], weights=frac_present[valid], minlength=n_display)
    counts = np.bincount(idx[valid], minlength=n_display)
    with np.errstate(invalid="ignore"):
        values = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return values


def display_edges(
    t_min: np.datetime64, t_max: np.datetime64, n_bins: int, min_step_s: int = 1
) -> NDArray[np.datetime64]:
    """Display bin edges: ~n_bins columns, but never finer than the fine-bin size
    (columns smaller than a fine bin would render spurious no-data stripes)."""
    span_s = max(1, int((t_max - t_min) / np.timedelta64(1, "s")))
    step_s = max(1, span_s // n_bins, min_step_s)
    n = span_s // step_s + 1
    return t_min + np.arange(n + 1) * np.timedelta64(step_s, "s")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=_TEXT_MUTED, labelcolor=_TEXT_PRIMARY, length=3)


def _format_time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _add_colorbar(fig: plt.Figure, mappable, label: str) -> None:
    cbar = fig.colorbar(
        mappable, ax=fig.axes, orientation="horizontal", fraction=0.05, pad=0.10, aspect=45
    )
    cbar.set_label(label, color=_TEXT_PRIMARY, fontsize=9)
    cbar.ax.tick_params(colors=_TEXT_MUTED, labelcolor=_TEXT_PRIMARY, length=3)
    cbar.outline.set_visible(False)


def plot_overview(
    stats_path: pathlib.Path,
    out_path: pathlib.Path,
    mode: str,
    n_display_bins: int,
    zoom: tuple[np.datetime64, np.datetime64] | None,
    dpi: int,
) -> None:
    """Heatmap of channel availability over time: one row per dataset, grouped by stream."""
    manifest = load_manifest(stats_path)
    groups = manifest["groups"]
    datasets = [open_group(stats_path, g["group"]) for g in groups]

    t_min = min(ds["time"].values[0] for ds in datasets)
    t_max = max(ds["time"].values[-1] for ds in datasets)
    if zoom is not None:
        t_min, t_max = zoom
    min_step = min(int(ds.attrs["bin_seconds"]) for ds in datasets)
    edges = display_edges(t_min, t_max, n_display_bins, min_step)

    matrix = np.full((len(datasets), len(edges) - 1), np.nan)
    row_labels, stream_of_row = [], []
    for i, (g, ds) in enumerate(zip(groups, datasets, strict=True)):
        matrix[i] = aggregate_to_display(ds, edges, mode)
        row_labels.append(_shorten(g["dataset"], 30))
        stream_of_row.append(g["stream"])

    n_rows = len(datasets)
    fig_h = max(2.8, 0.42 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_h), dpi=dpi)

    x = mdates.date2num(edges.astype("datetime64[us]").astype("O"))
    y = np.arange(n_rows + 1)
    mesh = ax.pcolormesh(
        x, y, matrix[::-1] * 100.0, cmap=availability_cmap(), vmin=0.0, vmax=100.0, rasterized=True
    )

    # 2px-equivalent surface gaps: thin white separators between rows, thick between streams
    for i in range(1, n_rows):
        below_stream = stream_of_row[::-1][i - 1]
        above_stream = stream_of_row[::-1][i]
        lw = 2.5 if below_stream != above_stream else 0.8
        ax.axhline(i, color="white", lw=lw)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels[::-1], fontsize=8, color=_TEXT_PRIMARY)
    ax.set_ylim(0, n_rows)
    ax.set_xlim(x[0], x[-1])

    # stream group labels in their own column, left of the dataset labels
    rev_streams = stream_of_row[::-1]
    i = 0
    while i < n_rows:
        j = i
        while j + 1 < n_rows and rev_streams[j + 1] == rev_streams[i]:
            j += 1
        ax.annotate(
            _shorten(rev_streams[i], 24),
            xy=(-0.52, (i + j + 1) / 2 / n_rows),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=_TEXT_PRIMARY,
            annotation_clip=False,
        )
        i = j + 1

    _format_time_axis(ax)
    _style_axes(ax)

    bin_min = manifest["bin_seconds"] // 60
    step_s = int((edges[1] - edges[0]) / np.timedelta64(1, "s"))
    step_str = f"{step_s / 3600:.1f}h" if step_s >= 3600 else f"{step_s / 60:.0f}min"
    ax.set_title(
        f"Dataset availability — {manifest['label']}",
        fontsize=12,
        color=_TEXT_PRIMARY,
        loc="left",
        pad=14,
    )
    ax.text(
        0.0,
        1.02,
        f"computed on {bin_min}-min bins, displayed at {step_str}/column, mode={mode}; "
        f"white = no data expected, lightest blue = expected but missing",
        transform=ax.transAxes,
        fontsize=8,
        color=_TEXT_MUTED,
    )
    fig.subplots_adjust(left=0.35, right=0.98, top=0.86, bottom=0.16)
    _add_colorbar(fig, mesh, "% of channels with at least one value in the interval")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def plot_per_channel(
    stats_path: pathlib.Path,
    target: str,
    out_path: pathlib.Path,
    mode: str,
    n_display_bins: int,
    zoom: tuple[np.datetime64, np.datetime64] | None,
    dpi: int,
) -> None:
    """Channel-by-time availability heatmap for a single dataset (``STREAM/DATASET``)."""
    manifest = load_manifest(stats_path)
    match = [g for g in manifest["groups"] if f"{g['stream']}/{g['dataset']}" == target]
    if not match:
        options = ", ".join(f"{g['stream']}/{g['dataset']}" for g in manifest["groups"])
        raise SystemExit(f"Unknown dataset '{target}'. Available: {options}")
    ds = open_group(stats_path, match[0]["group"])

    time = ds["time"].values
    t_min, t_max = (time[0], time[-1]) if zoom is None else zoom
    edges = display_edges(t_min, t_max, n_display_bins, int(ds.attrs["bin_seconds"]))
    channels = [str(c) for c in ds["channel"].values]
    completeness = ds["completeness"].values

    expected = ds["n_expected"].values > 0
    if mode == "raw":
        expected = np.ones_like(expected)
    idx = np.searchsorted(edges, time, side="right") - 1
    valid = (idx >= 0) & (idx < len(edges) - 1) & expected
    n_display = len(edges) - 1
    counts = np.bincount(idx[valid], minlength=n_display)

    present = ds["n_present"].values > 0  # [t, c]
    matrix = np.full((len(channels), n_display), np.nan)
    for c in range(len(channels)):
        sums = np.bincount(idx[valid], weights=present[valid, c], minlength=n_display)
        matrix[c] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    fig_h = max(3.0, 0.28 * len(channels) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_h), dpi=dpi)
    x = mdates.date2num(edges.astype("datetime64[us]").astype("O"))
    y = np.arange(len(channels) + 1)
    mesh = ax.pcolormesh(
        x, y, matrix[::-1] * 100.0, cmap=availability_cmap(), vmin=0.0, vmax=100.0, rasterized=True
    )
    labels = [
        f"{_shorten(c, 28)}  ({100 * v:.0f}%)" for c, v in zip(channels, completeness, strict=True)
    ]
    ax.set_yticks(np.arange(len(channels)) + 0.5)
    ax.set_yticklabels(labels[::-1], fontsize=7, color=_TEXT_PRIMARY)
    ax.set_ylim(0, len(channels))
    ax.set_xlim(x[0], x[-1])
    for i in range(1, len(channels)):
        ax.axhline(i, color="white", lw=0.6)
    _format_time_axis(ax)
    _style_axes(ax)
    ax.set_title(
        f"Per-channel availability — {target}",
        fontsize=12,
        color=_TEXT_PRIMARY,
        loc="left",
        pad=14,
    )
    ax.text(
        0.0,
        1.02,
        f"mode={mode}; y-labels show overall completeness (non-NaN fraction); "
        f"white = no data expected",
        transform=ax.transAxes,
        fontsize=8,
        color=_TEXT_MUTED,
    )
    fig.subplots_adjust(left=0.30, right=0.98, top=0.88, bottom=0.14)
    _add_colorbar(fig, mesh, "% of intervals with a value for the channel")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot dataset availability heatmaps from an availability stats store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stats", required=True, help="Path to the availability zarr store.")
    parser.add_argument("--out", default=None, help="Output image path (default: plots/...).")
    parser.add_argument(
        "--mode",
        choices=["coverage", "raw"],
        default="coverage",
        help="coverage: availability relative to expected timestamps; raw: plain duty cycle.",
    )
    parser.add_argument(
        "--display-bins",
        type=int,
        default=1400,
        help="Number of time columns in the rendered heatmap.",
    )
    parser.add_argument(
        "--zoom",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Restrict the time axis (ISO datetimes), e.g. --zoom 2021-01-01 2021-01-08.",
    )
    parser.add_argument(
        "--per-channel",
        default=None,
        metavar="STREAM/DATASET",
        help="Also/only plot a channel-by-time heatmap for one dataset.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    stats_path = pathlib.Path(args.stats)
    label = load_manifest(stats_path)["label"]
    zoom = None
    if args.zoom:
        zoom = (
            np.datetime64(args.zoom[0]).astype("datetime64[s]"),
            np.datetime64(args.zoom[1]).astype("datetime64[s]"),
        )

    if args.per_channel:
        default_out = f"plots/dataset_availability/{label}_{args.per_channel.replace('/', '_')}.png"
        plot_per_channel(
            stats_path,
            args.per_channel,
            pathlib.Path(args.out or default_out),
            args.mode,
            args.display_bins,
            zoom,
            args.dpi,
        )
    else:
        default_out = f"plots/dataset_availability/{label}.png"
        plot_overview(
            stats_path,
            pathlib.Path(args.out or default_out),
            args.mode,
            args.display_bins,
            zoom,
            args.dpi,
        )


if __name__ == "__main__":
    main()
