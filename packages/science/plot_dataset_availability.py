#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Create interactive reports from ``compute_dataset_availability.py`` output.

The main HTML report contains one heatmap row per dataset over a common time axis.
The cell value is the percentage of channels that have at least one value in the time
interval. Because the fine bins (5/15 min) of a multi-year archive are far more numerous
than a browser can usefully show, the time axis is aggregated to ``--display-bins``
columns with an explicit mean — never by implicit image downsampling. Hovering a cell
shows its exact percentage and the full dataset/channel name. ``--zoom`` renders a
subrange at (up to) native bin resolution; ``--per-channel`` renders a
channel-by-time heatmap for one dataset.

When the companion ``*.summary.json`` exists, the CLI also writes dataset-level summary
tables as HTML and CSV. The JSON remains the complete source, including per-channel
completeness.

Aggregation modes:

- ``coverage`` (default): the mean is taken only over fine bins where data is expected.
  A complete 6-hourly dataset therefore shows 100%, while gaps show as dips.
- ``raw``: plain duty cycle over all fine bins. This is honest about absolute temporal
  density but makes datasets with different frequencies harder to compare.

Example (run from the repository root):

    uv run python packages/science/plot_dataset_availability.py \\
        --stats results/dataset_availability/era5_georing_avhrr.zarr \\
        [--zoom 2021-01-01 2021-01-08] [--per-channel STREAM/DATASET]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import pathlib
from typing import Any

import numpy as np
import plotly.graph_objects as go
import xarray as xr
import zarr
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Sequential single-hue ramp (blue, steps 100->700) and neutral tones; lightest step
# means "0% available", while white marks bins outside a dataset's time span.
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
_NO_DATA_COLOR = "white"
_TEXT_PRIMARY = "#3a3a37"
_TEXT_MUTED = "#6f6e6a"


def availability_colorscale() -> list[list[Any]]:
    """Plotly colorscale equivalent of the sequential availability ramp."""
    return [[i / (len(_SEQ_RAMP) - 1), color] for i, color in enumerate(_SEQ_RAMP)]


def _shorten(name: str, max_chars: int) -> str:
    """Truncate from the left: filename tails usually carry the distinguishing part."""
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
    NaN where the dataset has no expected data (rendered as white).
    """
    time = ds["time"].values
    frac_present = (ds["n_present"].values > 0).mean(axis=1)
    expected = ds["n_expected"].values > 0
    if mode == "raw":
        expected = (time >= time[0]) & (time <= time[-1])

    idx = np.searchsorted(edges, time, side="right") - 1
    valid = (idx >= 0) & (idx < len(edges) - 1) & expected
    n_display = len(edges) - 1
    sums = np.bincount(idx[valid], weights=frac_present[valid], minlength=n_display)
    counts = np.bincount(idx[valid], minlength=n_display)
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def display_edges(
    t_min: np.datetime64, t_max: np.datetime64, n_bins: int, min_step_s: int = 1
) -> NDArray[np.datetime64]:
    """Make ~n_bins display columns, never finer than the source bin size."""
    span_s = max(1, int((t_max - t_min) / np.timedelta64(1, "s")))
    step_s = max(1, span_s // n_bins, min_step_s)
    n = span_s // step_s + 1
    return t_min + np.arange(n + 1) * np.timedelta64(step_s, "s")


# ---------------------------------------------------------------------------
# Interactive figures
# ---------------------------------------------------------------------------


def _write_figure(fig: go.Figure, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    logger.info(f"Wrote {out_path}")


def _base_layout(height: int) -> dict[str, Any]:
    return {
        "template": "plotly_white",
        "height": height,
        "paper_bgcolor": _NO_DATA_COLOR,
        "plot_bgcolor": _NO_DATA_COLOR,
        "font": {"color": _TEXT_PRIMARY, "size": 12},
        "margin": {"l": 350, "r": 50, "t": 105, "b": 120},
        "hoverlabel": {"bgcolor": "white", "font": {"color": _TEXT_PRIMARY}},
        "coloraxis": {
            "colorscale": availability_colorscale(),
            "cmin": 0.0,
            "cmax": 100.0,
            "colorbar": {
                "title": {"text": "Availability (%)", "side": "right"},
                "ticksuffix": "%",
                "len": 0.75,
            },
        },
    }


def plot_overview(
    stats_path: pathlib.Path,
    out_path: pathlib.Path,
    mode: str,
    n_display_bins: int,
    zoom: tuple[np.datetime64, np.datetime64] | None,
) -> None:
    """Interactive availability heatmap with one row per dataset."""
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
    row_ids, row_labels = [], []
    for i, (group, ds) in enumerate(zip(groups, datasets, strict=True)):
        matrix[i] = aggregate_to_display(ds, edges, mode)
        row_ids.append(f"{group['stream']}/{group['dataset']}")
        row_labels.append(f"{_shorten(group['stream'], 20)} · {_shorten(group['dataset'], 38)}")

    bin_min = manifest["bin_seconds"] // 60
    step_s = int((edges[1] - edges[0]) / np.timedelta64(1, "s"))
    step_str = f"{step_s / 3600:.1f}h" if step_s >= 3600 else f"{step_s / 60:.0f}min"
    fig = go.Figure(
        go.Heatmap(
            x=edges[:-1],
            y=row_ids,
            z=matrix * 100.0,
            coloraxis="coloraxis",
            hoverongaps=False,
            xgap=0,
            ygap=2,
            hovertemplate=(
                "<b>%{y}</b><br>Display interval starts: %{x|%Y-%m-%d %H:%M}<br>"
                "Available channels: %{z:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_base_layout(max(440, 34 * len(datasets) + 250)),
        title={
            "text": (
                f"Dataset availability — {manifest['label']}"
                f"<br><sup>Computed on {bin_min}-min bins; {step_str}/column; mode={mode}. "
                "White = no data expected. Hover for full names and exact percentages.</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis={"title": "Time", "showgrid": False, "rangeslider": {"visible": True}},
        yaxis={
            "title": None,
            "tickmode": "array",
            "tickvals": row_ids,
            "ticktext": row_labels,
            "autorange": "reversed",
            "showgrid": False,
            "automargin": True,
        },
    )
    _write_figure(fig, out_path)


def plot_per_channel(
    stats_path: pathlib.Path,
    target: str,
    out_path: pathlib.Path,
    mode: str,
    n_display_bins: int,
    zoom: tuple[np.datetime64, np.datetime64] | None,
) -> None:
    """Interactive channel-by-time heatmap for one ``STREAM/DATASET``."""
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

    present = ds["n_present"].values > 0
    matrix = np.full((len(channels), n_display), np.nan)
    for channel_idx in range(len(channels)):
        sums = np.bincount(idx[valid], weights=present[valid, channel_idx], minlength=n_display)
        matrix[channel_idx] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    labels = [
        f"{_shorten(channel, 42)} · overall {100 * value:.1f}%"
        for channel, value in zip(channels, completeness, strict=True)
    ]
    fig = go.Figure(
        go.Heatmap(
            x=edges[:-1],
            y=channels,
            z=matrix * 100.0,
            coloraxis="coloraxis",
            hoverongaps=False,
            xgap=0,
            ygap=1,
            hovertemplate=(
                "<b>%{y}</b><br>Display interval starts: %{x|%Y-%m-%d %H:%M}<br>"
                "Availability: %{z:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_base_layout(max(440, 26 * len(channels) + 250)),
        title={
            "text": (
                f"Per-channel availability — {target}"
                f"<br><sup>Mode={mode}. Labels show overall non-NaN completeness; "
                "hover for full channel names and exact interval percentages.</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis={"title": "Time", "showgrid": False, "rangeslider": {"visible": True}},
        yaxis={
            "title": None,
            "tickmode": "array",
            "tickvals": channels,
            "ticktext": labels,
            "autorange": "reversed",
            "showgrid": False,
            "automargin": True,
        },
    )
    _write_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Summary exports
# ---------------------------------------------------------------------------


_SUMMARY_COLUMNS = [
    "stream",
    "dataset",
    "reader_type",
    "analysis_mode",
    "time_min",
    "time_max",
    "total_rows",
    "n_channels",
    "native_frequency_seconds",
    "timestamp_spacing_median_seconds",
    "completeness_min_pct",
    "completeness_median_pct",
    "completeness_max_pct",
]


def summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten dataset-level statistics from the JSON summary for tabular output."""
    rows = []
    for dataset in summary.get("datasets", []):
        spacing = dataset.get("timestamp_spacing_seconds", {})
        rows.append(
            {
                "stream": dataset["stream"],
                "dataset": dataset["dataset"],
                "reader_type": dataset["reader_type"],
                "analysis_mode": dataset["analysis_mode"],
                "time_min": dataset["time_min"],
                "time_max": dataset["time_max"],
                "total_rows": dataset["total_rows"],
                "n_channels": dataset["n_channels"],
                "native_frequency_seconds": dataset.get("native_frequency_seconds"),
                "timestamp_spacing_median_seconds": spacing.get("median"),
                "completeness_min_pct": 100.0 * dataset["completeness_min"],
                "completeness_median_pct": 100.0 * dataset["completeness_median"],
                "completeness_max_pct": 100.0 * dataset["completeness_max"],
            }
        )
    return rows


def _format_duration(seconds: Any) -> str:
    if seconds in (None, "", 0):
        return "—"
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:g} h"
    if seconds >= 60:
        return f"{seconds / 60:g} min"
    return f"{seconds:g} s"


def _summary_html(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        dataset = html.escape(str(row["dataset"]), quote=True)
        short_dataset = html.escape(_shorten(str(row["dataset"]), 50))
        cells = [
            html.escape(str(row["stream"])),
            f'<span title="{dataset}">{short_dataset}</span>',
            html.escape(str(row["reader_type"])),
            html.escape(str(row["analysis_mode"])),
            html.escape(str(row["time_min"])),
            html.escape(str(row["time_max"])),
            f"{int(row['total_rows']):,}",
            str(row["n_channels"]),
            _format_duration(row["native_frequency_seconds"]),
            _format_duration(row["timestamp_spacing_median_seconds"]),
            f"{row['completeness_min_pct']:.2f}%",
            f"{row['completeness_median_pct']:.2f}%",
            f"{row['completeness_max_pct']:.2f}%",
        ]
        table_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    headers = [
        "Stream",
        "Dataset",
        "Type",
        "Analysis",
        "Start",
        "End",
        "Rows",
        "Channels",
        "Native frequency",
        "Median spacing",
        "Completeness min",
        "Completeness median",
        "Completeness max",
    ]
    label = html.escape(str(summary.get("label", "dataset availability")))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset availability summary — {label}</title>
<style>
body {{ color: {_TEXT_PRIMARY}; font: 14px system-ui, sans-serif; margin: 2rem; }}
h1 {{ font-size: 1.35rem; }}
p {{ color: {_TEXT_MUTED}; }}
input {{
  border: 1px solid #bbb; border-radius: 4px; margin: .5rem 0 1rem;
  padding: .5rem; width: 24rem;
}}
.table-wrap {{ max-height: calc(100vh - 11rem); overflow: auto; }}
table {{ border-collapse: collapse; min-width: 1450px; width: 100%; }}
th {{ background: #f2f5f8; position: sticky; top: 0; text-align: left; }}
th, td {{ border-bottom: 1px solid #ddd; padding: .45rem .6rem; white-space: nowrap; }}
tbody tr:hover {{ background: #eef5fd; }}
</style>
</head>
<body>
<h1>Dataset availability summary — {label}</h1>
<p>Dataset-level statistics from the companion JSON.
Hover shortened dataset names for the full identifier.</p>
<input id="filter" type="search" placeholder="Filter stream, dataset, type…"
       aria-label="Filter table">
<div class="table-wrap"><table id="summary"><thead><tr>
{"".join(f"<th>{html.escape(header)}</th>" for header in headers)}
</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
<script>
const input = document.getElementById('filter');
input.addEventListener('input', () => {{
  const query = input.value.toLowerCase();
  for (const row of document.querySelectorAll('#summary tbody tr')) {{
    row.hidden = !row.textContent.toLowerCase().includes(query);
  }}
}});
</script>
</body>
</html>
"""


def write_summary_exports(
    summary_path: pathlib.Path, html_path: pathlib.Path, csv_path: pathlib.Path
) -> None:
    """Write a filterable HTML table and flat CSV from a computation summary JSON."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary_rows(summary)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_summary_html(summary, rows), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {html_path}")
    logger.info(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create interactive reports from an availability stats store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stats", required=True, help="Path to the availability zarr store.")
    parser.add_argument("--out", default=None, help="Output HTML path (default: plots/...).")
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
        help="Number of time columns in the interactive heatmap.",
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
        help="Plot a channel-by-time heatmap for one dataset instead of the overview.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Companion summary JSON (default: STATS with .summary.json suffix).",
    )
    parser.add_argument("--summary-out", default=None, help="Dataset summary HTML path.")
    parser.add_argument("--summary-csv", default=None, help="Dataset summary CSV path.")
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
        default_out = (
            f"plots/dataset_availability/{label}_{args.per_channel.replace('/', '_')}.html"
        )
        plot_out = pathlib.Path(args.out or default_out)
        plot_per_channel(
            stats_path,
            args.per_channel,
            plot_out,
            args.mode,
            args.display_bins,
            zoom,
        )
    else:
        default_out = f"plots/dataset_availability/{label}.html"
        plot_out = pathlib.Path(args.out or default_out)
        plot_overview(stats_path, plot_out, args.mode, args.display_bins, zoom)

    if plot_out.suffix.lower() != ".html":
        logger.warning(f"{plot_out} contains HTML despite its non-.html suffix.")

    summary_path = (
        pathlib.Path(args.summary_json)
        if args.summary_json
        else stats_path.with_suffix(".summary.json")
    )
    if summary_path.exists():
        summary_html = pathlib.Path(args.summary_out or plot_out.parent / f"{label}_summary.html")
        summary_csv = pathlib.Path(args.summary_csv or plot_out.parent / f"{label}_summary.csv")
        write_summary_exports(summary_path, summary_html, summary_csv)
    else:
        logger.warning(f"Summary JSON not found at {summary_path}; skipping summary exports.")


if __name__ == "__main__":
    main()
