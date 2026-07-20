#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plot training metrics from ``{run_id}_train_metrics.json`` files against num_samples.

Reads the JSON-lines metrics files written during training for one or more runs
(``<results-dir>/<run-id>/<run-id>_train_metrics.json``) and produces one figure per
requested metric, with one curve per run, plotted over the logged ``num_samples``.
Metric names may be exact keys or fnmatch patterns (e.g. ``"LossPhysical.ERA5.mse.z_500.*"``).

Alongside the metric figures, a bar plot ``num_samples.<stage>.png`` compares the total
``num_samples`` each run reached, read from its last logged record.

Example usage:

    uv run python packages/science/plot_train_metrics.py \\
        --run-ids run1 run2 run3 \\
        --metrics loss_avg_mean LossPhysical.ERA5.mse.avg "LossPhysical.ERA5.mse.z_500.*" \\
        --results-dir results

Plots are written to ``<results-dir>/<run-id>/`` for a single run and to
``<results-dir>/metric_plots/`` when comparing several runs (override with --out-dir).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Colorblind-safe categorical palette (light surface), assigned to runs in fixed order.
_SERIES_COLORS = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]
# Beyond 8 runs the palette repeats; linestyle then disambiguates identity.
_SERIES_LINESTYLES = ["-", "--", ":", "-."]


def read_metrics(metrics_path: Path, stage: str) -> list[dict[str, float | int]]:
    """Read the JSON-lines metrics file, keeping only records of the given stage."""
    records: list[dict[str, float | int]] = []
    with open(metrics_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", line_num, metrics_path)
                continue
            if rec.get("stage") == stage:
                records.append({k: v for k, v in rec.items() if isinstance(v, int | float)})
    return records


def resolve_metric_names(
    patterns: list[str], runs: dict[str, list[dict[str, float | int]]]
) -> list[str]:
    """Expand fnmatch patterns against the union of metric keys across all runs."""
    all_keys: set[str] = set()
    for records in runs.values():
        for rec in records:
            all_keys.update(rec.keys())
    all_keys.discard("num_samples")

    resolved: list[str] = []
    for pattern in patterns:
        matches = sorted(fnmatch.filter(all_keys, pattern))
        if not matches:
            logger.warning("Metric pattern %r matched no keys in any run", pattern)
        for m in matches:
            if m not in resolved:
                resolved.append(m)
    return resolved


def extract_series(
    records: list[dict[str, float | int]], metric: str
) -> tuple[list[float], list[float]]:
    """Return (num_samples, values) for records that contain both keys, in file order."""
    xs: list[float] = []
    ys: list[float] = []
    for rec in records:
        if "num_samples" in rec and metric in rec:
            xs.append(float(rec["num_samples"]))
            ys.append(float(rec[metric]))
    return xs, ys


def plot_metric(
    metric: str,
    runs: dict[str, list[dict[str, float | int]]],
    out_dir: Path,
    stage: str,
    logy: bool,
) -> Path | None:
    """Plot one metric for all runs and save the figure; returns the path or None if no data."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for idx, (run_id, records) in enumerate(runs.items()):
        xs, ys = extract_series(records, metric)
        if not xs:
            logger.warning("Run %s has no values for metric %s", run_id, metric)
            continue
        ax.plot(
            xs,
            ys,
            label=run_id,
            color=_SERIES_COLORS[idx % len(_SERIES_COLORS)],
            linestyle=_SERIES_LINESTYLES[(idx // len(_SERIES_COLORS)) % len(_SERIES_LINESTYLES)],
            linewidth=1.8,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("num_samples")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} ({stage})")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if len(runs) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", metric)
    out_path = out_dir / f"{safe_name}.{stage}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_num_samples(
    runs: dict[str, list[dict[str, float | int]]],
    out_dir: Path,
    stage: str,
) -> Path | None:
    """Bar-plot the total num_samples per run, taken from each run's last logged record."""
    run_ids: list[str] = []
    totals: list[float] = []
    colors: list[str] = []
    for idx, (run_id, records) in enumerate(runs.items()):
        last = next((rec for rec in reversed(records) if "num_samples" in rec), None)
        if last is None:
            logger.warning("Run %s has no num_samples record", run_id)
            continue
        run_ids.append(run_id)
        totals.append(float(last["num_samples"]))
        # Keep bar colors aligned with the line plots, which index the palette by run order.
        colors.append(_SERIES_COLORS[idx % len(_SERIES_COLORS)])
    if not run_ids:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(run_ids, totals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)

    ax.set_ylabel("num_samples")
    ax.set_title(f"total num_samples ({stage})")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelrotation=45 if len(run_ids) > 3 else 0)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right" if len(run_ids) > 3 else "center")
    fig.tight_layout()

    out_path = out_dir / f"num_samples.{stage}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics of one or more runs against num_samples."
    )
    parser.add_argument("--run-ids", required=True, nargs="+", help="Run ids to plot")
    parser.add_argument(
        "--metrics",
        required=True,
        nargs="+",
        help="Metric names or fnmatch patterns (quote patterns to avoid shell globbing)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing <run-id>/<run-id>_train_metrics.json (default: results)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for the plots (default: the run's results subdirectory for a "
        "single run, <results-dir>/metric_plots for several runs)",
    )
    parser.add_argument(
        "--stage",
        default="train",
        help="Stage to plot, matched against the 'stage' field (default: train)",
    )
    parser.add_argument("--logy", action="store_true", help="Use a logarithmic y-axis")
    args = parser.parse_args()

    runs: dict[str, list[dict[str, float | int]]] = {}
    for run_id in args.run_ids:
        metrics_path = args.results_dir / run_id / f"{run_id}_train_metrics.json"
        if not metrics_path.exists():
            logger.warning("Metrics file not found, skipping run %s: %s", run_id, metrics_path)
            continue
        records = read_metrics(metrics_path, args.stage)
        if not records:
            logger.warning("No %r records in %s, skipping run", args.stage, metrics_path)
            continue
        runs[run_id] = records
        logger.info("Read %d %s records for run %s", len(records), args.stage, run_id)

    if not runs:
        raise SystemExit("No metrics could be read for any of the requested runs.")

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif len(runs) == 1:
        out_dir = args.results_dir / next(iter(runs))
    else:
        out_dir = args.results_dir / "metric_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_names = resolve_metric_names(args.metrics, runs)
    if not metric_names:
        raise SystemExit("No requested metric matched any key in the metrics files.")

    written = 0
    for metric in metric_names:
        out_path = plot_metric(metric, runs, out_dir, args.stage, args.logy)
        if out_path is not None:
            logger.info("Wrote %s", out_path)
            written += 1

    samples_path = plot_num_samples(runs, out_dir, args.stage)
    if samples_path is not None:
        logger.info("Wrote %s", samples_path)
        written += 1

    logger.info("Wrote %d plot(s) to %s", written, out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
