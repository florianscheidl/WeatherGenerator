#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compute temporal-availability statistics for the datasets referenced by a run config.

For every dataset (one entry of ``filenames`` per stream) the script measures, on a
regular time-bin grid (default 15 minutes):

- ``n_present[time, channel]`` — number of non-NaN values per channel per bin,
- ``n_expected[time]``          — number of timestamps at which data is expected
                                  (native timestamps for regular datasets, 1 inside the
                                  dataset's time span for irregular observations),
- ``n_reports[time]``           — number of raw rows/timestamps found per bin,

plus per-channel completeness and an estimate of the native temporal resolution.
Results are written to a single zarr store (one group per dataset) consumed by
``plot_dataset_availability.py``, together with a JSON summary.

Datasets are opened directly with zarr/anemoi-datasets rather than through the training
DataReaders: coverage must be measured on the raw files, before the sampling, masking
and point caps of the data pipeline (see agent_docs/data-pipeline.md).

Intended to run on the HPC (dataset paths resolve via the private config). Run from the
repository root:

    uv run python packages/science/compute_dataset_availability.py \\
        --config config/config_era5_georing_avhrr.yml \\
        --output results/dataset_availability/era5_georing_avhrr.zarr

Useful options: ``--bin-minutes 5|15``, ``--start/--end`` to restrict the scanned time
range, ``--streams`` to subset streams, ``--check-nans`` to sample gridded (anemoi)
datasets for NaNs instead of trusting their metadata, ``--row-chunk-stride N`` to
subsample huge observation datasets (reads every N-th chunk; counts become estimates
and sparse datasets may show spurious gaps — keep 1 unless a full scan is too slow).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import sys

import numpy as np
import xarray as xr
import zarr
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EPOCH = np.datetime64(0, "s")
_SECOND = np.timedelta64(1, "s")

# Reader types this script can analyze; anything else is reported as skipped.
_ANEMOI_TYPES = ("anemoi", "anemoi_operan")
_OBS_TYPES = ("obs",)


@dataclasses.dataclass
class DatasetAvailability:
    """Availability statistics of one dataset (one file of one stream)."""

    stream: str
    dataset: str
    path: str
    reader_type: str
    channels: list[str]
    bin_seconds: int
    time_bins: NDArray[np.datetime64]  # [n_bins] bin start times
    n_present: NDArray[np.int32]  # [n_bins, n_channels] non-NaN values per bin
    n_expected: NDArray[np.int64]  # [n_bins] expected timestamps per bin
    n_reports: NDArray[np.int64]  # [n_bins] rows / native timestamps found per bin
    completeness: NDArray[np.float64]  # [n_channels] non-NaN fraction over the scan range
    time_min: np.datetime64
    time_max: np.datetime64
    total_rows: int
    native_frequency_seconds: int | None  # regular cadence; None for irregular obs
    delta_stats_seconds: dict[str, float]  # spacing stats of distinct timestamps
    analysis_mode: str  # "full" | "metadata" | "sampled" | "strided"


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------


def _to_bin_idx(
    datetimes: NDArray[np.datetime64], bin_start: np.datetime64, bin_seconds: int
) -> NDArray[np.int64]:
    """Map datetimes to indices of time bins of size bin_seconds starting at bin_start."""
    offset = (datetimes.astype("datetime64[s]") - bin_start) / _SECOND
    return np.floor(offset / bin_seconds).astype(np.int64)


def _floor_to_bin(t: np.datetime64, bin_seconds: int) -> np.datetime64:
    """Floor a datetime to the global bin grid anchored at the epoch."""
    seconds = int((t.astype("datetime64[s]") - _EPOCH) / _SECOND)
    return _EPOCH + np.timedelta64((seconds // bin_seconds) * bin_seconds, "s")


def accumulate_counts(
    n_present: NDArray[np.int32],
    n_reports: NDArray[np.int64],
    bin_idx: NDArray[np.int64],
    finite: NDArray[np.bool_],
) -> None:
    """Add per-bin non-NaN counts and row counts of one slab of rows.

    Uses a fast segment-sum path when the slab's bin indices are sorted (the usual case,
    observation zarrs are time-ordered) and falls back to np.add.at otherwise.
    """
    if len(bin_idx) == 0:
        return
    n_reports += np.bincount(bin_idx, minlength=len(n_reports)).astype(np.int64)
    if np.all(np.diff(bin_idx) >= 0):
        starts = np.concatenate(([0], np.flatnonzero(np.diff(bin_idx)) + 1))
        seg_bins = bin_idx[starts]
        seg_sums = np.add.reduceat(finite.astype(np.int32), starts, axis=0)
        n_present[seg_bins] += seg_sums
    else:
        np.add.at(n_present, bin_idx, finite.astype(np.int32))


def _delta_stats(deltas_seconds: NDArray[np.float64]) -> dict[str, float]:
    """Summary statistics of the spacing between consecutive distinct timestamps."""
    if len(deltas_seconds) == 0:
        return {}
    return {
        "min": float(np.min(deltas_seconds)),
        "p10": float(np.percentile(deltas_seconds, 10)),
        "median": float(np.median(deltas_seconds)),
        "p90": float(np.percentile(deltas_seconds, 90)),
        "max": float(np.max(deltas_seconds)),
    }


# ---------------------------------------------------------------------------
# Observation-type datasets (flat point tables: data [rows, cols] + dates [rows, 1])
# ---------------------------------------------------------------------------


def analyze_obs(
    path: pathlib.Path,
    stream: str,
    bin_seconds: int,
    start: np.datetime64 | None,
    end: np.datetime64 | None,
    row_chunk_stride: int = 1,
    max_delta_samples: int = 200_000,
) -> DatasetAvailability:
    """Availability of an observation zarr; streams over row chunks, never loads it whole."""
    z = zarr.open(str(path), mode="r")
    data = z["data"]
    dates = z["dates"]
    colnames = list(data.attrs["colnames"])
    channel_idx = np.array(
        [i for i, c in enumerate(colnames) if "obsvalue" in c],
        dtype=np.int64,
    )
    if len(channel_idx) == 0:
        raise ValueError(f"{path}: no 'obsvalue' columns found in colnames; cannot analyze.")
    channels = [colnames[i] for i in channel_idx]

    n_rows_total = data.shape[0]
    t_first = np.asarray(dates[0]).ravel()[0].astype("datetime64[s]")
    t_last = np.asarray(dates[-1]).ravel()[0].astype("datetime64[s]")
    t_lo = max(t_first, start) if start is not None else t_first
    t_hi = min(t_last, end) if end is not None else t_last
    if t_lo > t_hi:
        raise ValueError(f"{path}: requested range [{start}, {end}] outside data span.")

    bin_start = _floor_to_bin(t_lo, bin_seconds)
    n_bins = int((t_hi - bin_start) / _SECOND) // bin_seconds + 1
    n_present = np.zeros((n_bins, len(channels)), dtype=np.int32)
    n_expected = np.ones(n_bins, dtype=np.int64)  # irregular: data may occur in any bin
    n_reports = np.zeros(n_bins, dtype=np.int64)

    chunk_rows = data.chunks[0]
    slab_rows = chunk_rows * max(1, 2_000_000 // max(1, chunk_rows))
    total_rows_seen = 0
    deltas: list[NDArray[np.float64]] = []
    n_deltas = 0

    slab_starts = range(0, n_rows_total, slab_rows * row_chunk_stride)
    for slab_no, r0 in enumerate(slab_starts):
        r1 = min(r0 + slab_rows, n_rows_total)
        dt_slab = np.asarray(dates[r0:r1]).reshape(-1).astype("datetime64[s]")
        in_range = (dt_slab >= t_lo) & (dt_slab <= t_hi)
        if not in_range.any():
            # data is time-ordered: once past the requested range, stop scanning
            if len(dt_slab) > 0 and dt_slab[0] > t_hi:
                break
            continue
        rows = np.flatnonzero(in_range)
        values = np.asarray(data[r0 + rows[0] : r0 + rows[-1] + 1])[:, channel_idx]
        dt_slab = dt_slab[rows[0] : rows[-1] + 1]
        bin_idx = _to_bin_idx(dt_slab, bin_start, bin_seconds)
        accumulate_counts(n_present, n_reports, bin_idx, np.isfinite(values))
        total_rows_seen += len(dt_slab)

        if n_deltas < max_delta_samples:
            uniq = np.unique(dt_slab)
            if len(uniq) > 1:
                d = np.diff(uniq) / _SECOND
                deltas.append(d.astype(np.float64))
                n_deltas += len(d)
        if slab_no % 50 == 0:
            logger.info(f"{stream}/{path.stem}: scanned {r1 / max(1, n_rows_total):.0%} of rows")

    completeness = (
        n_present.sum(axis=0, dtype=np.float64) / total_rows_seen
        if total_rows_seen > 0
        else np.zeros(len(channels))
    )
    time_bins = bin_start + np.arange(n_bins) * np.timedelta64(bin_seconds, "s")

    return DatasetAvailability(
        stream=stream,
        dataset=path.stem,
        path=str(path),
        reader_type="obs",
        channels=channels,
        bin_seconds=bin_seconds,
        time_bins=time_bins,
        n_present=n_present,
        n_expected=n_expected,
        n_reports=n_reports,
        completeness=completeness,
        time_min=t_lo,
        time_max=t_hi,
        total_rows=total_rows_seen,
        native_frequency_seconds=None,
        delta_stats_seconds=_delta_stats(np.concatenate(deltas) if deltas else np.array([])),
        analysis_mode="full" if row_chunk_stride == 1 else "strided",
    )


# ---------------------------------------------------------------------------
# Regular (gridded) datasets on a fixed time grid, e.g. anemoi
# ---------------------------------------------------------------------------


def analyze_regular_dates(
    dates: NDArray[np.datetime64],
    presence: NDArray[np.bool_],
    channels: list[str],
    *,
    stream: str,
    dataset: str,
    path: str,
    reader_type: str,
    bin_seconds: int,
    native_frequency_seconds: int | None,
    completeness: NDArray[np.float64],
    analysis_mode: str,
) -> DatasetAvailability:
    """Bin per-timestamp channel presence of a regular dataset onto the time-bin grid.

    ``presence[t, c]`` states whether channel ``c`` has data at native timestamp ``t``.
    For regular datasets a bin size coarser than the requested one is chosen when the
    native frequency is a multiple of it, which loses no information.
    """
    dates = dates.astype("datetime64[s]")
    if native_frequency_seconds and native_frequency_seconds % bin_seconds == 0:
        bin_seconds = native_frequency_seconds

    t_lo, t_hi = dates[0], dates[-1]
    bin_start = _floor_to_bin(t_lo, bin_seconds)
    n_bins = int((t_hi - bin_start) / _SECOND) // bin_seconds + 1
    n_present = np.zeros((n_bins, len(channels)), dtype=np.int32)
    n_reports = np.zeros(n_bins, dtype=np.int64)
    bin_idx = _to_bin_idx(dates, bin_start, bin_seconds)
    accumulate_counts(n_present, n_reports, bin_idx, presence)
    n_expected = np.bincount(bin_idx, minlength=n_bins).astype(np.int64)

    time_bins = bin_start + np.arange(n_bins) * np.timedelta64(bin_seconds, "s")
    deltas = (np.diff(np.unique(dates)) / _SECOND).astype(np.float64)

    return DatasetAvailability(
        stream=stream,
        dataset=dataset,
        path=path,
        reader_type=reader_type,
        channels=channels,
        bin_seconds=bin_seconds,
        time_bins=time_bins,
        n_present=n_present,
        n_expected=n_expected,
        n_reports=n_reports,
        completeness=completeness,
        time_min=t_lo,
        time_max=t_hi,
        total_rows=len(dates),
        native_frequency_seconds=native_frequency_seconds,
        delta_stats_seconds=_delta_stats(deltas),
        analysis_mode=analysis_mode,
    )


def analyze_anemoi(
    path: pathlib.Path,
    stream: str,
    stream_info: dict,
    bin_seconds: int,
    start: np.datetime64 | None,
    end: np.datetime64 | None,
    check_nans: bool,
    nan_time_samples: int,
) -> DatasetAvailability:
    """Availability of an anemoi dataset.

    By default presence is derived from metadata (the date grid and missing dates) —
    cheap and exact for the time axis, but blind to per-channel NaNs. ``check_nans``
    additionally reads ``nan_time_samples`` evenly spaced timesteps to estimate the
    per-channel non-NaN fraction and to catch fully NaN channels.
    """
    import anemoi.datasets as anemoi_datasets

    ds = anemoi_datasets.open_dataset(str(path))
    dates = np.asarray(ds.dates).astype("datetime64[s]")
    channels = list(ds.variables)

    keep = np.ones(len(dates), dtype=bool)
    if start is not None:
        keep &= dates >= start
    if end is not None:
        keep &= dates <= end
    kept_idx = np.flatnonzero(keep)
    if len(kept_idx) == 0:
        raise ValueError(f"{path}: requested range [{start}, {end}] outside data span.")

    missing = np.array(sorted(getattr(ds, "missing", set()) or set()), dtype=np.int64)
    presence = np.ones((len(kept_idx), len(channels)), dtype=bool)
    pos_of = {int(gi): p for p, gi in enumerate(kept_idx)}
    for gi in missing:
        p = pos_of.get(int(gi))
        if p is not None:
            presence[p, :] = False

    native_freq: int | None = None
    try:
        native_freq = int(np.timedelta64(ds.frequency) / _SECOND)
    except (TypeError, ValueError):
        logger.warning(f"{stream}/{path.stem}: could not parse dataset frequency.")

    analysis_mode = "metadata"
    completeness = presence.mean(axis=0, dtype=np.float64)
    if check_nans:
        analysis_mode = "sampled"
        stride = max(1, len(kept_idx) // max(1, nan_time_samples))
        sample_pos = np.arange(0, len(kept_idx), stride)
        finite_frac = np.zeros((len(sample_pos), len(channels)), dtype=np.float64)
        for j, p in enumerate(sample_pos):
            if not presence[p].any():
                continue  # missing date, nothing to read
            arr = np.asarray(ds[int(kept_idx[p])])  # [channels, ensemble, gridpoints]
            finite_frac[j] = np.isfinite(arr).mean(axis=tuple(range(1, arr.ndim)))
            presence[p] = finite_frac[j] > 0.0
            if j % 50 == 0:
                logger.info(f"{stream}/{path.stem}: NaN check {j}/{len(sample_pos)} samples")
        # value-level completeness estimate from the sampled timesteps
        completeness = finite_frac.mean(axis=0)

    return analyze_regular_dates(
        dates[kept_idx],
        presence,
        channels,
        stream=stream,
        dataset=path.stem,
        path=str(path),
        reader_type=stream_info.get("type", "anemoi"),
        bin_seconds=bin_seconds,
        native_frequency_seconds=native_freq,
        completeness=completeness,
        analysis_mode=analysis_mode,
    )


# ---------------------------------------------------------------------------
# Config / path resolution
# ---------------------------------------------------------------------------


def resolve_dataset_path(fname: str, data_paths: list[str]) -> pathlib.Path:
    """Resolve a stream filename the same way the training sampler does."""
    p = pathlib.Path(fname)
    if p.exists():
        return p
    candidates = [pathlib.Path(dp) / fname for dp in data_paths]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Did not find dataset '{fname}' in any of: {candidates}.")


def load_stream_definitions(
    config_paths: list[str],
    streams_directory: str | None,
    extra_data_paths: list[str],
) -> tuple[dict, list[str], str]:
    """Return (streams dict, data paths, label) from a run config or a streams directory.

    With ``--config``, streams and data paths resolve exactly as for a training run
    (default config + private config + overwrites); requires the private config, i.e.
    an HPC environment. With ``--streams-directory``, only the stream YAMLs are read and
    data paths must come from ``--data-path``.
    """
    from weathergen.common.config import load_merge_configs, load_streams

    if streams_directory is not None:
        streams = load_streams(pathlib.Path(streams_directory))
        return dict(streams), list(extra_data_paths), pathlib.Path(streams_directory).name

    cf = load_merge_configs(None, None, None, None, *config_paths)
    streams = load_streams(pathlib.Path(cf.streams_directory))
    data_paths = list(cf.get("data_paths", [])) + list(extra_data_paths)
    label = pathlib.Path(config_paths[-1]).stem if config_paths else "default_config"
    return dict(streams), data_paths, label


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _group_name(result: DatasetAvailability) -> str:
    safe = result.dataset.replace("/", "_").replace(" ", "_")
    return f"streams/{result.stream}/{safe}"


def result_to_xarray(result: DatasetAvailability) -> xr.Dataset:
    """Convert one dataset's availability statistics to an xarray Dataset."""
    return xr.Dataset(
        {
            "n_present": (("time", "channel"), result.n_present),
            "n_expected": (("time",), result.n_expected),
            "n_reports": (("time",), result.n_reports),
            "completeness": (("channel",), result.completeness),
        },
        coords={"time": result.time_bins, "channel": result.channels},
        attrs={
            "stream": result.stream,
            "dataset": result.dataset,
            "path": result.path,
            "reader_type": result.reader_type,
            "bin_seconds": result.bin_seconds,
            "time_min": str(result.time_min),
            "time_max": str(result.time_max),
            "total_rows": result.total_rows,
            "native_frequency_seconds": result.native_frequency_seconds or 0,
            "delta_stats_seconds": json.dumps(result.delta_stats_seconds),
            "analysis_mode": result.analysis_mode,
        },
    )


def write_store(
    out_path: pathlib.Path,
    results: list[DatasetAvailability],
    skipped: list[dict],
    label: str,
    bin_seconds: int,
) -> None:
    """Write all results into one zarr store with a manifest in the root attributes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in results:
        group = _group_name(r)
        result_to_xarray(r).to_zarr(out_path, group=group, mode="a")
        manifest.append({"stream": r.stream, "dataset": r.dataset, "group": group})
    root = zarr.open_group(str(out_path), mode="a")
    root.attrs.update(
        {
            "wg_availability": {
                "label": label,
                "bin_seconds": bin_seconds,
                "groups": manifest,
                "skipped": skipped,
            }
        }
    )


def summarize(result: DatasetAvailability) -> dict:
    """JSON-serializable per-dataset summary (frequency + completeness)."""
    comp = result.completeness
    return {
        "stream": result.stream,
        "dataset": result.dataset,
        "reader_type": result.reader_type,
        "analysis_mode": result.analysis_mode,
        "time_min": str(result.time_min),
        "time_max": str(result.time_max),
        "n_channels": len(result.channels),
        "total_rows": result.total_rows,
        "native_frequency_seconds": result.native_frequency_seconds,
        "timestamp_spacing_seconds": result.delta_stats_seconds,
        "completeness_min": float(comp.min()) if len(comp) else 0.0,
        "completeness_median": float(np.median(comp)) if len(comp) else 0.0,
        "completeness_max": float(comp.max()) if len(comp) else 0.0,
        "completeness_per_channel": {
            c: float(v) for c, v in zip(result.channels, comp, strict=True)
        },
        "fraction_bins_with_data": float((result.n_reports > 0).mean()),
    }


def _format_summary_table(summaries: list[dict]) -> str:
    header = (
        f"{'stream/dataset':<58s} {'type':<8s} {'span':<24s} "
        f"{'freq':>10s} {'#ch':>4s} {'compl.med':>10s}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        freq = s["native_frequency_seconds"]
        if freq:
            freq_str = f"{freq / 3600:.2g}h" if freq >= 3600 else f"{freq / 60:.0f}min"
        else:
            med = s["timestamp_spacing_seconds"].get("median")
            freq_str = f"~{med:.0f}s" if med else "n/a"
        span = f"{s['time_min'][:10]}..{s['time_max'][:10]}"
        name = f"{s['stream']}/{s['dataset']}"[:58]
        lines.append(
            f"{name:<58s} {s['reader_type']:<8s} {span:<24s} "
            f"{freq_str:>10s} {s['n_channels']:>4d} {s['completeness_median']:>10.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_dt(value: str | None) -> np.datetime64 | None:
    return np.datetime64(value).astype("datetime64[s]") if value else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute temporal availability statistics for the datasets of a run config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        nargs="+",
        help="Run overwrite config(s), merged onto the default config (like a training run).",
    )
    source.add_argument(
        "--streams-directory",
        help="Analyze a stream set directly, bypassing the run-config merge.",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        help="Additional directory to resolve dataset filenames against (repeatable).",
    )
    parser.add_argument("--output", default=None, help="Output zarr store path.")
    parser.add_argument("--bin-minutes", type=int, default=15, help="Time bin size in minutes.")
    parser.add_argument("--start", default=None, help="Restrict scan start (ISO datetime).")
    parser.add_argument("--end", default=None, help="Restrict scan end (ISO datetime).")
    parser.add_argument("--streams", nargs="*", default=None, help="Subset of stream names.")
    parser.add_argument(
        "--check-nans",
        action="store_true",
        help="Sample gridded datasets for per-channel NaNs (slower; otherwise metadata only).",
    )
    parser.add_argument(
        "--nan-time-samples",
        type=int,
        default=200,
        help="Timesteps to sample per gridded dataset with --check-nans.",
    )
    parser.add_argument(
        "--row-chunk-stride",
        type=int,
        default=1,
        help="Read every N-th chunk of observation datasets (estimate; may fake gaps).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    streams, data_paths, label = load_stream_definitions(
        args.config or [], args.streams_directory, args.data_path
    )
    if args.streams:
        unknown = set(args.streams) - set(streams)
        if unknown:
            raise SystemExit(f"Unknown streams {sorted(unknown)}; available: {sorted(streams)}")
        streams = {k: v for k, v in streams.items() if k in args.streams}

    bin_seconds = args.bin_minutes * 60
    start, end = _parse_dt(args.start), _parse_dt(args.end)
    out_path = pathlib.Path(args.output or f"results/dataset_availability/{label}.zarr")

    results: list[DatasetAvailability] = []
    skipped: list[dict] = []
    for stream_name, stream_info in streams.items():
        reader_type = stream_info.get("type", "")
        for fname in stream_info.get("filenames", []):
            try:
                path = resolve_dataset_path(str(fname), data_paths)
                logger.info(f"Analyzing {stream_name}/{path.name} (type={reader_type})")
                if reader_type in _OBS_TYPES:
                    r = analyze_obs(
                        path, stream_name, bin_seconds, start, end, args.row_chunk_stride
                    )
                elif reader_type in _ANEMOI_TYPES:
                    r = analyze_anemoi(
                        path,
                        stream_name,
                        dict(stream_info),
                        bin_seconds,
                        start,
                        end,
                        args.check_nans,
                        args.nan_time_samples,
                    )
                else:
                    skipped.append(
                        {
                            "stream": stream_name,
                            "dataset": str(fname),
                            "reason": f"unsupported reader type '{reader_type}'",
                        }
                    )
                    logger.warning(f"Skipping {stream_name}/{fname}: type '{reader_type}'.")
                    continue
                results.append(r)
            except (FileNotFoundError, ValueError) as exc:
                skipped.append({"stream": stream_name, "dataset": str(fname), "reason": str(exc)})
                logger.warning(f"Skipping {stream_name}/{fname}: {exc}")

    if not results:
        raise SystemExit("No dataset could be analyzed; see warnings above.")

    write_store(out_path, results, skipped, label, bin_seconds)
    summaries = [summarize(r) for r in results]
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"label": label, "datasets": summaries}, indent=2))

    sys.stdout.write("\n" + _format_summary_table(summaries) + "\n")
    if skipped:
        sys.stdout.write(f"\nSkipped {len(skipped)} dataset(s); see {summary_path}.\n")
    sys.stdout.write(f"\nStats store: {out_path}\nSummary:     {summary_path}\n")
    sys.stdout.write(
        "Plot with: uv run python packages/science/plot_dataset_availability.py "
        f"--stats {out_path}\n"
    )


if __name__ == "__main__":
    main()
