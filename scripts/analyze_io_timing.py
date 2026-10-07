#!/usr/bin/env -S uv run

# ruff: noqa: T201
"""
Summarize the data reading / preprocessing timings written with WEATHERGEN_IO_TIMING=1
(see src/weathergen/utils/io_timing.py).

Prints, per stream, the tree of timed operations with call counts, total and per-call times,
and each operation's share of its parent. "[untimed]" rows are the part of a parent's time
that none of its timed children cover. Times are summed over all processes (ranks and data
loader workers), so totals can exceed the wall-clock duration of the run.

USAGE EXAMPLES (from the root of the repo):
  uv run scripts/analyze_io_timing.py io_timing/
  uv run scripts/analyze_io_timing.py io_timing_zarr2/ --compare io_timing_zarr3/
  uv run scripts/analyze_io_timing.py io_timing/ --stream ERA5 --warmup 5 --csv summary.csv
  uv run scripts/analyze_io_timing.py io_timing/ --window startup   # nsys capture window
  uv run scripts/analyze_io_timing.py logs/<run-id>/profiling/io_timing/ --memory-csv memory.csv
      # memory needs launch-slurm.py --io-timing-memory (src/weathergen/utils/memory_sampler.py)

If the run has memory samples, the report ends with the peak memory of every process and the
node-wide total over time. Totals sum PSS (shared pages split between the processes sharing
them), so fork-inherited and shared-memory pages are not counted once per worker.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

# keep in sync with weathergen.utils.io_timing.PATH_SEP (not imported to keep this standalone)
PATH_SEP = " > "
UNTIMED = "[untimed]"
# keep in sync with weathergen.utils.nsys_windows.IO_EVENT_STREAM
EVENT_STREAM = "nsys-window"
# keep in sync with weathergen.utils.memory_sampler.STREAM
MEMORY_STREAM = "memory"
MEMORY_COLS = ["rss_mib", "pss_mib", "pss_anon_mib", "pss_file_mib", "pss_shmem_mib"]
GPU_COLS = ["gpu_alloc_mib", "gpu_reserved_mib", "gpu_max_alloc_mib"]


def load(sources: list[Path]) -> pd.DataFrame:
    """Load timing records from directories (all io_timing_*.jsonl inside) and/or files."""
    files: list[Path] = []
    for src in sources:
        files += sorted(src.glob("io_timing_*.jsonl")) if src.is_dir() else [src]
    if not files:
        sys.exit(f"No io_timing_*.jsonl files found in {[str(s) for s in sources]}.")

    records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # a process killed mid-write can leave a truncated last line
                    continue
    df = pd.DataFrame.from_records(records)
    # first-call time relative to the run start, used to order the tree
    df["t_rel"] = df["t_start"] - df["t_start"].min()
    return df


def select_window(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Keep the calls that started inside the nsys capture window ``name``.

    The window bounds are the ``<name> open`` / ``<name> close`` events written by
    weathergen.utils.nsys_windows (earliest open and latest close over all processes).
    """
    events = df[df["stream"] == EVENT_STREAM]
    t_open = events.loc[events["op"] == f"{name} open", "t_start"]
    t_close = events.loc[events["op"] == f"{name} close", "t_start"]
    if t_open.empty:
        known = sorted({op.rsplit(" ", 1)[0] for op in events["op"]})
        sys.exit(f"No window '{name}' in the records (found: {known}).")
    # a window that was never closed (run ended early) extends to the last record
    t_end = t_close.max() if not t_close.empty else (df["t_start"] + df["dt"]).max()
    return df[(df["t_start"] >= t_open.min()) & (df["t_start"] <= t_end)]


def drop_warmup(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Drop the first n calls of every operation in every process."""
    if n <= 0:
        return df
    df = df.sort_values("t_start")
    return df[df.groupby(["pid", "host", "stream", "path"]).cumcount() >= n]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (stream, path) statistics, with untimed remainder rows for every parent."""
    g = df.groupby(["stream", "path"])["dt"]
    stats = pd.DataFrame(
        {
            "calls": g.count(),
            "total_s": g.sum(),
            "mean_ms": g.mean() * 1e3,
            "p50_ms": g.median() * 1e3,
            "p95_ms": g.quantile(0.95) * 1e3,
            "max_ms": g.max() * 1e3,
            "t_rel": df.groupby(["stream", "path"])["t_rel"].min(),
        }
    ).reset_index()

    # untimed remainder: parent total minus the sum of its direct children
    stats["parent"] = stats["path"].map(_parent)
    children_total = stats.groupby(["stream", "parent"])["total_s"].sum()
    rows = []
    for row in stats.itertuples():
        key = (row.stream, row.path)
        if key in children_total.index:
            rest = row.total_s - children_total[key]
            rows.append(
                {
                    "stream": row.stream,
                    "path": row.path + PATH_SEP + UNTIMED,
                    "calls": row.calls,
                    "total_s": rest,
                    "mean_ms": rest / row.calls * 1e3,
                    "t_rel": float("inf"),
                    "parent": row.path,
                }
            )
    stats = pd.concat([stats, pd.DataFrame(rows)], ignore_index=True)

    parent_total = stats.set_index(["stream", "path"])["total_s"].to_dict()
    stats["pct_parent"] = [
        100 * row.total_s / parent_total.get((row.stream, row.parent), float("nan"))
        for row in stats.itertuples()
    ]
    # share of the stream total, i.e. of the sum of the stream's top-level operations
    stream_total = stats[stats["parent"].isna()].groupby("stream")["total_s"].sum()
    stats["pct_total"] = 100 * stats["total_s"] / stats["stream"].map(stream_total)
    return stats


def _parent(path: str) -> str | None:
    return path.rsplit(PATH_SEP, 1)[0] if PATH_SEP in path else None


def tree_order(paths: pd.DataFrame) -> list[str]:
    """Depth-first order of paths (columns path, parent, t_rel), siblings by first call."""
    children: dict[str | None, list[tuple[float, str]]] = {}
    known = set(paths["path"])
    for row in paths.itertuples():
        parent = row.parent if row.parent in known else None
        children.setdefault(parent, []).append((row.t_rel, row.path))

    order: list[str] = []

    def visit(node: str | None) -> None:
        for _, path in sorted(children.get(node, [])):
            order.append(path)
            visit(path)

    visit(None)
    return order


def _label(path: str) -> str:
    depth = path.count(PATH_SEP)
    return "  " * depth + path.rsplit(PATH_SEP, 1)[-1]


def _fmt(value: float, spec: str) -> str:
    return "" if pd.isna(value) else format(value, spec)


def print_table(
    header: list[tuple[str, int]], rows: list[list[str]], footer: list[str] | None = None
) -> None:
    width = max([len(r[0]) for r in rows + [footer or [""]]] + [len(header[0][0])])

    def fmt_row(r: list[str]) -> str:
        cells = zip(r[1:], header[1:], strict=False)
        return r[0].ljust(width) + "".join(c.rjust(w) for c, (_, w) in cells)

    line = header[0][0].ljust(width) + "".join(h.rjust(w) for h, w in header[1:])
    print(line)
    print("-" * len(line))
    for r in rows:
        print(fmt_row(r))
    if footer is not None:
        print("-" * len(line))
        print(fmt_row(footer))


def _stream_total(s: pd.DataFrame, col: str = "total_s") -> float:
    """Sum of the top-level operations (rows without a parent) of one stream."""
    return s.loc[s["parent"].isna(), col].sum()


def report(stats: pd.DataFrame, df: pd.DataFrame) -> None:
    header = [
        ("operation", 0),
        ("calls", 9),
        ("total[s]", 11),
        ("mean[ms]", 11),
        ("p50[ms]", 10),
        ("p95[ms]", 10),
        ("max[ms]", 10),
        ("%parent", 9),
        ("%total", 9),
    ]
    for stream, s in stats.groupby("stream", sort=True):
        n_proc = df[df["stream"] == stream][["host", "pid"]].drop_duplicates().shape[0]
        print(f"\n=== {stream}  ({n_proc} processes)")
        s = s.set_index("path")
        rows = []
        for path in tree_order(s.reset_index()):
            r = s.loc[path]
            rows.append(
                [
                    _label(path),
                    f"{int(r.calls):,}",
                    _fmt(r.total_s, ".3f"),
                    _fmt(r.mean_ms, ".3f"),
                    _fmt(r.p50_ms, ".3f"),
                    _fmt(r.p95_ms, ".3f"),
                    _fmt(r.max_ms, ".3f"),
                    _fmt(r.pct_parent, ".1f"),
                    _fmt(r.pct_total, ".1f"),
                ]
            )
        total = _stream_total(s)
        footer = ["total (sum of top level)", "", f"{total:.3f}"] + [""] * 5 + ["100.0"]
        print_table(header, rows, footer)


def report_compare(stats_a: pd.DataFrame, stats_b: pd.DataFrame, name_a: str, name_b: str):
    cols = ["calls", "total_s", "mean_ms", "pct_parent", "pct_total"]
    merged = stats_a.merge(
        stats_b, on=["stream", "path"], how="outer", suffixes=("_a", "_b")
    ).assign(
        parent=lambda m: m["path"].map(_parent),
        t_rel=lambda m: m["t_rel_a"].fillna(m["t_rel_b"]),
        ratio=lambda m: m["mean_ms_b"] / m["mean_ms_a"],
    )
    print(f"\nA = {name_a}\nB = {name_b}\nratio = mean B / mean A (< 1: B is faster)")
    header = [
        ("operation", 0),
        ("calls A", 9),
        ("calls B", 9),
        ("mean A[ms]", 12),
        ("mean B[ms]", 12),
        ("ratio", 8),
        ("total A[s]", 12),
        ("total B[s]", 12),
        ("%total A", 10),
        ("%total B", 10),
    ]
    for stream, s in merged.groupby("stream", sort=True):
        print(f"\n=== {stream}")
        s = s.set_index("path")
        rows = []
        for path in tree_order(s.reset_index()):
            r = s.loc[path]
            rows.append(
                [
                    _label(path),
                    _fmt(r.calls_a, ",.0f"),
                    _fmt(r.calls_b, ",.0f"),
                    _fmt(r.mean_ms_a, ".3f"),
                    _fmt(r.mean_ms_b, ".3f"),
                    _fmt(r.ratio, ".2f"),
                    _fmt(r.total_s_a, ".3f"),
                    _fmt(r.total_s_b, ".3f"),
                    _fmt(r.pct_total_a, ".1f"),
                    _fmt(r.pct_total_b, ".1f"),
                ]
            )
        total_a, total_b = _stream_total(s, "total_s_a"), _stream_total(s, "total_s_b")
        footer = ["total (sum of top level)"] + [""] * 4
        footer += [_fmt(total_b / total_a, ".2f"), f"{total_a:.3f}", f"{total_b:.3f}"]
        print_table(header, rows, footer + ["100.0", "100.0"])
    return merged[["stream", "path"] + [f"{c}_{x}" for c in cols for x in "ab"] + ["ratio"]]


def prepare_memory(mem: pd.DataFrame) -> pd.DataFrame:
    """Add ``mem_mib`` (PSS, RSS where PSS is unavailable) and ``group`` (rank main pid)."""
    mem = mem.copy()
    for col in MEMORY_COLS + GPU_COLS:
        if col not in mem:
            mem[col] = float("nan")
    mem["mem_mib"] = mem["pss_mib"].fillna(mem["rss_mib"])
    # a worker belongs to the rank whose main process is its parent
    mem["group"] = mem["pid"].where(mem["role"] == "main", mem["ppid"])
    return mem


def process_peaks(mem: pd.DataFrame) -> pd.DataFrame:
    """Peak memory (MiB) per process, and when the peak PSS happened (s since run start)."""
    g = mem.groupby(["host", "group", "role", "pid"])
    peaks = g[MEMORY_COLS + ["mem_mib"]].max()
    peaks["t_peak_s"] = mem.loc[g["mem_mib"].idxmax(), "t_rel"].to_numpy()
    return peaks.reset_index().sort_values(["host", "group", "role"])


def memory_timeline(mem: pd.DataFrame, bin_s: float = 1.0) -> pd.DataFrame:
    """Node-wide memory per time bin: PSS summed over processes, split main vs. workers.

    A sample is carried forward for 1.5 sampling intervals (the median spacing of the samples),
    so a worker that exited stops counting about one interval after its last sample, while a
    process whose next sample is delayed or lies outside a ``--window`` is still counted.
    GPU columns are the maximum over the main processes.
    """
    mem = mem.assign(bin=(mem["t_rel"] // bin_s).astype(int))
    per_proc = mem.groupby(["bin", "pid"]).agg(mib=("mem_mib", "last"), role=("role", "last"))
    wide = per_proc["mib"].unstack("pid")
    spacing = mem.sort_values("t_rel").groupby("pid")["t_rel"].diff().dropna()
    spacing = spacing[spacing > 0]
    interval = float(spacing.median()) if not spacing.empty else bin_s
    carry = max(1, math.ceil(1.5 * interval / bin_s))
    wide = wide.reindex(range(int(wide.index.min()), int(wide.index.max()) + 1)).ffill(limit=carry)
    is_main = per_proc["role"].unstack("pid").ffill().iloc[0].eq("main")
    main_cols = [c for c in wide.columns if is_main.get(c, False)]
    out = pd.DataFrame(
        {
            "t_s": wide.index * bin_s,
            "total_mib": wide.sum(axis=1).to_numpy(),
            "main_mib": wide[main_cols].sum(axis=1).to_numpy(),
        }
    )
    out["workers_mib"] = out["total_mib"] - out["main_mib"]
    gpu = mem.groupby("bin")[GPU_COLS].max().reindex(wide.index)
    for col in GPU_COLS:
        out[col] = gpu[col].to_numpy()
    return out


def report_memory(mem: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    """Print the memory report and return the node-wide timeline."""
    mem = prepare_memory(mem)
    print("\nMemory (MiB), peak per process")
    peaks = process_peaks(mem)
    header = [("host / rank group / role", 0)] + [
        (h, 10) for h in ["rss", "pss", "anon", "file", "shmem", "t_peak_s"]
    ]
    rows = [
        [f"{r.host} / {int(r.group)} / {r.role}"]
        + [_fmt(getattr(r, c), ".0f") for c in MEMORY_COLS]
        + [_fmt(r.t_peak_s, ".1f")]
        for r in peaks.itertuples()
    ]
    print_table(header, rows)
    if mem["pss_mib"].isna().any():
        print("(some samples have no PSS; the totals below use RSS for them and overcount shared)")

    timeline = memory_timeline(mem, bin_s)
    peak = timeline.loc[timeline["total_mib"].idxmax()]
    print(
        f"\nNode-wide total over all processes (PSS): peak {peak.total_mib:.0f} MiB at "
        f"t={peak.t_s:.0f}s (main processes {peak.main_mib:.0f}, workers {peak.workers_mib:.0f})"
    )
    if timeline["gpu_max_alloc_mib"].notna().any():
        print(
            f"GPU (PyTorch allocator, max over main processes): allocated peak "
            f"{timeline['gpu_alloc_mib'].max():.0f} MiB, reserved peak "
            f"{timeline['gpu_reserved_mib'].max():.0f} MiB, max-allocated "
            f"{timeline['gpu_max_alloc_mib'].max():.0f} MiB"
        )
    return timeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", nargs="+", type=Path, help="timing dir(s) or .jsonl file(s)")
    parser.add_argument(
        "--compare", nargs="+", type=Path, help="second run's dir(s)/file(s) to compare against"
    )
    parser.add_argument("--stream", nargs="+", help="only show these streams")
    parser.add_argument(
        "--window", help="only show calls inside this nsys capture window (e.g. startup, steady)"
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="drop the first N calls of each op per process"
    )
    parser.add_argument("--csv", type=Path, help="also write the summary table to this CSV")
    parser.add_argument(
        "--memory-bin", type=float, default=1.0, help="time bin (s) of the node-wide memory total"
    )
    parser.add_argument(
        "--memory-csv", type=Path, help="also write the node-wide memory timeline to this CSV"
    )
    args = parser.parse_args()

    def prepare(sources: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = load(sources)
        if args.window:
            df = select_window(df, args.window)
        mem = df[df["stream"] == MEMORY_STREAM]
        df = df[~df["stream"].isin([EVENT_STREAM, MEMORY_STREAM])]
        if df.empty and mem.empty:
            sys.exit(
                "No timing or memory records"
                + (f" in window {args.window}." if args.window else ".")
            )
        if args.stream:
            df = df[df["stream"].isin(args.stream)]
            if df.empty:
                sys.exit(f"No records for streams {args.stream}.")
        df = drop_warmup(df, args.warmup)
        return df, (summarize(df) if not df.empty else df), mem

    df_a, stats_a, mem_a = prepare(args.run)
    if args.compare:
        _, stats_b, _ = prepare(args.compare)
        names = [", ".join(str(p) for p in ps) for ps in (args.run, args.compare)]
        table = report_compare(stats_a, stats_b, *names)
    elif not df_a.empty:
        report(stats_a, df_a)
        table = stats_a.drop(columns=["t_rel", "parent"])
    else:
        table = pd.DataFrame()

    if not mem_a.empty:
        timeline = report_memory(mem_a, args.memory_bin)
        if args.memory_csv:
            timeline.to_csv(args.memory_csv, index=False)
            print(f"Wrote {args.memory_csv}")

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
