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
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# keep in sync with weathergen.utils.io_timing.PATH_SEP (not imported to keep this standalone)
PATH_SEP = " > "
UNTIMED = "[untimed]"


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


def print_table(header: list[tuple[str, int]], rows: list[list[str]]) -> None:
    width = max([len(r[0]) for r in rows] + [len(header[0][0])])
    line = header[0][0].ljust(width) + "".join(h.rjust(w) for h, w in header[1:])
    print(line)
    print("-" * len(line))
    for r in rows:
        print(
            r[0].ljust(width)
            + "".join(c.rjust(w) for c, (_, w) in zip(r[1:], header[1:], strict=False))
        )


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
                ]
            )
        print_table(header, rows)


def report_compare(stats_a: pd.DataFrame, stats_b: pd.DataFrame, name_a: str, name_b: str):
    cols = ["calls", "total_s", "mean_ms", "pct_parent"]
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
                ]
            )
        print_table(header, rows)
    return merged[["stream", "path"] + [f"{c}_{x}" for c in cols for x in "ab"] + ["ratio"]]


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
        "--warmup", type=int, default=0, help="drop the first N calls of each op per process"
    )
    parser.add_argument("--csv", type=Path, help="also write the summary table to this CSV")
    args = parser.parse_args()

    def prepare(sources: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = load(sources)
        if args.stream:
            df = df[df["stream"].isin(args.stream)]
            if df.empty:
                sys.exit(f"No records for streams {args.stream}.")
        df = drop_warmup(df, args.warmup)
        return df, summarize(df)

    df_a, stats_a = prepare(args.run)
    if args.compare:
        _, stats_b = prepare(args.compare)
        names = [", ".join(str(p) for p in ps) for ps in (args.run, args.compare)]
        table = report_compare(stats_a, stats_b, *names)
    else:
        report(stats_a, df_a)
        table = stats_a.drop(columns=["t_rel", "parent"])

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
