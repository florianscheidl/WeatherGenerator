#!/usr/bin/env -S uv run

# ruff: noqa: T201
"""
Convert the io_timing records (WEATHERGEN_IO_TIMING=1, see src/weathergen/utils/io_timing.py)
into a Chrome Trace Event JSON that opens in Perfetto (https://ui.perfetto.dev) or
chrome://tracing: one track group per process (rank, data loader worker), one track per thread,
nested calls stacked. Use it to inspect the data loader workers next to the nsys timeline when
nsys does not trace the (forked) workers. The nsys capture windows (``nsys-window`` events)
appear as spans on a separate "nsys windows" track; times are relative to the first record.

USAGE EXAMPLES (from the root of the repo):
  uv run scripts/io_timing_to_trace.py io_timing/ -o io_timing_trace.json
  uv run scripts/io_timing_to_trace.py io_timing/ --window startup -o startup_trace.json
"""

import argparse
import json
import sys
from pathlib import Path

# keep in sync with weathergen.utils.nsys_windows.IO_EVENT_STREAM (not imported: standalone)
EVENT_STREAM = "nsys-window"
WINDOWS_PID = 0  # synthetic process for the window spans; real pids are renumbered from 1


def load(sources: list[Path]) -> list[dict]:
    """Load records from directories (all io_timing_*.jsonl inside) and/or files."""
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
                    continue  # a process killed mid-write can leave a truncated last line
    return records


def window_spans(events: list[dict]) -> dict[str, tuple[float, float]]:
    """Window name -> (earliest open, latest close) over all processes; open-ended if unclosed."""
    spans: dict[str, tuple[float, float]] = {}
    last = max((e["t_start"] for e in events), default=0.0)
    for e in events:
        name, kind = e["op"].rsplit(" ", 1)
        t_open, t_close = spans.get(name, (float("inf"), float("-inf")))
        if kind == "open":
            t_open = min(t_open, e["t_start"])
        else:
            t_close = max(t_close, e["t_start"])
        spans[name] = (t_open, t_close)
    return {n: (o, c if c > o else last) for n, (o, c) in spans.items() if o != float("inf")}


def convert(records: list[dict], window: str | None = None) -> dict:
    """Build the Trace Event JSON (a dict with ``traceEvents``) from io_timing records."""
    events = [r for r in records if r["stream"] == EVENT_STREAM]
    calls = [r for r in records if r["stream"] != EVENT_STREAM]
    spans = window_spans(events)
    if window is not None:
        if window not in spans:
            sys.exit(f"No window '{window}' in the records (found: {sorted(spans)}).")
        t_open, t_close = spans[window]
        calls = [r for r in calls if t_open <= r["t_start"] <= t_close]
    if not calls:
        sys.exit("No records to convert.")

    t0 = min(r["t_start"] for r in calls + events)
    # stable small ids; first record decides the order, so the main process tends to come first
    procs: dict[tuple[str, int], int] = {}
    threads: dict[tuple[int, int], int] = {}
    out: list[dict] = []
    for r in sorted(calls, key=lambda r: (r["t_start"], -r["dt"])):
        pid = procs.setdefault((r["host"], r["pid"]), len(procs) + 1)
        tid = threads.setdefault((pid, r.get("tid", 0)), len(threads) + 1)
        out.append(
            {
                "name": r["op"],
                "cat": r["stream"],
                "ph": "X",
                "ts": (r["t_start"] - t0) * 1e6,
                "dur": r["dt"] * 1e6,
                "pid": pid,
                "tid": tid,
                "args": {"stream": r["stream"], "path": r["path"]},
            }
        )
    for (host, os_pid), pid in procs.items():
        out.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": {"name": f"{host} pid {os_pid}"},
            }
        )
    for name, (t_open, t_close) in spans.items():
        if window is not None and name != window:
            continue
        out.append(
            {
                "name": f"nsys window: {name}",
                "cat": EVENT_STREAM,
                "ph": "X",
                "ts": (t_open - t0) * 1e6,
                "dur": (t_close - t_open) * 1e6,
                "pid": WINDOWS_PID,
                "tid": 0,
            }
        )
    if spans:
        out.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": WINDOWS_PID,
                "args": {"name": "nsys windows"},
            }
        )
    return {"traceEvents": out, "displayTimeUnit": "ms"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", nargs="+", type=Path, help="timing dir(s) or .jsonl file(s)")
    parser.add_argument("-o", "--output", type=Path, default=Path("io_timing_trace.json"))
    parser.add_argument("--window", help="only calls inside this nsys capture window")
    args = parser.parse_args()

    trace = convert(load(args.run), args.window)
    args.output.write_text(json.dumps(trace))
    n = sum(1 for e in trace["traceEvents"] if e["ph"] == "X")
    print(f"Wrote {args.output} ({n} events). Open it at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
