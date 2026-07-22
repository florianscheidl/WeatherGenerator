#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# ruff: noqa: T201

"""
Summarise a CUDA allocator snapshot as text, for reading without memory_viz.

`torch.cuda.memory._dump_snapshot` pickles plain dicts, so this needs no torch. It
replays the allocation trace and attributes every block to the chain of project frames
that allocated it, which answers the questions memory_viz makes you eyeball:

  - what is live at the high-water mark, grouped by call site (who *retains* memory)
  - how much each call site allocates in total (who *churns* memory)
  - how long each call site's blocks stay alive (transient vs retained to backward)

The trace is a ring buffer of the *last* `max_entries` events, so it almost never starts
at an empty allocator: memory allocated before the window (parameters, optimizer state,
persistent buffers) is live but has no `alloc` event. Replaying from zero therefore
measures a delta, not a peak. This tool anchors the replay on the exact live set recorded
in `snapshot["segments"]` at dump time and reconstructs the absolute curve backwards from
it, so `live at peak` is real device bytes. The reconciliation line in the header proves
the arithmetic closed.

`--by-section` groups by `record_function` ranges from `snapshot["external_annotations"]`
(`nn.Module: ...`, `ProfilerStep#N`, `FSDP::...`), joined to allocations on `time_us`.
Those are only present when the torch profiler ran alongside the memory recorder.

`--diff` compares two snapshots per call site, which is what an activation-checkpointing
ablation actually needs: a per-line number instead of "the trace looks the same".

Usage:
    scripts/memory_snapshot_report.py <snapshot.pickle>
    scripts/memory_snapshot_report.py <after.pickle> --diff <before.pickle>
    scripts/memory_snapshot_report.py <snapshot.pickle> --depth 4 --top 25
    scripts/memory_snapshot_report.py <snapshot.pickle> --by-section
    scripts/memory_snapshot_report.py <snapshot.pickle> --by-section --section-pattern nn.Module
"""

import argparse
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIB = 2**20
GIB = 2**30

# blocks that predate the trace window: still live at dump time (attributable through the
# frames the allocator kept with the block) or freed inside the window (no stack anywhere)
PRE_PREFIX = "[pre-window] "
PRE_TRANSIENT = "[pre-window] <freed during window, no stack recorded>"

# an alloc event with no frames comes from a thread with no Python stack — in practice the
# autograd engine's backward thread, since C++ unwinding yields nothing on this build
NO_STACK_EVENT = "<no python stack: backward or non-Python thread>"
# a live block with no frames was allocated before _record_memory_history() was enabled
NO_STACK_BLOCK = "<allocated before recording started>"
OUTSIDE_SECTION = "<outside any matching section>"


def _label(frame: dict[str, Any], with_line: bool = True) -> str:
    name = Path(frame.get("filename", "?")).name
    line = f":{frame.get('line')}" if with_line else ""
    return f"{name}{line}:{frame.get('name', '?')}"


def call_site(
    frames: list[dict[str, Any]] | None,
    pkg: str,
    depth: int = 3,
    missing: str = NO_STACK_EVENT,
    with_line: bool = True,
) -> str:
    """Innermost `depth` project frames, outermost first.

    A single frame is not enough to identify an allocation in this codebase: the encoder,
    the forecast engine and the decoder all funnel into the same few `attention.py` lines,
    so attributing to the innermost frame alone merges unrelated call paths. The chain
    keeps them apart — `model.py:703 forecast_engine > ... > attention.py:608` is a
    different site from the same attention line reached via `predict_decoders`.
    """
    if not frames:
        return missing
    own = [
        f
        for f in frames
        if f"/{pkg}/" in f.get("filename", "") or f.get("filename", "").startswith(f"{pkg}/")
    ]
    if own:
        chain = own[: max(1, depth)]
        return " > ".join(_label(f, with_line) for f in reversed(chain))
    for frame in frames:
        if frame.get("filename", "").endswith(".py"):
            return _label(frame, with_line)
    return "<no python frame>"


@dataclass
class Snapshot:
    """One device's view of a dumped allocator snapshot."""

    path: Path
    device: int
    trace: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    settings: dict[str, Any]

    @property
    def reserved(self) -> int:
        return sum(s.get("total_size", 0) for s in self.segments)

    @property
    def allocated(self) -> int:
        return sum(s.get("allocated_size", 0) for s in self.segments)

    def live_blocks(self) -> dict[int, tuple[int, list[dict[str, Any]] | None]]:
        """Blocks in use at dump time, by address: (requested bytes, frames).

        `requested_size` is the unrounded request, which is what `alloc` trace entries
        record (`orig_size` in the allocator), so the two are directly comparable.
        """
        out: dict[int, tuple[int, list[dict[str, Any]] | None]] = {}
        for seg in self.segments:
            for block in seg.get("blocks", []):
                if block.get("state") == "active_allocated":
                    size = block.get("requested_size", block.get("size", 0))
                    out[block["address"]] = (size, block.get("frames"))
        return out

    def awaiting_free(self) -> int:
        return sum(
            b.get("requested_size", b.get("size", 0))
            for s in self.segments
            for b in s.get("blocks", [])
            if b.get("state") == "active_awaiting_free"
        )


class SiteStats:
    """Per-call-site accounting over one trace."""

    def __init__(self) -> None:
        self.n_allocs = 0
        self.total_bytes = 0
        self.live_now = 0
        self.max_concurrent = 0
        self.live_at_peak = 0
        self.lifetimes_us: list[int] = []

    def alloc(self, size: int) -> None:
        self.n_allocs += 1
        self.total_bytes += size
        self.live_now += size
        self.max_concurrent = max(self.max_concurrent, self.live_now)

    def free(self, size: int, lifetime_us: int) -> None:
        self.live_now -= size
        if lifetime_us >= 0:
            self.lifetimes_us.append(lifetime_us)

    @property
    def median_lifetime_us(self) -> int:
        if not self.lifetimes_us:
            return -1
        ordered = sorted(self.lifetimes_us)
        return ordered[len(ordered) // 2]


@dataclass
class Replay:
    """Result of replaying one device trace, anchored on the dump-time live set."""

    sites: dict[str, SiteStats] = field(default_factory=lambda: defaultdict(SiteStats))
    peak_live: int = 0
    peak_event: int = -1
    peak_time_us: int = 0
    n_events: int = 0
    n_allocs: int = 0
    n_frees: int = 0
    # bytes live when the trace window opens, split by what we can say about them
    pre_persistent: int = 0  # still live at dump: attributable through segment block frames
    pre_transient: int = 0  # freed inside the window: an alloc event exists nowhere
    pre_at_peak: int = 0  # how much of the two above was still live at the peak instant
    unmatched_frees: int = 0
    anchored: bool = False
    dump_live: int = 0
    reconciled: int = 0  # (start + net) - dump_live; zero when the arithmetic closes
    span_us: int = 0
    start_us: int = 0
    off_stream_bytes: int = 0
    never_freed: list[tuple[int, str]] = field(default_factory=list)
    # event index -> (bytes, site) for window blocks live at the peak; feeds --by-section
    peak_entries: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def pre_window(self) -> int:
        return self.pre_persistent + self.pre_transient


def replay(snap: Snapshot, pkg: str, depth: int = 3, with_line: bool = True) -> Replay:
    """Replay alloc/free events, tracking absolute live bytes overall and per call site.

    The relative curve is offset by a constant (the bytes live when the window opens), so
    the argmax is the same before and after anchoring — one pass suffices.
    """
    out = Replay()
    trace = snap.trace
    out.n_events = len(trace)
    if trace and "time_us" in trace[0] and "time_us" in trace[-1]:
        out.start_us = trace[0]["time_us"]
        out.span_us = trace[-1]["time_us"] - out.start_us

    # traces carry free_requested and/or free_completed; prefer completed, but fall
    # back so a trace with only one of them still balances
    actions = {e.get("action") for e in trace}
    free_action = "free_completed" if "free_completed" in actions else "free_requested"

    dump_live = snap.live_blocks()
    out.anchored = bool(dump_live)
    out.dump_live = sum(size for size, _ in dump_live.values())

    live: dict[int, tuple[int, str, int, int]] = {}  # addr -> (size, site, index, time_us)
    rel = 0
    rel_peak = 0
    freed_unmatched = 0
    freed_unmatched_at_peak = 0
    peak_live: dict[int, tuple[int, str, int, int]] = {}

    for i, event in enumerate(trace):
        action = event.get("action")
        if action == "alloc":
            size = int(event.get("size", 0))
            site = call_site(event.get("frames"), pkg, depth, NO_STACK_EVENT, with_line)
            live[int(event["addr"])] = (size, site, i, int(event.get("time_us", 0)))
            rel += size
            out.sites[site].alloc(size)
            out.n_allocs += 1
            if event.get("stream"):
                out.off_stream_bytes += size
            if rel > rel_peak:
                rel_peak = rel
                out.peak_event = i
                out.peak_time_us = int(event.get("time_us", 0))
                peak_live = dict(live)
                freed_unmatched_at_peak = freed_unmatched
        elif action == free_action:
            size = int(event.get("size", 0))
            rel -= size
            out.n_frees += 1
            entry = live.pop(int(event.get("addr", -1)), None)
            if entry is None:
                # the block was allocated before the window opened
                out.unmatched_frees += 1
                freed_unmatched += size
                continue
            born_size, site, _, born_us = entry
            now_us = int(event.get("time_us", 0))
            out.sites[site].free(born_size, now_us - born_us if now_us and born_us else -1)

    # blocks still live at dump that never got an alloc event are the persistent part of
    # the pre-window pool; the allocator kept their stacks, so they are attributable
    pre_by_site: Counter[str] = Counter()
    for addr, (size, frames) in dump_live.items():
        if addr in live:
            continue
        out.pre_persistent += size
        site = call_site(frames, pkg, depth, NO_STACK_BLOCK, with_line)
        pre_by_site[PRE_PREFIX + site] += size
    out.pre_transient = freed_unmatched

    start_live = out.pre_window
    out.peak_live = start_live + rel_peak
    out.reconciled = (start_live + rel) - out.dump_live

    # live at the peak = window blocks live then + everything that predates the window and
    # had not yet been freed at that point
    for size, site, idx, _ in peak_live.values():
        out.sites[site].live_at_peak += size
        out.peak_entries.append((idx, size, site))
    for site, total in pre_by_site.items():
        # these appear in the peak table only: they have no alloc event, so no churn
        out.sites[site].live_at_peak += total
    pool_at_peak = out.pre_transient - freed_unmatched_at_peak
    out.pre_at_peak = out.pre_persistent + pool_at_peak
    if pool_at_peak:
        out.sites[PRE_TRANSIENT].live_at_peak += pool_at_peak

    still_live = ((size, site) for size, site, _, _ in live.values())
    out.never_freed = sorted(still_live, key=lambda t: -t[0])[:10]
    return out


def _table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    if not rows:
        return "  (nothing to show)"
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row, strict=True)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
    return "\n".join(lines)


def _ms(us: int) -> str:
    return "never freed" if us < 0 else f"{us / 1000:.1f}"


def header(snap: Snapshot, rep: Replay, label: str, max_entries: int) -> None:
    print(f"=== {label}  (device {snap.device})")
    conf = snap.settings.get("PYTORCH_CUDA_ALLOC_CONF")
    if conf:
        print(f"allocator: {conf}")
    print(
        f"at dump: {snap.allocated / GIB:.3f} GiB allocated, {snap.reserved / GIB:.3f} GiB reserved"
        f" ({(snap.reserved - snap.allocated) / GIB:.3f} GiB held by the caching allocator)"
    )
    span = f", {rep.span_us / 1e6:.2f} s" if rep.span_us else ""
    print(
        f"trace window: {rep.n_events:,} events{span}"
        f" (alloc {rep.n_allocs:,} / free {rep.n_frees:,})"
    )

    if not rep.anchored:
        print(
            "WARNING: no segments in this snapshot — the replay could not be anchored, so"
            " every number below is a delta from the start of the trace window, not an"
            " absolute figure."
        )
    else:
        print(
            f"live when the window opens: {rep.pre_window / GIB:.3f} GiB"
            f" ({rep.pre_persistent / GIB:.3f} GiB still live at dump and attributed below as"
            f" '{PRE_PREFIX.strip()}', {rep.pre_transient / GIB:.3f} GiB freed inside the window"
            f" by {rep.unmatched_frees:,} frees whose allocation is not in the trace)"
        )
    at = ""
    if rep.start_us and rep.peak_time_us:
        at = f", t+{(rep.peak_time_us - rep.start_us) / 1e6:.2f} s into the window"
    print(f"peak live: {rep.peak_live / GIB:.3f} GiB at event {rep.peak_event:,}{at}")
    if rep.anchored and rep.peak_live:
        print(
            f"  of which {rep.pre_at_peak / GIB:.3f} GiB"
            f" ({100 * rep.pre_at_peak / rep.peak_live:.0f}%) was allocated before the window and"
            " was still live at that instant; the rest of the opening balance had been freed"
            " by then"
        )

    if rep.anchored and abs(rep.reconciled) > MIB:
        print(
            f"WARNING: replay does not reconcile with the dump: reconstructed end state is"
            f" {rep.reconciled / MIB:+.1f} MiB off the {rep.dump_live / GIB:.3f} GiB in"
            " segments. Treat the absolute numbers with suspicion."
        )
    elif rep.anchored:
        print(
            f"  reconciles with dump-time segments to {rep.reconciled / MIB:+.3f} MiB"
            f" (end state {rep.dump_live / GIB:.3f} GiB)"
        )

    if rep.n_events >= max_entries:
        print(
            f"NOTE: the trace holds exactly {max_entries:,} events, so the ring buffer wrapped and"
            " only the tail of the run is covered. Anchoring keeps the totals correct; what is"
            " lost is per-site attribution for anything allocated earlier. Profile fewer steps"
            " (or raise max_entries) to widen the window."
        )
    if awaiting := snap.awaiting_free():
        print(
            f"NOTE: {awaiting / MIB:.1f} MiB in active_awaiting_free blocks is excluded from"
            " the anchor (another stream still holds it)."
        )
    if rep.off_stream_bytes:
        print(
            f"NOTE: {rep.off_stream_bytes / MIB:.1f} MiB was allocated on non-default streams;"
            " those blocks follow a different lifetime regime."
        )
    print()


def report(snap: Snapshot, rep: Replay, label: str, top: int, max_entries: int) -> None:
    header(snap, rep, label, max_entries)

    if not rep.sites:
        print("no allocation events with stacks: record with stacks enabled.\n")
        return

    ranked = sorted(rep.sites.items(), key=lambda kv: -kv[1].live_at_peak)[:top]
    rows = [
        (
            f"{s.live_at_peak / MIB:10.1f}",
            f"{100 * s.live_at_peak / rep.peak_live:5.1f}%" if rep.peak_live else "-",
            f"{s.max_concurrent / MIB:10.1f}",
            f"{s.total_bytes / MIB:11.1f}",
            f"{s.n_allocs:,}",
            site,
        )
        for site, s in ranked
        if s.live_at_peak
    ]
    print("--- live at peak, by allocating call site")
    print(_table(rows, ("live@peak MiB", "share", "max conc MiB", "alloc'd MiB", "n", "call site")))

    churn = sorted(rep.sites.items(), key=lambda kv: -kv[1].total_bytes)[:top]
    rows = [
        (
            f"{s.total_bytes / MIB:11.1f}",
            f"{s.live_at_peak / MIB:10.1f}",
            f"{s.n_allocs:,}",
            _ms(s.median_lifetime_us),
            site,
        )
        for site, s in churn
        if s.total_bytes
    ]
    print("\n--- total allocated, by call site (median block lifetime)")
    print(_table(rows, ("alloc'd MiB", "live@peak MiB", "n", "median life ms", "call site")))

    if rep.never_freed:
        print("\n--- largest blocks allocated in the window and still live at the end")
        for size, site in rep.never_freed[:5]:
            print(f"  {size / MIB:10.1f} MiB  {site}")
    print()


def sections_for_allocs(snap: Snapshot, pattern: re.Pattern[str]) -> dict[int, str]:
    """Innermost enclosing `record_function` range per alloc event, by event index.

    Annotations are a flat, time-ordered list of START/END markers; nesting is recovered by
    replaying them as a stack and interleaving the allocations on `time_us`. At equal
    timestamps a START is applied before the allocation and an END after it, so a block
    allocated exactly on a boundary is counted inside the range.
    """
    ann = [a for a in snap.annotations if a.get("device", snap.device) == snap.device]
    if not ann:
        return {}
    merged: list[tuple[int, int, Any]] = [
        (a["time_us"], 0 if a.get("stage") == "START" else 2, a.get("name", "?")) for a in ann
    ]
    merged += [
        (e["time_us"], 1, i)
        for i, e in enumerate(snap.trace)
        if e.get("action") == "alloc" and "time_us" in e
    ]
    merged.sort(key=lambda t: (t[0], t[1]))

    out: dict[int, str] = {}
    stack: list[str] = []
    for _t, kind, payload in merged:
        if kind == 0:
            stack.append(str(payload))
        elif kind == 2:
            name = str(payload)
            if name in stack:  # tolerate the occasional unpaired END
                del stack[len(stack) - 1 - stack[::-1].index(name)]
        else:
            match = [s for s in stack if pattern.search(s)]
            if match:
                out[int(payload)] = match[-1]
    return out


def report_sections(snap: Snapshot, rep: Replay, pattern: re.Pattern[str], top: int) -> None:
    print("--- by record_function section")
    if not snap.annotations:
        print(
            "  this snapshot has no external_annotations. They are recorded only when the torch"
            " profiler runs alongside the memory recorder (profiling.pytorch_profiling), and"
            " `nn.Module: ...` ranges additionally need wrap_module_forward_with_profiling.\n"
        )
        return

    sections = sections_for_allocs(snap, pattern)
    if not sections:
        print(f"  no annotation matched /{pattern.pattern}/ around any allocation.\n")
        return

    alloc_bytes: Counter[str] = Counter()
    n_alloc: Counter[str] = Counter()
    covered_bytes = 0
    total_bytes = 0
    for i, event in enumerate(snap.trace):
        if event.get("action") != "alloc":
            continue
        size = int(event.get("size", 0))
        total_bytes += size
        name = sections.get(i)
        if name is None:
            alloc_bytes[OUTSIDE_SECTION] += size
            n_alloc[OUTSIDE_SECTION] += 1
            continue
        covered_bytes += size
        alloc_bytes[name] += size
        n_alloc[name] += 1

    peak_bytes: Counter[str] = Counter()
    for idx, size, _ in rep.peak_entries:
        peak_bytes[sections.get(idx, OUTSIDE_SECTION)] += size
    # keep the shares summing to the whole peak: what predates the window has no section
    if rep.pre_at_peak:
        label = PRE_PREFIX + "<no section: allocated before the trace window>"
        peak_bytes[label] = rep.pre_at_peak

    print(
        f"  matched /{pattern.pattern}/; covers {len(sections):,} of {rep.n_allocs:,} allocations"
        f" ({100 * covered_bytes / total_bytes:.0f}% of allocated bytes)."
        f" {rep.pre_at_peak / GIB:.3f} GiB of the peak predates the window and cannot be"
        " attributed to a section at all."
    )
    ranked = sorted(peak_bytes.items(), key=lambda kv: -kv[1])[:top]
    rows = [
        (
            f"{v / MIB:10.1f}",
            f"{100 * v / rep.peak_live:5.1f}%" if rep.peak_live else "-",
            f"{alloc_bytes.get(k, 0) / MIB:11.1f}",
            f"{n_alloc.get(k, 0):,}",
            k,
        )
        for k, v in ranked
    ]
    print(_table(rows, ("live@peak MiB", "share", "alloc'd MiB", "n", "section")))
    print()


def diff(after: Replay, before: Replay, top: int, with_line: bool) -> None:
    print("=== diff (after - before), sorted by change in bytes live at peak")
    print(
        "sites keyed by file:line — a code edit that shifts a line reads as a removal plus an"
        " addition\n"
        if with_line
        else "sites keyed by file:function, so the two code versions still line up\n"
    )
    sites = set(after.sites) | set(before.sites)
    rows = []
    for site in sites:
        a = after.sites.get(site, SiteStats())
        b = before.sites.get(site, SiteStats())
        d_peak = a.live_at_peak - b.live_at_peak
        d_total = a.total_bytes - b.total_bytes
        if not d_peak and not d_total:
            continue
        rows.append(
            (
                d_peak,
                (
                    f"{d_peak / MIB:+11.1f}",
                    f"{b.live_at_peak / MIB:10.1f}",
                    f"{a.live_at_peak / MIB:10.1f}",
                    f"{d_total / MIB:+12.1f}",
                    f"{b.n_allocs:,}->{a.n_allocs:,}",
                    site,
                ),
            )
        )
    rows.sort(key=lambda r: r[0])
    freed = [r[1] for r in rows if r[0] < 0][:top]
    grew = [r[1] for r in rows if r[0] > 0][-top:]
    ordered = freed + grew
    headers = ("d live@peak MiB", "before MiB", "after MiB", "d alloc'd MiB", "n allocs", "site")
    print(_table(ordered, headers))
    print(
        f"\ntotal peak live: {before.peak_live / GIB:.3f} -> {after.peak_live / GIB:.3f} GiB "
        f"({(after.peak_live - before.peak_live) / MIB:+.1f} MiB)"
    )
    if before.pre_window != after.pre_window:
        print(
            f"pre-window baseline moved {(after.pre_window - before.pre_window) / MIB:+.1f} MiB"
            " between the two runs; that part of the change is not attributable to a call site."
        )


def load(path: Path, device: int) -> Snapshot:
    with path.open("rb") as fh:
        raw = pickle.load(fh)
    traces = raw.get("device_traces") or []
    if device >= len(traces):
        raise SystemExit(f"{path}: no trace for device {device} (found {len(traces)})")
    return Snapshot(
        path=path,
        device=device,
        trace=traces[device],
        segments=[s for s in raw.get("segments") or [] if s.get("device", device) == device],
        annotations=list(raw.get("external_annotations") or []),
        settings=dict(raw.get("allocator_settings") or {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--diff", type=Path, help="baseline snapshot to compare against")
    parser.add_argument("--pkg", default="weathergen", help="package to attribute frames to")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--depth", type=int, default=3, help="project frames per call site (1 = innermost)"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--by-section",
        action="store_true",
        help="also group by record_function ranges from external_annotations",
    )
    parser.add_argument(
        "--section-pattern",
        default=".",
        help="regex selecting which annotation names count as sections (default: any)",
    )
    parser.add_argument(
        "--diff-exact-lines",
        action="store_true",
        help="key diff rows by file:line instead of file:function (only for identical code)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=100_000,
        help="max_entries the trace was recorded with; used to detect ring-buffer wrap",
    )
    args = parser.parse_args()

    snap = load(args.snapshot, args.device)
    after = replay(snap, args.pkg, args.depth)
    report(snap, after, args.snapshot.name, args.top, args.max_entries)
    if args.by_section:
        report_sections(snap, after, re.compile(args.section_pattern), args.top)
    if args.diff:
        base_snap = load(args.diff, args.device)
        before = replay(base_snap, args.pkg, args.depth)
        report(base_snap, before, f"{args.diff.name} (baseline)", args.top, args.max_entries)
        if args.diff_exact_lines:
            diff(after, before, args.top, with_line=True)
        else:
            # an ablation edits the code under measurement, so line numbers move between the
            # two snapshots; matching on file:function keeps the same site on one row
            diff(
                replay(snap, args.pkg, args.depth, with_line=False),
                replay(base_snap, args.pkg, args.depth, with_line=False),
                args.top,
                with_line=False,
            )


if __name__ == "__main__":
    main()
