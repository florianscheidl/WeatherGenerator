# Memory profiling levels — design

Plan for making memory profiling readable without memory_viz, on branch
`dev/agent-legible-memory-profiling`. Motivation: a checkpointing change that removes a
453 MB retained activation is invisible in an allocator snapshot, because the allocation
still happens (twice — forward and recompute) and only its *lifetime* changes. See
`docs/decoder-activation-memory.md`, "Reading the 1a trace".

## The ladder

One cumulative knob, `profiling.memory_level`, each level adding to the one below:

| level | name | what it gives | scope | cost |
| --- | --- | --- | --- | --- |
| 0 | `off` | nothing | — | none |
| 1 | `window` | peak allocated/reserved per `train_logging.metrics` window | whole run, all ranks | ~2 calls per window |
| 2 | `sections` | retained + peak bytes per named code section | whole run, all ranks | ~2 calls per section entry |
| 3 | `trace` | CUDA allocator snapshot (`.pickle`) | opening iterations, rank 0 | high — perturbs, capped at 100k events |

Level 1 already exists: `MemoryTracker` in `utils/performance.py`, gated by
`train_logging.memory_tracking`. Level 3 already exists: `profiling.memory_profiling`.
**Level 2 is the new work**; the rest of this document is about it.

### Config surface

Add `profiling.memory_level: off | window | sections | trace`, and derive the existing
flags from it rather than adding a fourth independent boolean:

- `train_logging.memory_tracking` ← `level >= window`
- `profiling.memory_profiling` ← `level >= trace`

Keep both legacy keys working (if either is set explicitly, it wins and logs a
deprecation note), because `config_memory_profiling.yml`, `config_pytorch_profiling.yml`
and the private launch configs set them today. Note `get_trainer()` selects
`ProfilingTrainer` from `memory_profiling or pytorch_profiling`, so the derivation must
happen before that call.

## Level 2: section memory profiling

### API

```python
from weathergen.utils.performance import memory_section

with memory_section("predict_decoders", step=step):
    ...
```

Returns a shared no-op context manager when disabled, so call sites stay unconditional
and cost one attribute lookup in normal runs. Model code must not import the trainer, so
enablement lives in a module-level singleton that `Trainer.init` configures.

### What it records

Per section entry, using the caching allocator's host-side counters (no CUDA sync, so
unlike the torch profiler this does not distort what it measures):

- `retained` = `memory_allocated()` after − before. **The number that matters for
  activation work**: bytes the section leaves behind for backward.
- `peak` = max live inside the section.
- `n_calls`.

Aggregated per metrics window and logged as
`performance.memory.section.<name>.{retained,peak}_mib` alongside the existing
`performance.memory.*` keys, so both land in the same record.

### The one hard constraint: who owns `reset_peak_memory_stats`

`torch.cuda.reset_peak_memory_stats()` and `max_memory_allocated()` are **device-global**.
Measuring a per-section peak requires resetting, and `MemoryTracker.collect()` already
resets once per window. If both reset, level 1's window peak silently becomes the peak of
whatever fragment of the window happened to come last — an understated number that still
looks plausible. This is the main thing to get right.

Resolution: a single owner of the counter. The section profiler records the peak
immediately *before* every reset it performs and folds it into a running window maximum;
`MemoryTracker` reads that maximum instead of calling `max_memory_allocated()` itself when
level 2 is active. The composition is exact — the peak over a window is the max over its
sub-intervals, and each sub-interval's peak is by definition ≥ the live bytes at its start,
so no crossing peak can be missed.

### Nesting

Keep an explicit stack. On exit: record the inner peak, reset, and fold the inner peak
into the parent frame's pending peak, so an outer section's peak still accounts for
everything its children did. Assert a maximum depth (2 is enough for the granularity
below) to keep the bookkeeping honest and cheap.

### No collectives

Do **not** reduce per section — a cross-rank MAX in the hot path inserts a sync barrier
per section per step, which is exactly what the sync-barrier workstream is removing.
Aggregate locally and reduce once per window, at the existing `MemoryTracker.collect()`
call, or log rank 0 only. Section names must then be identical on every rank, which rules
out deriving names from data present in a given batch.

### Granularity

The rules that keep this useful: **stable names across steps** (so windows aggregate),
**O(10) entries per step, not O(1000)** (so overhead and log volume stay flat), and **at
most 2 levels of nesting**.

Default (`coarse`) — 8 sections:

| section | where |
| --- | --- |
| `train_batch.forward` | `Trainer._train_batch` |
| `train_batch.loss` | " |
| `train_batch.backward` | " |
| `train_batch.optimizer` | " |
| `model.encoder` | `Model.forward` |
| `model.forecast_engine[step]` | `Model.forward`, per output step |
| `model.predict_decoders[step]` | " |
| `model.predict_latent[step]` | " |

The forecast step belongs in the key: retention differs per step, and "does the decoder
retain one copy per step or one in total" is precisely the question 1a raised.

Optional `profiling.memory_sections.detail: fine` adds, inside `predict_decoders`, one
section per stream (`model.predict_decoders[step].<stream>`) and one per engine block —
enough to isolate a single checkpoint boundary, at ~50 entries per step. Off by default.

### Known limitation: forward only

A context manager in `forward` cannot see the backward pass, where the true peak usually
sits. Level 2 answers "what did this section leave behind", which is the activation
question; "when was the global peak" stays with level 1, and per-allocation detail with
level 3. Instrumenting backward would need autograd hooks — out of scope, worth a note in
the docstring so nobody reads a section peak as a global peak.

## Level 3 integration

- **Done, but a no-op on GH200**: `stacks="python"` in `start_record_memory_history`.
  Snapshots from that build already contained Python frames only under torch's
  `stacks="all"` default — C++ unwinding appears to be unavailable there — so this buys
  determinism, not a smaller pickle. Do not count it as a win. Confirmed against two
  snapshots (Jun 09 with `stacks="all"`, Jul 22 with `stacks="python"`): every frame in
  both has a `.py` filename, and both carry ~10.4k frameless allocations.
- **Done**: `--depth` in `scripts/memory_snapshot_report.py`. Attribution by innermost
  frame alone is not usable in this codebase: encoder, forecast engine and decoder all
  reach the same `attention.py` lines. Sites are now identified by a chain of project
  frames (default 3).
- **Done**: the replay is anchored on the dump-time live set from `snapshot["segments"]`.
  The trace is a ring buffer of the last `max_entries` events, so it opens with the
  allocator already holding several GiB; replaying from zero reported a within-window
  delta as if it were the peak (31.9 GiB where the real peak was 35.5 GiB, with 3.6 GiB of
  live blocks missing entirely). The report now prints a reconciliation line that must read
  ±0.000 MiB.
- **Done, and it needed no new flag**: `--by-section`. `global_record_annotations` turned
  out to be beside the point — snapshots already carry `external_annotations`, a flat
  START/END list of `record_function` ranges with `time_us`, and every trace event carries
  `time_us` too, so the two join directly. Coverage on the Jun 09 snapshot: 91% of
  allocations sit inside some annotation, 40% inside an `nn.Module:` range. Caveats: the
  annotations exist only when the torch profiler runs alongside the memory recorder,
  `nn.Module:` ranges need `wrap_module_forward_with_profiling`, and only the part of the
  annotation stream that overlaps the trace window is usable (3.5k of 19k entries on Jul 22).
- **Still the case for level 2**: sections in the *forward* code do not cover backward,
  which is where the frameless ~30% of allocated bytes comes from. The FSDP annotations
  (`FSDP::pre_backward`, `post_backward_*`) are the only backward-phase labels available
  today.

## Using `scripts/memory_snapshot_report.py`

Reads a level-3 snapshot pickle and prints what memory_viz makes you eyeball. Stdlib only —
no torch, so it runs on a laptop against a pickle copied off the cluster.

### Producing the snapshot

`config/config_memory_profiling.yml`. Recording is **rank 0 only** and covers the opening
`(wait + warmup + active) * repeat` training steps (`trainer.py:_profile_opening_iterations`);
the pickle lands in `profiling/<run-id>/<timestamp>_rank_0.pickle`. The shipped 6 steps
overflow the 100k-event ring buffer — set `wait_iteration: 0, active_iteration: 1` when you
want the whole recorded window to fit in the trace. Keep `data_loading.rng_seed` fixed so the
arms of an ablation see identical token counts.

### The two commands

One snapshot — rank the retention targets:

```
python3 scripts/memory_snapshot_report.py profiling/<run-id>/<file>.pickle
```

Two snapshots — what an ablation actually changed (this is the one that matters):

```
python3 scripts/memory_snapshot_report.py after.pickle --diff before.pickle
```

The script has a `uv run --script` shebang but is not executable in the repo; invoke it
through `python3` (>=3.12), or `chmod +x` it first.

### Reading the header, in order

```
at dump: 7.123 GiB allocated, 39.174 GiB reserved (32.051 GiB held by the caching allocator)
trace window: 100,000 events, 4.03 s (alloc 32,285 / free 33,858)
live when the window opens: 13.530 GiB (3.597 GiB still live at dump ..., 9.933 GiB freed ...)
peak live: 35.504 GiB at event 87,877, t+3.47 s into the window
  of which 3.597 GiB (10%) was allocated before the window and was still live at that instant
  reconciles with dump-time segments to +0.000 MiB (end state 7.121 GiB)
```

1. **The reconciliation line must read ±0.000 MiB.** It is the proof that the anchored replay
   closed against the allocator's own end state. Anything else means the absolute numbers are
   wrong, and the report says so instead of printing the confirmation.
2. **The pre-window share** is how much of the peak the trace cannot explain. 10% is fine; at
   40% widen the window before concluding anything about the split.
3. **A truncation NOTE** (trace holds exactly `max_entries` events) degrades attribution for
   older allocations, not the totals.

Reserved minus allocated is caching-allocator overhead under `expandable_segments:True`, not
a leak.

### The two tables answer different questions

- **live at peak, by call site** — who *retains*. This is what decides whether you OOM; rank
  targets by it.
- **total allocated + median life ms** — who *churns*. Huge `alloc'd`, ~0 `live@peak` and a
  sub-ms lifetime is checkpointing working. A lifetime spanning to backward is a target.

Moving an allocation into a checkpoint leaves the allocation where it was — it now happens
twice, forward and recompute — and only shortens its lifetime, which is why the churn table
alone reads as "no change".

Rows that are not ordinary call sites:

| row | meaning |
| --- | --- |
| `[pre-window] <site>` | allocated before the trace window, attributed through the frames the allocator kept with the still-live block |
| `[pre-window] <allocated before recording started>` | live at dump with no frames at all |
| `<no python stack: backward or non-Python thread>` | autograd's backward thread; ~30% of allocated bytes, unreachable by any `stacks` setting on this build |

### `--by-section` needs the torch profiler

Sections come from `record_function` ranges, so which families exist depends on the config:

| config | families present | pattern to use |
| --- | --- | --- |
| `config_memory_profiling.yml` (`pytorch_profiling: False`) | `FSDP::*`, `nccl:*`, `Optimizer.step`, `enumerate(DataLoader)` | `--section-pattern 'FSDP::'` |
| plus `pytorch_profiling: True` | adds `nn.Module: model....`, `ProfilerStep#N` | `--section-pattern 'nn.Module'` |

```
python3 scripts/memory_snapshot_report.py snap.pickle --by-section --section-pattern 'nn.Module'
```

The trade-off is real: the torch profiler wraps every custom module's forward and perturbs
what is being measured. Use `--by-section` for attribution — it is the only view that reaches
backward-pass allocations — and the plain memory config for absolute numbers.

### Flags

| flag | when to change it |
| --- | --- |
| `--depth 3` | frames per call-site chain; raise to split sites that still merge, drop to 1 to aggregate |
| `--top 20` | rows per table |
| `--pkg weathergen` | which package counts as project frames |
| `--device 0` | only rank 0 records, and its trace is device 0 — rarely needed |
| `--max-entries 100000` | must match `MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT` or wrap detection misfires |
| `--diff-exact-lines` | key diff rows by file:line; only valid when both arms ran identical code |

`--diff` keys sites by **file:function** by default: an ablation edits the code under
measurement, so line numbers move and line-keyed rows show every site as a removal plus an
addition. Watch the closing `pre-window baseline moved ...` line — that part of a delta is not
attributable to any call site.

## Implementation order

1. `utils/performance.py`: `SectionMemoryProfiler` (stack, per-section aggregation, peak
   ownership) + module-level `memory_section()`. Inject the counter reads behind a small
   protocol so it is unit-testable on CPU — CI has no GPU.
2. Extend `tests/test_performance_utils.py`: nesting, peak composition against a fake
   allocator, no-op path, and the level-1/level-2 interaction (window peak must equal the
   max of the section peaks).
3. Wire enablement in `Trainer.init`; have `MemoryTracker` consume the arbiter's peak.
4. Add the 8 coarse call sites.
5. Config: `profiling.memory_level` + back-compat derivation; update
   `config_memory_profiling.yml` to use it.
6. Measure the overhead of the coarse set against an uninstrumented run before adding
   `fine`.
7. Docs: `agent_docs/performance-profiling.md` (context repo — needs a `context fork`
   check first) and this file.

## Risks

- **Peak-counter contention** (above) — the one correctness trap; covered by test 2.
- **Overhead** — two allocator queries per section entry; trivial at 8 sections per step,
  needs measuring before `fine` is recommended.
- **Rank divergence** — section names must not depend on which streams a rank's batch
  happened to contain, or a later cross-rank reduction will mismatch.
- **Config precedence** — the derivation must run before `get_trainer()`, and must not
  silently override an explicitly set legacy key.
