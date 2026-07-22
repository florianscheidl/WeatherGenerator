# Decoder activation memory investigation

Tracking document for the activation-memory work on `Model.predict_decoders`
(`src/weathergen/model/model.py`) and the target prediction engines
(`src/weathergen/model/engines.py`). Part of the `exp/activation-checkpoint-ablation-*`
series; this branch is `exp/activation-checkpoint-ablation-decoder-nbor-gather`, based on
`b4e36782` ("Coarsen activation checkpointing in the stream embedders") and checked out in
`../WeatherGenerator.worktree.1`. It deliberately does **not** include
`exp/activation-checkpoint-ablation-avoid-casting-in-norm` (`e79d23ed`), so profiling
compares against the stream-embedder checkpointing state alone.

## Baseline accounting

Sizes below are for `config/default_config.yml` (healpix level 5 → `C = 12288` cells,
`ae_local_num_queries = 1`, `ae_global_dim_embed = 2048`, bf16 autocast) and an
ERA5-style stream (`embed_target_coords.dim_embed = 512`, `target_readout.num_layers = 2`
→ 6 checkpointed blocks, `max_num_targets = 20000`), per sample and per forecast step.

| tensor | site | size | retained until backward? |
| --- | --- | --- | --- |
| `tokens` | `model.py` (after aux-token slice) | 50 MB | yes (needed by the next step anyway) |
| gathered 1-ring neighbourhood | `model.py`, was `tokens_nbors` | **453 MB** (9×) | yes — before item 1a |
| block-boundary `tc_tokens` | `engines.py`, 6 checkpoints | 6 × 20.5 MB = 123 MB **per stream** | yes |
| `tc_tokens` out of `embed_target_coords` | `model.py` | 20.5 MB per stream | yes |
| `pred` | `model.py`, `EnsPredictionHead` | small (`ens_size: 1`, `num_layers: 1` in every shipped stream config) | yes, needed by the loss |

The gathered neighbourhood dominates: it is a plain gather produced outside any
checkpoint, and every checkpointed decoder block saves a reference to it as its
key/value input, so it lives from creation until that step's backward. Four output steps
at batch size 1 is ~1.8 GB of pure 9× replication.

## Measured outcome: the decoder is not the problem

A snapshot from run `gim27ey3` (post-1a), analysed with `scripts/memory_snapshot_report.py`,
puts peak live memory at **35.50 GiB** and attributes it as:

| what | live at peak | share |
| --- | --- | --- |
| stream-embedder **recompute** (`embeddings.py:115:<lambda> > …`, no enclosing `forward`) | 25.33 GiB | 71.3% |
| pre-window blocks still live at the peak (params, optimizer state, `_train_batch`) | 3.60 GiB | 10.1% |
| source tokens on device (`stream_data.py:155:to_device`) | 2.66 GiB | 7.5% |
| `<no python stack: backward or non-Python thread>` | 1.32 GiB | 3.7% |
| positional encoding (`positional_encoding.py:33`) | 1.11 GiB | 3.1% |
| everything else attributed to a `weathergen` call site | 1.49 GiB | 4.2% |

Earlier revisions of this table read 31.9 GiB peak with a 75.5% recompute share. Those came
from a replay that started from an empty allocator and so measured a delta over the trace
window rather than device memory; the report is now anchored on the dump-time segment state
(see "Measuring"). The ranking is unchanged, the denominator grew, and 3.60 GiB that used to
be invisible is now on the books.

Two groups share `embeddings.py:115`. The three-frame one
(`embeddings.py:115:forward > …:<lambda> > layers.py:86`) is the forward pass: 88 GiB
allocated, **0 live at peak**, median lifetime 9 events — checkpointing working. The
two-frame one is the same lambda invoked without `StreamEmbedTransformer.forward` on the
stack, i.e. the autograd engine's **recompute**: 78 GiB allocated, 24 GiB live at the peak
instant.

That is the cost side of `b4e36782` ("Coarsen activation checkpointing in the stream
embedders"): coarsening trades retained boundaries (1.26 GiB, cheap) for a larger
simultaneous recompute set (24 GiB). On this evidence the embedder wants *finer*
granularity, not coarser — the next experiment worth running, and the opposite direction
to the last step in this series.

1a behaved as designed — all three `_block_with_gathered_kv` rows show `live@peak 0.0`, so
the deferred gather retains nothing — but the decoder holds 0.2% of peak, so items 2–4
below are worth ~0.25% between them and are not worth implementing for memory.

Caveat: the trace still hits the 100k-event cap, so it covers only the last 4.03 s of the
run. That no longer distorts the totals — anchoring makes them reconcile with the dump to
±0.000 MiB — but 9.93 GiB was allocated before the window opened and freed inside it, and
that memory can never be attributed to a call site. Widening the window
(`active_iteration: 1`, `wait_iteration: 0`) is still worth doing before acting on the
per-site split.

## Status

| # | item | status |
| --- | --- | --- |
| 1a | gather the 1-ring neighbourhood inside the decoder's per-block checkpoints | **implemented** (`62de7981`); verified to retain nothing, but worth ~0.2% of peak |
| 1b | one coarse checkpoint per stream around embed → tte → head | dropped — decoder is 0.2% of peak |
| 2 | coarsen block granularity inside `TargetPredictionEngineClassic` | dropped — ~0.25% of peak |
| 3 | skip the neighbourhood setup when it cannot be used | dropped — negligible after 1a |
| 4 | drop the no-op `checkpoint` around linear coord embeddings | dropped — negligible |
| 5 | batch offset missing from the neighbour gather index | **open — correctness, not memory** |
| 6 | `tokens_cells` copy when `num_aux_tokens > 0` | dropped — negligible |
| 7 | stream-embedder checkpoint granularity: revisit `b4e36782`, try finer | **open — 71% of peak** |
| 8 | `source_tokens_cells` held on device (`stream_data.py:155`) | **open — 7.5% of peak** |

Next action: re-run the trace untruncated (`active_iteration: 1`, `wait_iteration: 0`),
then A/B the stream-embedder checkpoint granularity (item 7). Item 5 stands on its own as
a correctness fix regardless of the memory work.

## 1a — defer the neighbourhood gather into the block checkpoints (implemented)

`predict_decoders` no longer materializes `tokens_nbors`. It passes the ungathered
`tokens_cells` (`[B*C, Q, D]`) plus `nbors_idxs` into the target prediction engine, and
`_block_with_gathered_kv` performs the gather *inside* each cross-attention block's
existing checkpoint. Backward therefore saves `tokens_cells` — alive regardless — instead
of the 9× larger gathered tensor.

- Expected saving at the peak: ~453 MB × batch size × **(output steps − 1)**. The
  retained bytes drop by one copy per output step, but the new code still materializes
  one copy transiently inside each cross-attention block, so at any instant where the old
  code held one copy the new code holds one too. Only the accumulation across output
  steps is removed — with a single output step the peak does not move at all.
  (`default_config.yml` has `forecast.num_steps: 2`, i.e. one copy's worth.)
- Expected cost: the gather runs a second time per cross-attention block during
  recompute (~0.3 ms per gather at GH200 HBM bandwidth). The gather's backward
  scatter-add now runs once per cross-attention block rather than once per step, but the
  total bandwidth is roughly unchanged — before, each block's backward accumulated into a
  453 MB gradient buffer and one scatter-add followed.
- MLP blocks take the else-branch of the block loop with the same 5-argument call, but
  `MLP.forward(*args)` only reads `args[0]` and `args[-1]`, so they are passed
  `latent_idx = None` and never trigger a gather.
- Bit-exactness: verified on CPU that forward output and the gradient w.r.t. the
  pre-slice latent are `torch.equal` to the old formulation, through the checkpoint
  recompute path, and that the `latent_idx = None` branch is an identity passthrough.
  The batch-offset bug (item 5) is deliberately preserved so the ablation stays a pure
  memory refactor.
- `TargetPredictionEngine` (all `decoder_type` values other than
  `PerceiverIOCoordConditioning` and `Linear`) conditions on the gathered neighbourhood
  itself — `pos_embed` is `[1, 9, D]` and the aux is `latent[:, 0]` — so it cannot defer
  the gather. It materializes the neighbourhood at the top of its `forward` instead,
  reproducing exactly what it received before. That path is dead code today: `model.py`
  constructs it with `stream_config=si` while its `__init__` takes `stream_name`, which
  raises `TypeError`, and it was already being handed a 2-D tensor that its `pos_embed`
  broadcast cannot accept.

1b (a single coarse checkpoint per stream, wrapping the gather together with embed → tte
→ head) would additionally collapse rows 3 and 4 of the baseline table, but pays a full
extra decoder forward and re-nests the inner checkpoints, so the blocks would run three
times. Not pursued while 1a covers the dominant term.

### Reading the 1a trace

The first memory trace showed roughly no change in the memory allocated inside
`predict_decoders`. That is consistent with the change working — the metric does not see
it — so check these before concluding 1a is a dud:

1. **Is the method live at all?** `pred_heads` is only populated when `LossPhysical` is in
   the training or validation losses, and `predict_decoders` returns immediately when it
   is empty. On JEPA-only configs (`config_jepa.yml`, `config_performance_jepa.yml`) the
   whole section is a no-op.
2. **How many output steps?** The saving is one copy per output step beyond the first
   (see above). At `forecast.num_steps: 1` there is nothing to save.
3. **What the metric measures.** memory_viz attributes blocks to the stack that
   *allocated* them, and the gather still allocates inside `predict_decoders` — now twice
   per cross-attention block rather than once in total. Allocation volume attributed to
   the method is therefore unchanged or slightly higher; what changes is the **lifetime**
   of those blocks (several short bars ending inside the method, instead of one long bar
   spanning to backward). `MemoryTracker` reports per-step peaks, so it does not isolate
   the method either.

The direct measurement is the live-memory delta across the call, which measures retention
rather than allocation:

```python
before = torch.cuda.memory_allocated()
output = self.predict_decoders(model_params, step, tokens, batch, output)
logger.info(f"step {step}: retained {(torch.cuda.memory_allocated() - before) / 2**20:.1f} MiB")
```

The mechanism itself is verified on CPU with the same block structure: retained bytes
after forward drop from 18.51 MiB to 0.51 MiB for an 18 MiB gathered tensor, i.e. exactly
the 9× copy stops being held.

## 2 — coarsen block granularity in `TargetPredictionEngineClassic` (open)

Each of the 6 blocks per stream (cross-attention, self-attention, MLP × 2 layers) is
checkpointed individually, so 6 boundary tensors of `[N_targets, 512]` are retained.
Checkpointing the (cross, self, MLP) triple jointly retains 2 instead of 6: ~82 MB per
stream per step, with no extra recompute — the same trade already taken for the stream
embedders in `embeddings.py` (`b4e36782`). The transient peak during backward grows by
the triple's internals.

## 3 — skip the neighbourhood setup when it cannot be used (open)

The neighbourhood index and lens are built unconditionally, before the stream loop, before
the `len(t_coords) == 0` skips, and even when `decoder_type == "Linear"` — where the Linear
branch never touches them. After 1a this is only the index and lens tensors rather than
453 MB, so the remaining win is small; still free and risk-free.

## 4 — no-op `checkpoint` around linear coordinate embeddings (open)

Every shipped stream config uses `embed_target_coords.net: linear`, i.e. `NamedLinear`. A
`Linear` backward needs exactly its input, `t_coords`, which the checkpoint saves anyway —
and `t_coords` stays alive regardless as the AdaLN aux passed to every decoder block. So
the checkpoint recomputes for zero saving. Worth making conditional on the `mlp` variant,
where `hidden_factor = 8` means a real 8×dim hidden activation per target token. Same
reasoning as the note at `embeddings.py:156`.

## 5 — missing batch offset in the neighbour gather index (open, correctness)

`model_params.hp_nbours` holds cell ids in `[0, C)`. `predict_decoders` builds the gather
index as `hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)`, i.e. without
adding a `b * C` offset, but indexes a tensor whose row index is `b * C + c`. For
`batch_size > 1` every sample's neighbourhood is therefore gathered from **sample 0's**
latent state. Benign at `batch_size_per_gpu == 1`.

Fixing it as `tokens.reshape(B, C, D)[:, hp_nbours]` gives the correct per-sample gather
and drops the materialized index, but it changes numerics for `batch_size > 1`, so it is
kept out of the memory ablation. Land it separately.

## 6 — `tokens_cells` copy (open, minor)

`tokens.reshape(s).flatten(0, 1)` is a view only when `num_aux_tokens == 0`; with register
or class tokens present the merge of dims 0 and 1 across the sliced stride forces a copy
(~50 MB per sample per step, now retained as a saved input). Folding the aux-token offset
into the gather index would avoid it. Pre-existing — the old code made the same copy — and
9× smaller than what 1a removed.

## Not worth touching

`EnsPredictionHead` is not checkpointed, but every shipped stream config has
`ens_size: 1` and `num_layers: 1`, so it is a single `Linear` whose input `tc_tokens` is
retained regardless. Only becomes a target if ensemble size or head depth grows.

## Measuring

`config/config_memory_profiling.yml` (allocator snapshot only); load the resulting
`.pickle` into <https://pytorch.org/memory_viz>. Before 1a the gathered neighbourhood shows
up as a long-lived ~453 MB allocation spanning each forecast step — that is the one that
should disappear. Invocation details in `agent_docs/performance-profiling.md`.

For a text summary that does not need memory_viz — and that an agent can read — use
`scripts/memory_snapshot_report.py` (stdlib only, no torch needed). Full usage guide, flags
and the annotation-family table: `docs/memory-profiling-levels.md`, "Using
scripts/memory_snapshot_report.py". The essentials for this investigation:

```
scripts/memory_snapshot_report.py after.pickle --diff before.pickle
```

It replays the trace and attributes every block to the innermost `weathergen` frames that
allocated it, reporting per call site: bytes **live at the high-water mark** (who retains),
bytes **allocated in total** (who churns), and median block lifetime in ms (transient vs
retained to backward). `--diff` prints the per-call-site delta between two runs, which is the
number an ablation needs; it keys sites by file:function rather than file:line, because the
edit under test moves line numbers and would otherwise show every site as a removal plus an
addition (`--diff-exact-lines` restores line-level keys).

The replay is **anchored**: the trace is a ring buffer of the last `max_entries` events, so it
opens with the allocator already full (13.53 GiB in `gim27ey3`), and replaying from zero
measures a delta rather than a peak. The tool reconstructs the opening balance from the live
block set in `snapshot["segments"]` and prints a reconciliation line — reconstructed end state
versus the segments — which must read ±0.000 MiB. Anything else means the arithmetic did not
close and the absolute numbers should not be trusted. Blocks that predate the window are shown
as `[pre-window]` rows, attributed through the frames the allocator kept with each live block,
or as `<allocated before recording started>` where recording began after the allocation.

`--by-section` groups by `record_function` ranges (`snapshot["external_annotations"]`), joined
to allocations on `time_us`: `nn.Module: model.…` when `wrap_module_forward_with_profiling` is
active, plus `FSDP::*`, `nccl:*`, `ProfilerStep#N` and `Optimizer.step#…`. Use
`--section-pattern` to pick the family, e.g. `--section-pattern 'nn.Module'`. This is the only
attribution that reaches backward-pass allocations, which carry no Python stack at all.

This distinction is the whole point for checkpointing work: moving an allocation inside a
checkpoint leaves the *allocation* where it was (it now happens twice — forward and
recompute) and only shortens its *lifetime*. On paired synthetic traces of the 1a change,
the report shows `model.py:769` losing its full retained size while the engine's call site
gains a transient copy and its total allocation goes **up** — a net peak reduction that
looks like "no change" if you only compare allocation volume per section.

Recording tips: `--depth` controls how many project frames identify a call site. The
default of 3 matters here — the encoder, forecast engine and decoder all funnel into the
same few `attention.py` lines, so attributing to the innermost frame alone merges them.
A trace holding exactly `max_entries` events (the report says so) wrapped the ring buffer and
covers only the tail of the run: the totals stay correct, but attribution for anything older
is lost, so profile fewer steps to widen the window.

About 30% of allocated bytes carry **no Python frames at all** (10.4k allocations per trace in
both snapshots checked, including 2 GiB blocks). Those come from threads without a Python
stack — in practice the autograd engine's backward thread — and no `stacks` setting recovers
them, since no snapshot from this build contains a single C++ frame. `--by-section` is the way
to attribute them.

`start_record_memory_history()` now passes `stacks="python"` explicitly, but note this is
a no-op in practice on the GH200 build: observed snapshots already contain Python frames
only under torch's `stacks="all"` default, presumably because C++ unwinding is unavailable
there. Keep the explicit value for determinism, not for any size or clarity win.

Local verification limits: `./scripts/actions.sh unit-test` and `uv run --extra cpu` do not
resolve on macOS (the lockfile is `aarch64` + `linux` only), and `flash_attn_interface` is
absent, so `weathergen.model.attention` cannot be imported locally. Lint and targeted CPU
equivalence checks are the only local gates; anything touching attention has to run on the
cluster.
