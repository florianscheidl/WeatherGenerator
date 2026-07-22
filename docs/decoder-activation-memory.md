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

## Status

| # | item | status |
| --- | --- | --- |
| 1a | gather the 1-ring neighbourhood inside the decoder's per-block checkpoints | **implemented**, unprofiled |
| 1b | one coarse checkpoint per stream around embed → tte → head | not planned (see below) |
| 2 | coarsen block granularity inside `TargetPredictionEngineClassic` | open |
| 3 | skip the neighbourhood setup when it cannot be used | open |
| 4 | drop the no-op `checkpoint` around linear coord embeddings | open |
| 5 | batch offset missing from the neighbour gather index | open — correctness, not memory |
| 6 | `tokens_cells` copy when `num_aux_tokens > 0` | open, minor |

Next action: profile 1a, then decide on 2–4.

## 1a — defer the neighbourhood gather into the block checkpoints (implemented)

`predict_decoders` no longer materializes `tokens_nbors`. It passes the ungathered
`tokens_cells` (`[B*C, Q, D]`) plus `nbors_idxs` into the target prediction engine, and
`_block_with_gathered_kv` performs the gather *inside* each cross-attention block's
existing checkpoint. Backward therefore saves `tokens_cells` — alive regardless — instead
of the 9× larger gathered tensor.

- Expected saving: ~453 MB × batch size × forecast steps.
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

Local verification limits: `./scripts/actions.sh unit-test` and `uv run --extra cpu` do not
resolve on macOS (the lockfile is `aarch64` + `linux` only), and `flash_attn_interface` is
absent, so `weathergen.model.attention` cannot be imported locally. Lint and targeted CPU
equivalence checks are the only local gates; anything touching attention has to run on the
cluster.
