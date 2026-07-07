# Training-memory estimator

Interactive, self-contained HTML tool that estimates **per-GPU memory over one training
step** as a function of the model/config parameters — a phase-by-phase timeline with the
separate contributions of activations, parameters, gradients, and optimizer states — so the
impact of config changes on memory can be explored without launching a run.

```
open tools/memory-estimator/index.html    # no server or dependencies needed
```

Defaults reproduce `config/config_operan_georing_avhrr_forecasting_lowres.yml` with the
`operan_georing_avhrr_synop_lowres` streams (8 embedding streams, 2 decoded streams). The
"Reset" button restores these defaults.

## What it shows

A stacked step-area chart over the phases of one training step:

```
baseline → fwd embedders → fwd assimilate_local → fwd global AE → Σ steps
→ [fwd FE s → fwd decode s] per forecast step → loss
→ backward through the same regions in reverse → optimizer step → end
```

with four tracked categories (plus fixed overhead), a GPU-budget line, a peak marker, a
hover tooltip and a per-phase table. **Activation checkpointing and FSDP are modeled and
toggleable.**

## Streams (per-stream data, from the YAMLs)

The `STREAMS_DEF` table in `index.html` encodes each stream of
`config/streams/operan_georing_avhrr_synop_lowres/` individually and is editable in the UI:

| stream | role | token_size | embed dim | keep rate |
|--------|------|-----------:|----------:|----------:|
| ERA5_in | forcing (encoder only) | 8 | 512 | 0.1 |
| ERA5 (out) | diagnostic (decoder only, etc dim 512, 68 target chan) | – | – | – |
| AVHRR/IASI | forcing | 512 | 256 | 1.0 |
| SEVIRI / GOES / Himawari IR+VIS (5×) | forcing | 1024 | 512 | 1.0 |
| SYNOP | in+out (etc dim 512, 7 target chan) | 64 | 512 | 0.1 |

Key semantics (verified in `datasets/masking.py:520-548`): the per-stream
`masking_override…rate` is a **keep rate** (`mask = rng.uniform < keep_rate`, True = keep) —
satellites keep all source tokens, ERA5_in and SYNOP contribute only ~10% of their raw
tokens as model input. Forcing streams have no decoder; diagnostic streams have no
embedder (`engines.py:54-55`); embed attention sequence length = data + geoinfo channels;
values per token = `token_size × channels` (drives input-data bytes).

**Editable vs. fixed**: token_size, embed/etc dims, roles, keep rates come from the config
(keep rates still editable for what-ifs). Per-stream **raw source tokens** (`kTok`) and obs
**channel counts** are data-dependent (live in the zarr, not the config) — the defaults are
rough guesses and the first thing to calibrate against a real run. Target counts: ERA5 out
= 40.3k (o96 grid), SYNOP a guess. A global `tokenScale` knob scales all token counts for
quick what-ifs. Streams can be toggled on/off individually.

## The model

Fully analytic (no torch, no measurement). Token counts, with `C = 12·4^level` cells,
batch `B`, input steps `S`, queries `Q`, unmasked fraction `f`:

| symbol  | meaning | formula |
|---------|---------|---------|
| `Nloc`  | local (observation) tokens | `B·S·Σ_streams kTok·keep` |
| `NgAll` | global latent tokens | `B·S·C·Q` |
| `Nfe`   | latent after summing input steps | `B·C·Q` |
| `Nnb`   | decoder 1-ring neighborhood kv | `B·C·9·Q` |
| `Nt_s`  | target points per forecast step | per decoded stream |

### Checkpoint regions (mirroring the code)

| region | checkpoint structure | consequence |
|--------|---------------------|-------------|
| embed engine | outer ckpt (`encoder.py:137`) + nested per-layer ckpts (`embeddings.py:107-110`) | fwd stores only output; bwd stores all layer *inputs* + one layer's internals |
| `assimilate_local` | **one** outer ckpt (`encoder.py:141`), **no inner ckpts** | bwd rematerializes the *entire* region internals at once — local AE + adapter + aggregation. This is typically the peak. |
| global AE | outer ckpt (`encoder.py:145`) + per-block ckpts (`engines.py:572`) | bwd: block inputs + one block's internals |
| Σ over input steps | not checkpointed (`model.py:696`) | glue tensor stored |
| FE (per fstep) | per-block ckpts (`engines.py:677-679`) | fwd stores every block input (`(2·blocks+1)·Nfe·D`); bwd adds one block's internals |
| decoder (per fstep) | per-block ckpts (`engines.py:840-869`); `tokens_nbors` gather (`model.py:774`) and pred heads NOT checkpointed | nbors gather (`9×` the latent!) and head internals stored in fwd |

Per-region we define three element counts (see `regions` in `simulate()`):
`stash` (stored during forward when ckpt on), `internals` (live during forward when ckpt
off), `repeak` (extra materialized during that region's backward when ckpt on). Stored-
tensor multipliers per block — attention 7, MLP `2+2h` — are tunable in "advanced";
a region's backward also adds `gradActFrac` (default 0.15) × its recompute as the
activation-gradient transient.

Timeline mechanics: forward accumulates `stash` (or `internals` when ckpt off); each
backward phase shows remaining stash + that region's `repeak` + transient, then frees the
region's stash. Predictions of *all* forecast steps stay resident until the loss (matches
`ModelOutput` accumulation).

### Parameters / gradients / optimizer states

- Per-module parameter formulas: `4D²` per attention, `2hD²` per MLP,
  `Dq·P + 2·Dkv·P + P·Dq` per cross-attention (adapter `P = heads·head_dim`, decoder
  `P = dDec`); embedders and heads as in `simulate()`. Check the table against
  `Model.print_num_parameters()` (`model.py:592`).
- Master weights fp32, AdamW m+v fp32, optional EMA copy (`validate_with_ema`), grads fp32.
- **FSDP on**: master/optimizer/EMA and resident grads divided by `world_size`; plus
  transients: unsharded bf16 compute copies of the active (+ prefetched, knob) units
  during their fwd/bwd phase, and unsharded grads of the unit currently in backward until
  its reduce-scatter. Param groups = {embedders, assimilate_local(+queries), global AE,
  FE, decoders}; FE is re-unsharded every forecast step.
- Gradients become resident (sharded) at the **first backward touch** of a param group —
  so FE grads appear after `bwd FE <last step>`, embed grads last. All freed after the
  optimizer step (`zero_grad(set_to_none=True)` assumed).

### Sanity values at lowres defaults (W=1, ckpt on)

~916 M params (FE 537 M, local/global AE 134 M each); ~15k effective local tokens (after
keep rates), ~55k targets/step; baseline ~16.5 GB (optimizer states + EMA dominate at
10.2 GB); peak ≈ 24.5 GB at `bwd assimilate_local`. With `tokenScale = 10` the peak grows
to ~61 GB, still at `bwd assimilate_local` — activation terms scale with data volume while
the state floor is fixed by the 916 M params.

## Known gaps / roadmap

1. **Calibration against measurement** — compare with `torch.cuda.max_memory_allocated()`
   / the traces in `profiling/`. The dominant unknowns are now the per-stream raw token
   counts (`kTok`) and obs channel counts (zarr-derived, not in config); then the
   multipliers `attnMult`, `mlpExtra`, `asmMult`, `gradActFrac`.
2. **Config import** — paste a run config + streams YAML and auto-populate inputs and the
   stream table (the stream table is currently hand-extracted from the lowres set).
3. **Chunking in `assimilate_local`** — the 2-chunk (level ≤ 5) / 8-chunk split is not
   modeled; it does not reduce backward recompute residency (the whole region's graph is
   needed), but slightly staggers forward transients.
4. **Allocator effects** — fragmentation and caching-allocator high-water behavior are only
   covered by the fixed-overhead knob.

## Files

- `index.html` — the whole tool (vanilla JS, no deps, offline, theme-aware).
  - `STREAMS_DEF` — per-stream data extracted from the lowres streams directory.
  - `SCHEMA` — global input definitions + lowres defaults.
  - `simulate(p, streams)` — token counts, param formulas, checkpoint regions, and the
    phase walker producing `phases[] = {label, params, grads, optim, acts, overhead, total}`.
  - the rest is rendering (SVG stacked step-area chart with tooltip, phase table,
    params table).
- `README.md` — this file.

Headless check (used during development):

```bash
node -e 'const s=require("fs").readFileSync("tools/memory-estimator/index.html","utf8")
  .match(/<script>([\s\S]*)<\/script>/)[1]; const cut=s.slice(0,s.indexOf("const CATS"));
  const {simulate,DEFAULTS,STREAMS_DEF}=new Function(cut+";return {simulate,DEFAULTS,STREAMS_DEF}")();
  console.log(simulate(DEFAULTS, STREAMS_DEF.map(x=>({...x,on:true}))).phases)'
```
