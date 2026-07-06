# CUDA Synchronization Barriers in the Model Forward Passes

Investigation of host↔device synchronization points in the forward passes of the
six engines in `src/weathergen/model/engines.py`:

- `EmbeddingEngine`
- `LocalAssimilationEngine`
- `Local2GlobalAssimilationEngine`
- `GlobalAssimilationEngine`
- `ForecastingEngine`
- `TargetPredictionEngine` (and `TargetPredictionEngineClassic`)

Because most engines delegate the real work to attention/MLP blocks, the analysis
traces into `src/weathergen/model/attention.py` and `src/weathergen/model/blocks.py`.

## Status summary

Most barriers found in the original investigation have been eliminated:

| Fix | Commit |
|-----|--------|
| EmbeddingEngine: size scatter target from tensor shape instead of `.item()` | `3a895bd9` |
| EmbeddingEngine: replace per-step assert on GPU reduction with try/except | `e02d143e` |
| Varlen attention blocks accept host-side `max_*_len` ints; static bounds threaded from all encoder engines (local, adapter, aggregation) and readout kv side | `4a470033` |
| Target readout q side: `target_coords_lens` max precomputed on CPU in the dataset workers | `31e6b01d` |
| Encoder wrapper: host-side chunk offsets, split sizes, and index-based (instead of boolean-mask) selection/scatter, using a pinned CPU copy of `tokens_lens` | working tree (not yet committed) |
| EmbeddingEngine index builders: shape-stable `searchsorted` formulation replaces boolean-mask compression (and the tensor-valued `arange` bound) | working tree (not yet committed) |
| Model target loop: NaN guard on the coord embeddings (`if torch.isnan(tc_tokens).any()`) made opt-in via `pred_nan_check` (default off) | working tree (not yet committed) |
| `AdaLayerNormLayer`: per-token conditioning expansion via `searchsorted` instead of single-arg `torch.repeat_interleave(x_lens)` (data-dependent output size → sync at the top of every wrapped readout block, forward and checkpoint-recompute) | working tree (not yet committed) |
| `predict_decoders`: `tokens_nbors_lens[0] = 0` (scalar setitem on a GPU tensor → pageable H2D copy + sync per forecast step, right before the first readout `NamedLinear`) replaced with in-place `[:1].zero_()` | working tree (not yet committed) |
| `tokens_to_latent_state`: indexing with the python lists `register_token_idxs`/`class_token_idxs` (list → CPU index tensor → pageable H2D copy + sync, twice, after the readout at step 0) replaced with plain slices (the ranges are contiguous) | working tree (not yet committed) |
| Loss calculation (`loss_module_physical.py`, `loss_module_ssl.py`): all `torch.tensor(scalar, device=...)` accumulators → `torch.zeros(())`; counters and per-channel nan-masking moved on-device; substep boolean masks → host-computed index tensors (pinned); per-stream channel weights cached on device; per-channel `w.item()` → one `tolist()` | working tree (not yet committed) |

**Still open** (conditional paths only):

- `Local2GlobalSumEngine` — `torch.repeat_interleave` with a tensor of repeats
  (data-dependent output shape); only if the `ae_adapter_type: sum` alternative is used.
- `torch.compile(flex_attention, dynamic=False)` recompilation on shape change in
  `MultiSelfAttentionHeadLocal` — not a sync, but a stall that looks like one in traces.

## Background: what forces a sync

A CUDA sync barrier occurs whenever the CPU must read a value that only exists on the
GPU, forcing it to wait for the compute stream to drain. In this codebase the triggers are:

1. **Explicit `.item()`** — copies a scalar to the host.
2. **A GPU tensor used in Python control flow** — `assert <gpu_tensor_cond>`, `if <gpu_tensor>:`,
   `bool(...)`, or using a 0-dim GPU tensor as a slice bound / Python `int`.
3. **Data-dependent output shapes** — boolean-mask indexing (`x[mask]`),
   `torch.repeat_interleave` with a tensor of repeats, `torch.nonzero`, `.unique`. The kernel
   must report how many elements it produced before the host can proceed. Note that
   boolean-mask *assignment* (`x[mask] = y`) also syncs — `index_put_` calls `nonzero`
   internally, in the forward and again in the backward.
4. **A 0-dim CUDA tensor passed where an `int` is required** — e.g. `flash_attn_varlen_func`'s
   `max_seqlen_q`/`max_seqlen_k` arguments. The pybind boundary converts the tensor to a Python
   int, which is an implicit `.item()`.
5. **A pageable host→device copy** — `cpu_tensor.to(device)` from non-pinned memory
   synchronizes the stream even with `non_blocking=True`. Pin first (see
   `_host_to_device_async` in `encoder.py:30`).
6. **Assigning a Python scalar into a GPU tensor** — `gpu_tensor[0] = 0` wraps the scalar
   as a *CPU* tensor (`THPVariable_setitem` → `valueToTensor(..., at::Device(kCPU))`) and
   the index-put copies it host→device: a pageable copy, i.e. trigger 5 in disguise. In
   traces it shows as a tiny `cudaMemcpyAsync` immediately followed by
   `cudaStreamSynchronize`. Use an in-place kernel instead: `gpu_tensor[:1].zero_()` or
   `gpu_tensor[0].fill_(0)` (a `Scalar` argument is passed by value into the kernel).
7. **Indexing a GPU tensor with a Python list (or CPU index tensor)** —
   `x[:, [0, 1, 2]]` converts the list to a CPU int64 tensor which is then copied to the
   device: again trigger 5, same `cudaMemcpyAsync` + `cudaStreamSynchronize` signature.
   For contiguous ranges use slices (views, no copy at all); otherwise register the
   indices once as a device buffer.

**Key fact for this codebase:** `batch.tokens_lens` and all the `*_lens` tensors are built on
the CPU in the dataloader workers, pinned, and moved to the GPU with `non_blocking=True`
(`datasets/batch.py`, `datasets/stream_data.py`). Any length-derived value the host needs is
therefore available for free *before* the transfer — reading it back from the GPU afterwards
is a self-inflicted sync. The fixes below all exploit this: `BatchSamples.to_device` now
retains the pinned host copy as `tokens_lens_cpu` (`batch.py:192`), and `StreamData` records
`target_coords_lens_max` as Python ints at data-prep time (`stream_data.py:103`).

---

## Findings by engine

### 1. EmbeddingEngine (`engines.py:81`) — ✅ resolved

| Status | Original issue | Resolution |
|--------|----------------|------------|
| ✅ `3a895bd9` | `num_tokens = torch.sum(...).item()` to size the scatter target | Target is sized from `torch.cat(x_embeds)` directly (`engines.py:106-112`); no host read. |
| ✅ `e02d143e` | `assert ... <= max_tokens` on a GPU reduction, every forward | Replaced by try/except around the `pe_embed` indexing (`engines.py:123`); the `.item()` only runs in the error path. |
| ✅ working tree | `pe_idxs = rows[rows < tok_counts.unsqueeze(1)]` plus `torch.arange(tok_counts.max())` — two syncs per forward | Shape-stable `searchsorted` formulation (`get_pe_idxs_vectorized`, `engines.py:135`): the total token count comes from `cat_embeds.shape[0]` (host-side shape metadata), the per-token segment id from `torch.searchsorted` over the count cumsum, the intra-cell position by subtracting segment starts. No mask, no tensor-valued arange bound. |
| ✅ working tree | `scatter_idxs = idxs[valid_mask]` plus `torch.arange(tok_counts.max())` — two syncs, multi-stream path only | Same `searchsorted` trick over the stream-major counts, reusing the existing destination-offset cumsums (`get_scatter_idxs_vectorized`, `engines.py:183`). Also no longer materializes the `(segments × max_count)` rectangle, a memory/compute win. |

Both rewrites were verified bit-identical to the original mask-based implementations over
randomized multi-stream cases (including empty cells and a fully empty stream).

### 2. LocalAssimilationEngine — ✅ resolved (`4a470033`)

The sync was `x_lens.max()` (0-dim CUDA tensor) passed to `flash_attn_varlen_func`'s
`max_seqlen` int parameters in `MultiSelfAttentionHeadVarlen` — one sync per block per
forward, doubled by activation-checkpoint recompute in the backward.

Fix: the varlen attention blocks accept optional host-side ints
(`max_x_len` / `max_q_len` / `max_kv_len`, see `attention.py:87`) and fall back to the old
behavior when they are omitted. `LocalAssimilationEngine` passes the static bound
`ae_local_max_tokens_per_cell`, which is already enforced upstream by the `pe_embed` size in
`EmbeddingEngine`. flash-attn tolerates an over-estimate: `max_seqlen` only sizes the launch
grid (surplus row-blocks exit immediately); correctness comes from `cu_seqlens`.

### 3. Local2GlobalAssimilationEngine — ✅ resolved (`4a470033`)

This was the densest sync site: `MultiCrossAttentionHeadVarlenSlicedQ` computed
`x_q_lens.max()` and `x_kv_lens.max()` **inside the per-slice loop** — up to
`2 * num_slices_q` syncs per adapter block. Moreover `x_q_lens` is `q_cells_lens`, a frozen
all-ones tensor (`model.py:154`), so the device round-trips computed the constant `1`.

Fix: the engine passes `max_q_len=1` and `max_kv_len=ae_local_max_tokens_per_cell`; the
(fallback) max computation in the block is also hoisted out of the slice loop
(`attention.py:537`).

### 4. GlobalAssimilationEngine — no syncs (unchanged)

Uses dense `flash_attn_func` and `flex_attention` with a precomputed block mask; no host
reads. Caveat: `torch.compile(flex_attention, dynamic=False)` recompiles if token counts
vary between iterations — a stall, not a sync.

### 5. ForecastingEngine — no syncs (unchanged)

Same block types as `GlobalAssimilationEngine`; the training-noise path stays on device.

### 6. TargetPredictionEngine / TargetPredictionEngineClassic — ✅ resolved (`4a470033` + `31e6b01d`)

The readout's varlen cross/self attention synced on `*_lens.max()` per block, per stream,
per forecast step (×2 with checkpoint recompute).

Fix, in two parts:
- kv side (`4a470033`): the latent lens are the constant 9 (1-ring neighborhood incl. self,
  `torch.full(..., fill_value=num_nbors)` in `model.py`); `max_latent_len=num_nbors` is
  threaded through the engines and `blocks.py` down to the attention heads.
- q side (`31e6b01d`): `StreamData` records `target_coords_lens_max` as a Python int when the
  lens are built on CPU in the dataset workers (`stream_data.py:247`, `stream_data.py:338`);
  `model.py:822` reduces over the batch on the host and passes `max_output_len`. Exact
  per-batch max, no over-estimate.

### 7. QueryAggregationEngine — ✅ resolved (`4a470033`)

Passes the static bound `num_healpix_cells + num_class_tokens + num_register_tokens` as
`max_x_len`.

### 8. Model target loop (`model.py`) — ✅ resolved (working tree), behavior change

`if torch.isnan(tc_tokens).any():` before the target readout branched on a GPU reduction —
a full device sync per stream, per forecast step (it shows up in traces immediately before
the `tcs_lens` `torch.cat` preceding `TargetPredictionEngineClassic`). There is no sync-free
way to keep a host-side branch, so the guard is now opt-in via `pred_nan_check: true`
(default off).

Context for that trade-off: the guard only fires once the coord-embedding weights are
already NaN — i.e. training has already diverged — and merely lets the run continue with
that stream's readout skipped. On fp16 runs `torch.amp.GradScaler` independently skips
optimizer steps with non-finite gradients; bf16 runs (`NoOpGradScaler`) have no other NaN
protection, so enable the flag there if limping past a diverged decoder matters more than
the per-step sync.

### 9. AdaLayerNormLayer (`norms.py`) — ✅ resolved (working tree)

`torch.repeat_interleave(x_lens)` (single-argument form) expanded the per-sequence
conditioning to per-token, but its output length is `x_lens.sum()` — a data-dependent
shape, hence a stream sync at the top of every wrapped block in the readout (and again in
the checkpoint recompute). In traces this sync appears immediately *before* the target
prediction engine's kernels, since it fires before any of the engine's compute launches.
Replaced by `torch.searchsorted` over the lens cumsum with the output length taken from
`x.shape[0]` (host-side shape metadata); verified equal to the original including
zero-length sequences.

### 10. Loss calculation (`train/loss_modules/`) — ✅ resolved (working tree)

`LossPhysical.compute_loss` was the densest sync region outside the model, hit every
training step right after the forward:

- `torch.tensor(0.0, device=...)` accumulators at five nesting levels (per stream,
  timestep, correspondence, loss function) and `torch.tensor(sw, device=...)` for the
  spoof weight — each a pageable H2D copy + sync (trigger 6/5). Replaced with
  `torch.zeros((), device=...)` (device kernel) and a plain python float.
- Host branches on device values: `1 if loss > 0.0 else 0`, `if loss_cur_w > 0.0`,
  `if ctr_... > 0`, `if loss == 0.0` — one bool sync each, per loop iteration. The whole
  counter chain now stays on device (`ctr + (x > 0)`, `clamp(min=1)` divisions); the
  loss==0 misconfiguration warning only runs during the first three steps.
- Per-channel `v != 0.0` when filling the logging dict — a sync per channel per loss
  function. Replaced with one `torch.where` per loss-function call; the avg aggregation
  handles the resulting on-device nan markers without host reads.
- `mask_t = torch.tensor(t == target_times).to(device, non_blocking=True)` per substep —
  unpinned copy (sync) *and* the downstream `target[mask_t]` boolean indexing synced
  again. Substeps are now host-computed index tensors (`np.flatnonzero` → pinned →
  `index_select`), with a `None` fast path for the common non-spacetime case, which
  previously built and applied an all-True mask.
- `torch.tensor(stream_info["target_channel_weights"]).to(device)` per stream per step —
  now cached on device at first use (the weights are static config).
- `w.item()` per channel for EMA-weight logging — one batched `tolist()` per stream.

Still open in the loss path: `loss_value.item()` per SSL loss head
(`loss_module_ssl.py`) — only relevant for SSL runs; and whatever the trainer does with
the returned losses at logging time (deferred by design via the history lists).

---

## Encoder wrapper (`encoder.py`) — ✅ resolved (working tree)

The wrappers around the engine calls had their own syncs. All length bookkeeping now runs
on the host copy `batch.tokens_lens_cpu`; index tensors that must reach the GPU are pinned
and copied with `non_blocking=True` (`_host_to_device_async`, `encoder.py:30`), which does
not stall the stream.

| Original sync | Resolution |
|---------------|------------|
| `cell_lens[...].cumsum(0)[-1]` as Python slice bounds, per chunk | Chunk offsets `l0`/`l1` are Python ints from a CPU cumsum (`encoder.py:188-196`); the empty-chunk check compares ints. |
| Boolean-mask selection of unmasked cells (`toks_global[mask]`, `q_cells_lens_cur[1:][mask]`, `cell_lens_cur[1:][mask]`) | `index_select` with `torch.nonzero` indices computed on the CPU lens (`encoder.py:217`); `q_cells_lens_unmasked` is a plain slice since the tensor is `[0, 1, 1, ...]`; `cell_lens_unmasked` is assembled on CPU and shipped pinned (`encoder.py:222`). |
| `expected_len = batch_lens.sum().item()` | `int(batch_lens.sum())` on the CPU tensor (`encoder.py:272`) — the assert is now free. |
| `list(batch_lens)` for `torch.split` (a per-element device read) | `batch_lens.tolist()` on the CPU tensor (`encoder.py:277`). |
| Per-sample `rope_cell_coords[mask_b]` in the packed-coords loop | One combined `index_select` with host-computed cell indices, then a host-sized `split` (`encoder.py:298`). |
| `tokens_global[mask] = ...` final fill (bool-mask assignment → `nonzero` in forward *and* backward; found during the fix, not in the original list) | Out-of-place `index_copy` on host-computed flat indices (`encoder.py:392-397`); gradients flow through `index_select`/`index_fill`, both sync-free. |

Fallback: if a code path arrives without the host copy, `assimilate_local` does a one-off
`batch.tokens_lens.cpu()` (single sync) rather than failing.

## Summary table

| Engine | Sync? | Notes |
|--------|-------|-------|
| EmbeddingEngine | ✅ none | scatter target sized from shape; `searchsorted`-based index builders |
| LocalAssimilationEngine | ✅ none | static `max_seqlen` bound |
| Local2GlobalAssimilationEngine | ✅ none | `max_q_len=1`, static kv bound |
| QueryAggregationEngine | ✅ none | static bound from cell/class/register counts |
| GlobalAssimilationEngine | ✅ none | unchanged; flex-attention recompile caveat |
| ForecastingEngine | ✅ none | unchanged |
| TargetPredictionEngine(Classic) | ✅ none | constant kv max (9), exact q max from dataset |
| Encoder wrapper | ✅ none | host bookkeeping via `tokens_lens_cpu` |
| Model target loop | ✅ none by default | NaN guard opt-in via `pred_nan_check` |
| Loss (physical) | ✅ none steady-state | device accumulators/counters; loss==0 check first 3 steps only |

---

## Verifying changes without a GPU

`flash_attn` is not installable on macOS, so `tests/sync_barrier_equivalence.py` stubs it
with a CPU reference implementation that also **asserts every flash-attn `max_seqlen`
argument arrives as a Python int** (i.e. no device sync at the pybind boundary). It
contains old-vs-new equivalence tests for every rewritten code path: the varlen attention
plumbing (incl. backward through checkpoint), the encoder chunk/aggregation/scatter
rewrites, the EmbeddingEngine index builders, the `AdaLayerNormLayer` segment ids, the
dataset-side `target_coords_lens_max`, and the loss-module substep/accumulator rework.

Run: `uv run python tests/sync_barrier_equivalence.py`

Extend this script when removing further barriers: copy the *old* implementation into the
script as a reference and assert bit-identical outputs against the new code.

---

## Operational notes for HPC runs (environment, not code)

Findings from nsys traces on the cluster during this work:

### `cuKernelSetAttribute` stalls (~360 ms), clustered in the readout Linears

`cuKernelSetAttribute` sets a kernel's max dynamic shared memory before launch (cuBLASLt
does this when configuring a GEMM kernel variant). With CUDA's default **lazy module
loading**, the first touch of each kernel variant also pays the kernel load — including a
PTX→SASS JIT compile if the library ships no SASS for the GPU arch or the JIT cache is
cold. The ragged token counts of the readout keep steering cuBLASLt to new kernel
variants, hence the clustering in `AdaLayerNorm`/`Linear` there. Mitigations:

- `CUDA_MODULE_LOADING=EAGER` in the job env — front-loads all module loading at startup.
- Persistent JIT cache: `CUDA_CACHE_PATH=<persistent writable path>` and
  `CUDA_CACHE_MAXSIZE=4294967296`; a node-local ephemeral cache re-pays every stall each
  job.
- Verify SASS coverage: compare `python -c "import torch; print(torch.cuda.get_arch_list())"`
  with `nvidia-smi --query-gpu=compute_cap --format=csv`.
- If churn persists late in training: bucket/pad the ragged token counts in the readout
  (also a prerequisite for CUDA-graphing it).

### Allocator-induced syncs

`tokens_nbors` in `predict_decoders` materializes the latent 9× — likely the largest
transient allocation of the step — amid ragged readout allocations. Under fragmentation
the caching allocator frees cached segments, and `cudaFree` synchronizes the device. If
`torch.cuda.memory_stats()["num_alloc_retries"]` grows over training, set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Data-loading pinning

`data_loading.memory_pinning` must be `true` (default is true in `default_config.yml`,
but some FSDP2 + DINOv2 runs disable it to avoid a hang). With it off, every tensor moved
in `batch.to_device` is a pageable synchronizing copy at step start — a burst of
memcpy+sync pairs at the top of each step in a trace.

### Diagnosis playbook

- `torch.cuda.set_sync_debug_mode("warn")` early in setup prints a Python stack trace at
  every synchronizing op (including pageable copies) — the fastest attribution tool, use
  it before triangulating from timelines.
- nsys: add `--cuda-memory-usage=true` to see allocator `cudaMalloc`/`cudaFree` next to
  suspect syncs.
- Trace signatures:

  | Signature | Cause |
  |-----------|-------|
  | tiny `cudaMemcpyAsync` + `cudaStreamSynchronize` | pageable copy: `.item()`, scalar setitem, list indexing, `torch.tensor(..., device=...)`, unpinned `.to(device)` |
  | `cudaStreamSynchronize` without adjacent memcpy | data-dependent output shape: `nonzero`, boolean indexing/assignment, single-arg `repeat_interleave`, `unique` |
  | `cudaFree` + device-wide sync | caching allocator releasing segments (fragmentation) |
  | long `cuKernelSetAttribute` / module load | lazy kernel loading, PTX JIT |

- Caution: timeline attribution is unreliable — a sync at the top of a checkpointed block
  appears *before* that module's kernels in the trace. This misled the investigation twice
  (NaN guard, `AdaLayerNormLayer`).

---

## Open items / next steps

- Commit the working-tree changes (encoder/batch host bookkeeping, EmbeddingEngine
  `searchsorted` builders, `AdaLayerNormLayer`, `pred_nan_check` gating, latent-state
  slices, `tokens_nbors_lens` fill, loss-module rework) and update the "working tree"
  rows above with the commit hash.
- Re-profile a training step on the HPC. Expected remaining host reads: the loss==0
  check on the first 3 steps, one-off cache fills (channel weights, kernel warmup), and
  the deliberate leftovers below.
- `BilinearDecoder` (`engines.py`, only `decoder_type: Linear`): `repeat_interleave` with
  tensor repeats → sync; same searchsorted/`output_size` fix applies if that path is used.
- `Local2GlobalSumEngine` (only `ae_adapter_type: sum`): same `repeat_interleave` issue.
- `loss_value.item()` per SSL loss head (`loss_module_ssl.py`) — SSL runs only.
- Trainer logging path: `losses_all` now carries more device tensors; confirm conversion
  happens once per log interval, not per step. Also check `total_norm` from
  `clip_grad_norm_` is not read to host every step.
- Reminder: with `pred_nan_check` off, bf16 runs (`NoOpGradScaler`) have no NaN
  protection; fp16 runs are covered by `GradScaler`.
