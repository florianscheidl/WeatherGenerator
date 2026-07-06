# Sync-barrier removal: why the backward pass *looks* (and partly is) slower

Companion note to `docs/sync-barriers.md` (on `exp/axing-syncs`). Written after
profiling showed that with the sync barriers removed, the backward pass appears
slower, with many `cudaMemsetAsync` calls inside the FSDP MLP backward
(`Linear.backward`) and `unrolled_elementwise_kernel` where the profile previously
showed mostly `vectorized_elementwise_kernel`.

Ranked diagnosis — check 1 before trusting any per-phase numbers.

## 1. Attribution artifact: removing syncs moves where time is *accounted*, not where work happens

Before the rewrite, the sync points (`.item()`, boolean indexing,
`torch.tensor(0.0, device=...)`) drained the GPU queue in the middle of the
forward pass. By the time `loss.backward()` ran, the queue was empty and the
"backward" region of the trace contained only backward work.

With the syncs gone, the CPU runs through forward + loss without waiting; the
first hard sync now sits *after* backward (grad scaler, clip-norm, logging).
Two consequences:

- GPU kernels still queued from the forward execute — and get attributed —
  inside the backward window.
- Every checkpointed block (`use_reentrant=False` throughout the model)
  **recomputes its forward during backward**. All forward-side kernels (arange,
  cumsum, searchsorted, index_select, zero pads, pinned H2D copies) replay
  there. Previously some of those replays synced and appeared as CPU stalls in
  the forward instead.

**Verification:** wrap the phases in `torch.cuda.synchronize()` for a
measurement-only run, or compare *total step* wall time between branches. If
the step got faster and only the backward slice grew, it is attribution.

## 2. On-device 0-dim scalar accumulators in the loss module (the genuinely new kernels)

The loss-module rewrite (`loss_module_physical.py` on `exp/axing-syncs`,
commit `95813be3`) replaced Python-int counters and
`torch.tensor(0.0, device=...)` accumulators with 0-dim device tensors to avoid
host-device syncs. Side effects:

- **Every `torch.zeros((), device=...)` issues a `cudaMemsetAsync`.** They are
  created in nested loops (per stream × timestep × source-target correspondence
  × loss function: `loss_stream`, `ctr_timesteps`, `loss_timestep`, `ctr_batch`,
  `loss_st_corr`, `ctr_loss_fcts`, `loss_lfct`, `losses_chs`, `ctr_substeps`),
  easily hundreds of memsets per step.
- **Elementwise ops with a 0-dim device-tensor operand cannot take the
  vectorized path.** A Python float is baked into the kernel as a scalar; a
  0-dim CUDA tensor is a stride-0 broadcast operand, and TensorIterator falls
  back to `unrolled_elementwise_kernel`. This applies to every
  `ctr + (x > 0.0)`, `clamp(min=1.0)`, and `loss / denom`.
- Each of these ops is an autograd node, so the backward mirrors the swarm with
  another set of tiny unrolled kernels. In the trace they cluster at the seam
  between the loss and the prediction-head MLPs — i.e. they *look like* they are
  inside "FSDPMLP backward (Linear.backward)".

Each kernel is µs-scale and launch-bound, but hundreds per step in forward and
backward are real milliseconds and serialize the backward stream.

**A/B:** the branch `exp/axing-syncs-before-loss-module-rewrite` is
`exp/axing-syncs` *without* this rewrite. If the memset swarm and most unrolled
kernels disappear there, this item is confirmed.

**Fix (keeps everything sync-free, without going back to synced scalars):**
collect the per-loss-function scalar losses in Python lists and reduce once per
level with `torch.stack(...).sum()` (one kernel instead of N chained adds);
accumulate counters the same way; drop the `torch.zeros(())` accumulators
entirely (a stacked sum needs no zero-init).

## 3. Index-op backwards: memset + scatter — but boolean masks had the same

`index_select` / gather backward is `zeros_like(input)` (a `cudaMemsetAsync`)
plus `index_add_` (never vectorized). `index_copy` backward is a full `clone` +
`index_fill_` + `index_select`. This matches the observed kernel mix — but the
boolean-mask indexing these ops replaced had essentially the same backward
structure (zeros + masked scatter). This shifts the kernel *mix* toward
memset+unrolled; it is not the regression itself.

## 4. `pin_memory()` per tensor per step: host-side churn

`_host_to_device_async` pins a *fresh* small tensor on every call (substep
index tensors per stream/timestep, encoder chunk indices). Each `pin_memory()`
is a `cudaHostAlloc`/`cudaHostRegister`: expensive, takes the driver lock, and
can stall kernel launches from the launch thread. Once the step is launch-bound
(which item 2 encourages), this amplifies the slowdown. It shows up as launch
gaps, not as memsets.

**Fix:** reuse a cached pinned staging buffer per (stream, purpose) instead of
pinning fresh tensors every step.

## How to attribute the memsets in a profile

Profile with `torch.profiler` and `with_stack=True` and look at who owns each
`cudaMemsetAsync`:

| Owner in the stack | Item |
|---|---|
| `torch.zeros` under `compute_loss` / `_loss_per_loss_function` | 2 |
| `IndexSelectBackward0` / `IndexPutBackward0` / `IndexCopyBackward0` | 3 |
| FSDP reduce-scatter padding fills | pre-existing, now merely visible in a denser timeline |
