---
name: document-perf-changes
description: Update the sync-barrier/performance docs after changing WeatherGenerator model or training code. Use after ANY edit to src/weathergen that removes or introduces a host-device sync, changes kernel behavior (memsets, vectorized vs unrolled, launch counts), alters training logic or defaults (e.g. pred_nan_check), or restructures loss/encoder/attention code — before declaring the task done. Also use when asked to "document the changes".
---

# Document performance / sync-barrier changes

The sync-barrier workstream lives or dies by its docs: profiling happens on the HPC,
fixes happen locally, and branches are switched constantly. Undocumented changes get
lost or re-litigated. After changing relevant code, update the docs **in the same
working tree**, before finishing the task.

## Where to write

1. **`docs/sync-barriers.md`** — the source of truth. It has three places that must
   stay consistent with each other:
   - the **Status summary** table near the top (`| Fix | Commit |`),
   - the **per-engine section** ("Findings by engine", numbered; tables use
     `| Status | Original issue | Resolution |` or prose),
   - the **Summary table** near the bottom (`| Engine | Sync? | Notes |`).
2. **`docs/sync-barriers-backward-pass.md`** — backward-pass / profiling-signature
   insights (memset swarms, unrolled-vs-vectorized, attribution artifacts). Extend it
   when the change affects the backward pass or explains a profile observation.
3. If neither doc exists on the current branch, check other branches before creating a
   duplicate: `git log --all --oneline -- docs/sync-barriers.md` and restore with
   `git show <commit>:docs/<file> > docs/<file>`.

## Conventions

- **Commit refs**: cite the short hash when the change is committed; write
  `working tree (not yet committed)` otherwise. Florian commits himself — never
  commit for him; he replaces the placeholder with the hash later.
- **Supersede, don't delete**: when a fix replaces an earlier one, keep the old row
  and mark it, e.g. `` `e02d143e` (superseded by `0c6f90f7`: …) ``. The history of
  *why* an approach was abandoned (e.g. try/except can't catch CUDA device-side
  asserts) is the most valuable content.
- **State the sync mechanism**, not just the change: which of the trigger classes
  fired (`.item()`, host branch on device tensor, boolean indexing / data-dependent
  shape, pageable H2D copy, tensor-valued arange bound, …) and where it appears in a
  trace ("immediately before X's kernels").
- **Record verification and its limits**: what was checked (values, gradients,
  randomized cases, edge cases like empty levels / spoofed targets) and whether the
  result is bit-identical or only mathematically equivalent (e.g. changed summation
  order). Mac has no flash-attn — say when something could only be compile/CPU-checked.
- **Flag behavior changes separately from perf changes** (defaults flipped,
  regularization semantics, guards disabled). These are what silently nullify
  speedups; they get their own sentence, not a subordinate clause.

## Stale-claim sweep (mandatory)

New changes routinely invalidate older doc statements. After writing the new entry,
grep the docs for claims your change contradicts and fix them in the same pass:

- defaults ("default off/on"), e.g. `pred_nan_check`,
- "resolved with X" where X was just replaced,
- the bottom Summary table's `Sync?` column,
- kernel/uncommitted-status claims (`working tree` rows whose commits now exist —
  update when the hash is known).

A doc that contradicts the code is worse than no doc.
