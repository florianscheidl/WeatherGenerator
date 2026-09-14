# Spatial-resolution feasibility on Jupiter

Compare the memory/feasibility envelope of the same lowres-derived model on one
four-GPU Jupiter Booster node. The primary matrix varies only source dataset
resolution and HEALPix level. All target datasets stay fixed, including hourly
ERA5 O96. No training runs have been performed for this matrix yet.

Code branches:

- `exp/spatial-resolution-experiment-develop`: based on ECMWF develop `f730876c`.
- `exp/spatial-resolution-experiment-reader-data-parallel`: based exactly on
  `b4cd88ad`, the production spatial-selective reading slice used by SIO-E021/E022.
  Despite the branch name, its primary treatment uses spatial parallelism.

The files in this directory are identical on both branches. Use the data-parallel
arm on develop and the spatial-local arm on the reader branch. The reader branch
also supports the two control arms below. Develop does not implement the spatial
switches: do not run its spatial arms and interpret them as spatial experiments.

## Fixed model and memory settings

`base.yml` is a frozen copy of `config_operan_georing_avhrr_forecasting_lowres.yml`,
with experiment-wide overrides explicitly recorded:

- Eight DataLoader workers per rank, memory pinning off. Training uses the built-in
  prefetch factor of two: up to 64 outstanding batches over four ranks. There is
  no supported training `data_loading.prefetch_factor` override on these bases.
- One source sample/view and one six-hour input window; one-hour sample stride.
  Two six-hour forecast steps, offset one, fixed forecast policy.
- Local/global width 2048, four local blocks, four global blocks, 16 forecast
  blocks; one query per cell, no class/register tokens, no aggregation blocks.
  `ae_local_max_tokens_per_cell=128` is unchanged.
- FSDP, BF16 attention/mixed precision, FlashAttention, compilation off, existing
  code checkpoint placement; physical MSE and ensemble size one.
- Lowres stream architecture throughout: GEO transformer embeddings, dimension
  512, two blocks, four heads, token size 1024. All channel lists, geoinformation,
  stream names/IDs, masking, target caps, and other stream settings are frozen.
  AVHRR/IASI and surface observations are unchanged.
- All stage dates are in 2021. This avoids the existing hires Himawari IR file list
  falling back to O256 for Himawari-9 after 2022. The inherited Himawari-9 files
  remain listed, but are outside the configured experiment dates.
- Configured seed 20260914. **Develop overwrites this with wall-clock time**;
  `b4cd88ad` honors it. These configurations do not claim identical sampled masks
  or loss trajectories across branches. No seed implementation is backported.
- Constant learning-rate parallel scaling prevents the LR amplitude changing
  with the number of data-parallel groups. Other lowres optimizer/schedule values,
  4096 samples per mini-epoch, 56 mini-epochs, and logging cadence are retained.
  Runtime batch/world-size adaptations still differ between execution arms.

Private platform configuration can override a base config. The commands below
therefore also apply `base.yml` first as an overwrite, before input/level/arm.
This preserves platform paths but reasserts the experimental model settings.

## Resolution and execution matrix

| Input overlay | Operational analysis input | GEO input | Diagnostic ERA5 target |
| --- | --- | --- | --- |
| `input_o96_o256.yml` | O96 | O256 | O96 |
| `input_n320_o256.yml` | N320 | O256 | O96 |
| `input_n320_h512.yml` | N320 | H512 (2021) | O96 |

Cross each row with `healpix_5.yml` (12,288 cells) and `healpix_6.yml` (49,152 cells).
Separate stream directories are intentional: `run_train` reloads streams from
`streams_directory`, discarding inline stream overrides. Within those snapshots,
only source filenames differ; target files and network definitions are identical.
The H512 filenames are taken from the historical hires config, but its linear
embeddings, larger tokens, changed masking, and metadata are not adopted.

| Arm | Spatial size | Reader filtering | Local physical targets/loss |
| --- | ---: | --- | --- |
| `data_parallel.yml` | 1 | off | off |
| `spatial_full_read.yml` | 4 | off | off |
| `spatial_local.yml` | 4 | on | on |

The full-read control still includes the branch's always-active spatial behavior
(including local source token construction and local target coordinates). It is
not an exact recreation of the pre-investigation baseline.

## Launch examples

Run from the repository root, after checking out the intended branch and setting
up its locked environment. These commands submit one job each; select each of the
three input overlays and both HEALPix overlays for the full matrix.

Develop baseline, O96/O256 at level 5:

```bash
../WeatherGenerator-private/hpc/launch-slurm.py \
  --nodes=1 --time=30:00 \
  --base-config config/experiments/spatial_resolution/base.yml \
  --config config/experiments/spatial_resolution/base.yml:config/experiments/spatial_resolution/input_o96_o256.yml:config/experiments/spatial_resolution/healpix_5.yml:config/experiments/spatial_resolution/data_parallel.yml
```

Reader branch, N320/H512 at level 6:

```bash
../WeatherGenerator-private/hpc/launch-slurm.py \
  --nodes=1 --time=30:00 \
  --base-config config/experiments/spatial_resolution/base.yml \
  --config config/experiments/spatial_resolution/base.yml:config/experiments/spatial_resolution/input_n320_h512.yml:config/experiments/spatial_resolution/healpix_6.yml:config/experiments/spatial_resolution/spatial_local.yml
```

Allow more wall time if needed to observe at least 100 optimizer steps after
startup. The sample/mini-epoch settings are training limits, not a 100-step stop.
At the capacity boundary, repeat both the last passing and first failing cases,
and run the reader branch with `spatial_full_read.yml` for attribution. A reader
branch `data_parallel.yml` run provides a same-code non-spatial control.

Retry develop's first failure with `--options data_loading.num_workers=1` as a
separately labelled capacity check. This distinguishes loader concurrency limits
from per-sample GPU capacity; it is not part of the fixed eight-worker matrix.

## Verification and interpretation

```bash
uv run --no-sync python config/experiments/spatial_resolution/verify.py
git diff exp/spatial-resolution-experiment-develop \
  exp/spatial-resolution-experiment-reader-data-parallel -- config/experiments/spatial_resolution
```

The first command loads all 18 configurations through the branch's actual config
loader with an empty private config, then reloads streams as training does. It
checks the permitted differences and prints config hashes for cross-branch
comparison. The second command should print no differences. Dataset access,
actual channel availability, and GPU execution must still be checked on Jupiter.
Inspect saved runtime configs for the actual seed, channels, and platform settings.

Record node cgroup peak/events, shared-memory peak, maximum allocated/reserved
GPU memory across ranks, completed steps, source token counts, and target rows.
Classify failures as host OOM, shared-memory exhaustion, GPU OOM, token capacity,
collective failure, or insufficient observation time; a timeout is not an OOM.

Four data-parallel ranks process four global samples per optimizer step; spatial
size four processes one global sample. Report complete global samples/s, never
count four spatial shards as four samples. Temporal work partitioning also differs.
This is a capacity experiment, not a matched-batch convergence comparison.

Validation retains the global path (`spatial_local_validation=false`) in all arms;
measure it separately from training. The global latent/forecast path is not fully
spatially sharded, so level 6 may still fail in GPU memory after loading fits.

This is **not an exact E021 reproduction**: E021 used N320 targets and linear GEO
embeddings. Transformer H512 feasibility is unmeasured. Increasing target resolution
would be a separate experiment and must not be mixed into the input-only matrix.
