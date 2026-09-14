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

The launcher receives `base.yml` only through `--base-config`. Do not repeat it
under `--config`: the launcher logs each extra YAML separately to MLflow, where
repeating a top-level parameter such as `data_loading` with another value fails.
The input, HEALPix, and execution overlays deliberately have disjoint top-level
keys. Platform-specific configuration should provide paths rather than overwrite
the model settings fixed in `base.yml`.

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

## Direct launch commands

Run these blocks from the repository root on Jupiter. Each invocation submits one
independent ten-minute job with a generated run ID. Ten minutes is intended only
to determine whether the configuration reaches and starts training; it is not a
throughput or sustained-memory measurement.

Define this helper once in the shell. Its three arguments are the input-resolution
name, HEALPix level, and execution arm. The three YAMLs passed through `--config`
have disjoint top-level MLflow parameter keys.

```bash
launch_spatial_resolution() {
  input_resolution="$1"
  healpix_level="$2"
  execution_arm="$3"

  ../WeatherGenerator-private/hpc/launch-slurm.py \
    --nodes=1 --time=10:00 \
    --base-config config/experiments/spatial_resolution/base.yml \
    --config \
      "config/experiments/spatial_resolution/input_${input_resolution}.yml" \
      "config/experiments/spatial_resolution/healpix_${healpix_level}.yml" \
      "config/experiments/spatial_resolution/${execution_arm}.yml"
}
```

### Develop: primary data-parallel baseline

Check out `exp/spatial-resolution-experiment-develop`. These are the six valid
primary configurations on develop:

```bash
launch_spatial_resolution o96_o256 5 data_parallel
launch_spatial_resolution n320_o256 5 data_parallel
launch_spatial_resolution n320_h512 5 data_parallel

launch_spatial_resolution o96_o256 6 data_parallel
launch_spatial_resolution n320_o256 6 data_parallel
launch_spatial_resolution n320_h512 6 data_parallel
```

Within each HEALPix level, the three jobs isolate native input resolution as far
as an end-to-end run permits. Within each input row, levels 5 and 6 primarily test
the larger HEALPix latent grid, though token fragmentation and local assimilation
also change with the level.

### Reader branch: primary complete spatial path

Check out `exp/spatial-resolution-experiment-reader-data-parallel`. These six jobs
enable spatial size four, reader-boundary filtering, and rank-local physical
targets/loss:

```bash
launch_spatial_resolution o96_o256 5 spatial_local
launch_spatial_resolution n320_o256 5 spatial_local
launch_spatial_resolution n320_h512 5 spatial_local

launch_spatial_resolution o96_o256 6 spatial_local
launch_spatial_resolution n320_o256 6 spatial_local
launch_spatial_resolution n320_h512 6 spatial_local
```

The twelve jobs above are the primary capability comparison. Record whether each
job reaches its first optimizer step, the number of completed steps before the
wall time, and the failure class if it does not start training.

### Reader branch: same-code data-parallel controls

These six jobs run the reader-branch code with spatial size one and all locality
switches off. They separate a branch/base-code difference from the cumulative
spatial feature when a develop and spatial-local result differ:

```bash
launch_spatial_resolution o96_o256 5 data_parallel rv3vj7ap
launch_spatial_resolution n320_o256 5 data_parallel ziwwbnii
launch_spatial_resolution n320_h512 5 data_parallel oxeq5dr9

launch_spatial_resolution o96_o256 6 data_parallel x6blzdqm
launch_spatial_resolution n320_o256 6 data_parallel txqwrj44
launch_spatial_resolution n320_h512 6 data_parallel k3h1irm8
```

### Reader branch: spatial full-read attribution controls

These six jobs enable spatial size four while leaving reader filtering and local
physical targets/loss off:

```bash
launch_spatial_resolution o96_o256 5 spatial_full_read
launch_spatial_resolution n320_o256 5 spatial_full_read
launch_spatial_resolution n320_h512 5 spatial_full_read

launch_spatial_resolution o96_o256 6 spatial_full_read
launch_spatial_resolution n320_o256 6 spatial_full_read
launch_spatial_resolution n320_h512 6 spatial_full_read
```

Run the controls only where they resolve an ambiguity in the primary twelve jobs.
The full-read arm still includes the branch's always-active spatial behavior,
including local source token construction and local target coordinates, so it
does not isolate reader filtering alone.

Worker-count sweeps need dedicated execution-arm YAMLs so that `data_loading` is
logged only once. They are outside this fixed eight-worker matrix.

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
