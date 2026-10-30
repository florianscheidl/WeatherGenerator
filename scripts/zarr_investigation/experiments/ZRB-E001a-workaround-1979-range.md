# ZRB-E001a — N320 v2, sharded, March 2022: run before the store fix

- Date: 2026-10-30
- Status: planned
- Question: same as [ZRB-E001](ZRB-E001-n320-sharded-month3-2022.md), run on the store as it
  is, without its prerequisite (the time-metadata fix).
- Code: WeatherGenerator [`acfe78b6`](https://github.com/florianscheidl/WeatherGenerator/commit/acfe78b60d662449acd3f6e9f16b7d07ed1c8139)
  (newer than ZRB-E001's `65dfdf2f`; only `restore_zarr_attrs.py` differs, the benchmark is
  unchanged), WeatherGenerator-private [`7fb48aba`](https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/commit/7fb48aba885b2df117b1a9d0387e5c2103b8fec0),
  weathergen-configs [`ab7a0c53`](https://github.com/florianscheidl/weathergen-configs/commit/ab7a0c535f8dceeb1631dcbf606f137ad161358f)
- Config, dataset: as ZRB-E001.

## Workaround

The store's `dates` start at 1979-01-01T00 (see ZRB-E001, Result), and the reader numbers time
steps from there. So it dates the 744 stored steps (March 2022) as 1979-01-01T00 …
1979-01-31T23, and the training range must be given in those dates:
`training_config.start_date=1979-01-01T00:00`, `training_config.end_date=1979-01-31T18:00`
(the last 6 h window still ends inside the data). Only the window timestamps are wrong; the
bytes read are those a run on the fixed store with the matching 2022 range would read, so
`--checksum` values should agree with it.

## Reproducibility

Run on JUPITER; paste each block as a whole. `WG_ROOT` is the directory holding the three
checkouts side by side.

**1. Checkout.** Check out the three repositories at the recorded commits (detached HEAD) and
sync the env. `git switch` refuses if a checkout has uncommitted changes.

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
(
set -e
cd "$WG_ROOT/WeatherGenerator"
git fetch \
https://github.com/florianscheidl/WeatherGenerator.git \
exp/anemoi-reader-zarr-benchmark
git switch --detach acfe78b60d662449acd3f6e9f16b7d07ed1c8139
./scripts/actions.sh sync
cd "$WG_ROOT/WeatherGenerator-private"
git fetch \
git@gitlab.jsc.fz-juelich.de:esde/WeatherGenerator-private.git \
exp/anemoi-reader-zarr-benchmark
git switch --detach 7fb48aba885b2df117b1a9d0387e5c2103b8fec0
cd "$WG_ROOT/weathergen-configs"
git fetch \
https://github.com/florianscheidl/weathergen-configs.git \
zarr-experiment-configs
git switch --detach ab7a0c535f8dceeb1631dcbf606f137ad161358f
echo "Checkouts ready"
)
```

**2a. Run through Slurm** (`booster`, 1 node, 1 task, 72 CPUs). The job runs from the
checkout, so don't switch commits until it finishes.

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
DS=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
DS=$DS/aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51
DS=${DS}_sharded_month3_2022.zarr/
cd "$WG_ROOT/WeatherGenerator"
WEATHERGEN_NSYS_PROFILING=1 sbatch \
../WeatherGenerator-private/hpc/jupiter/benchmark_anemoi_reader.sh \
--config ../weathergen-configs/forecast/config_era5_lowres_dataonly.yml \
--stream ERA5 --stage train --order shuffled --seed 0 \
--num-windows 100 --warmup 2 --checksum \
--filename "$DS" \
--options training_config.start_date=1979-01-01T00:00 \
training_config.end_date=1979-01-31T18:00
```

**2b. Run directly**, on a compute node (`--nsys` needs nsys on PATH):

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
DS=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
DS=$DS/aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51
DS=${DS}_sharded_month3_2022.zarr/
cd "$WG_ROOT/WeatherGenerator"
ml Stages/2025 Nsight-Systems/2024.7.1
uv run python scripts/zarr_investigation/benchmark_anemoi_reader.py \
--config ../weathergen-configs/forecast/config_era5_lowres_dataonly.yml \
--stream ERA5 --stage train --order shuffled --seed 0 \
--num-windows 100 --warmup 2 --checksum \
--filename "$DS" \
--options training_config.start_date=1979-01-01T00:00 \
training_config.end_date=1979-01-31T18:00 \
--nsys
```

Check that the summary line reports `num_empty: 0`.

- Job ID(s) / output name(s): —

## Result

—
