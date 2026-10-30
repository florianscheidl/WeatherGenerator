# ZRB-E001 — N320 v2, sharded, March 2022

- Date: 2026-10-30
- Status: blocked — the store's time metadata doesn't match its data; the fix (prerequisite
  below) needs write access to the store, i.e. its owner (kreshpa1)
- Code: WeatherGenerator [`65dfdf2f`](https://github.com/florianscheidl/WeatherGenerator/commit/65dfdf2fa4688f29da6553daf1f153d543557885),
  WeatherGenerator-private [`7fb48aba`](https://gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private/-/commit/7fb48aba885b2df117b1a9d0387e5c2103b8fec0),
  weathergen-configs [`ab7a0c53`](https://github.com/florianscheidl/weathergen-configs/commit/ab7a0c535f8dceeb1631dcbf606f137ad161358f)
- Config: `forecast/config_era5_lowres_dataonly.yml` (streams from `streams/era5_split_lowres/`)
- Dataset: `aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51_sharded_month3_2022.zarr`
  in `/e/scratch/hclimrep/kreshpa1/zarr3_datasets/`, sharded from the Zarr 2 source
  `aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51.zarr` in
  `/e/data1/slmet/ml_training/`

## Reproducibility

Run on JUPITER. Each block is self-contained (it sets its own variables); paste it as a whole.
Lines are flush-left and short on purpose: indentation, and terminal wrapping of long lines,
break them when pasted. `WG_ROOT` is the directory holding the three checkouts side by side.

**Prerequisite:** the store's time metadata must be fixed first (once, by its owner):
[`ZRB-E001-prerequisite-fix-store.md`](ZRB-E001-prerequisite-fix-store.md).

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
git switch --detach 65dfdf2fa4688f29da6553daf1f153d543557885
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
--filename "$DS"
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
--nsys
```

`--nsys` (or `WEATHERGEN_NSYS_PROFILING=1`) adds `--nvtx` itself. The profiler adds overhead,
so take durations from a run without it (drop `WEATHERGEN_NSYS_PROFILING=1` / `--nsys`).

- Output: `profiling/zarr_experiment_results/anemoi_reader_benchmark/<dataset>_<shards>_<time>[_<job>].jsonl`,
  trace with the same name in `profiling/zarr_experiment_results/profiler_traces/`
- Job ID(s) / output name(s): —

## Result

First run (2026-10-30, direct, `--nsys`, before the prerequisite) failed in
`DataReaderAnemoi._get`: `ValueError: cannot reshape array of size 0 into shape (0,newaxis)`.

Cause: the store's time metadata describes the full source dataset, not the data it holds.
`data` has shape `(744, 113, 1, 542080)` (31 days × 24 h), but `dates` has 403248 entries and
the root attributes say `start_date: 1979-01-01T00:00:00`, `end_date: 2024-12-31T23:00:00`,
`frequency: 1h`. anemoi reports `len(ds) == 744` with dates 1979-01-01 … 2024-12-31, so the
reader maps training windows (1990–2022) to time indices far past 744, and the read returns
nothing.

Origin: after sharding, the broken `restore_zarr_attrs.py` copied the source's full `dates`
array and root attributes into the store. The fixed script's dry run (prerequisite) matched
v3 `data[0]`/`data[743]` to v2 steps 378360/379103, so the store is exactly March 2022; the
fix itself is pending (no write access for the experimenter). Rerun once it is done.
