# ZRB-E001 prerequisite — fix the store's time metadata

Run once, by someone with write access to the store (its owner, kreshpa1), before ZRB-E001.
The store was restored with the broken `restore_zarr_attrs.py` (see "Known issue" in
`../README.md`), so its `dates` and `start_date`/`end_date` describe 1979–2024 while `data`
holds March 2022 only. This rewrites them for March 2022 with the fixed script.

Run on JUPITER; paste each block as a whole. Set `WG_ROOT` to the directory holding your
WeatherGenerator checkout.

**1. Checkout** of the fixed script (detached HEAD; `git switch` refuses if the checkout has
uncommitted changes):

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
echo "Checkout ready"
)
```

**2. Fix.** Run the steps one at a time: dry run, fix, check.

```bash
WG_ROOT=/e/project1/weatherai/scheidl2
V2=/e/data1/slmet/ml_training
V2=$V2/aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51.zarr
V3=/e/scratch/hclimrep/kreshpa1/zarr3_datasets
V3=$V3/aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51
V3=${V3}_sharded_month3_2022.zarr
R=scripts/zarr_investigation/restore_zarr_attrs.py
cd "$WG_ROOT/WeatherGenerator"

# 1. Dry run: compares data at both ends, prints what would change.
uv run python $R "$V2" "$V3" --start-date 2022-03-01T00:00 --dry-run

# 2. Fix: rewrites dates, start/end_date, missing_dates; then checks.
uv run python $R "$V2" "$V3" --start-date 2022-03-01T00:00

# 3. Check again, read-only.
uv run python $R "$V2" "$V3" --start-date 2022-03-01T00:00 --check-only
```

Expected: the dry run compares v3 `data[0]`/`data[743]` with v2 steps 378360/379103
(2022-03-01T00 / 2022-03-31T23) and plans to rewrite `dates` (744 steps) and replace
`start_date`/`end_date`; steps 2 and 3 end with `metadata matches the data`.

Status: dry run done 2026-10-30 (output as expected); fix and check pending.
