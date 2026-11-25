# ZRB-E004 — Training on Zarr 3 copies of all datasets

- Date: 2026-11-16
- Status: done, recorded in the experiments repository as
  [Zarr 3 001](https://gitlab.jsc.fz-juelich.de/esde/weathergenerator-experiments/-/blob/main/experiments/infrastructure/2026-11-zarr3-datasets/zarr3_001_long_run.md) (2026-11-25).
  Enxhi Kreshpa converted all ten stores (`<source>-zarr3-shard<N>.zarr` next to the sources in
  `/e/data1/slmet/ml_training/`). Three seeds, each on Zarr 2 and on Zarr 3, with zarr-python
  3.4.2: training works on the same data (losses identical to four decimals for 50–100
  steps), with the same startup and memory, and costs 0 to +7 % of training time in runs
  started together, within the run-to-run spread of these data-bound runs. Only the ERA5 store
  differs from its source, in attributes no reader uses. The rest of this record is the plan
  as written on 2026-11-16.
- Question: does a training run with the hackathon baseline config work on Zarr 3 copies of
  all its datasets, and how do its step time and data waits compare with the same run on the
  Zarr 2 sources? ZRB-E001 to E003 timed only the reader, and only on the N320 ERA5 store.
- Config: the config of hackathon
  [baseline 001](https://gitlab.jsc.fz-juelich.de/esde/weathergenerator-experiments/-/blob/main/experiments/performance/2026-11-perf-zurich-hackathon/baseline/baseline_001_nsys_1_node.md):
  weathergen-configs
  [`70c2caa8`](https://github.com/florianscheidl/weathergen-configs/commit/70c2caa81070a593e0e0001abbb785523d1d92b4),
  `performance/base_performance_config.yml`, streams from
  `streams/era5_georing_avhrr_synop_highres_2/`. The run in baseline 001 used WeatherGenerator
  `7fe82df1` (`develop-ssl-diffusion-performance`) and WeatherGenerator-private `1ed133a6`.
  The data paths come from `hpc/jupiter/config/paths.yml` in the private repo.
- Builds on [ZRB-E002](ZRB-E002-n320-v2-v3-march-2022.md) (Zarr 3 with 24-step shards was about
  1.2× slower per window than Zarr 2) and [ZRB-E003](ZRB-E003-n320-2022-shard-sizes.md)
  (shard size; planned). The shard size for ERA5 should come from E003.

## Conversion

These are the dataset files that the stream configs in
`streams/era5_georing_avhrr_synop_highres_2/` (at `70c2caa8`) reference. All Zarr 2 sources are
in `/e/data1/slmet/ml_training/` on JUPITER (`data_path_anemoi` and `data_path_obs`).

| # | Dataset file | Stream (type) | Stream file |
|--:|---|---|---|
| 1 | `aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51.zarr` | `ERA5_in` (`anemoi_operan`, source), `ERA5` (`anemoi`, target) | `era5.yml`, `era5_out.yml` |
| 2 | `observations-file-2001-2023-eumetsat-avhrr-o256-v4.zarr` | `AVHRR` (obs) | `avhrr.yml` |
| 3 | `observations-od-ai-0001-2011-2023-iasi-pc-bufr-v1.zarr` | `METOP_IASI_PC` (obs) | `avhrr.yml` |
| 4 | `observations-file-2014-2024-seviri-h512-v5.zarr` | `METEOSAT_SEVIRI_IR_HIRES` (obs) | `geos.yml` |
| 5 | `observations-file-2017-2024-abi-goes16-IR-h512-v2.zarr` | `GOES_ABI_IR_HIRES` (obs) | `geos.yml` |
| 6 | `observations-file-2017-2024-abi-goes16-VIS-h512-v2.zarr` | `GOES_ABI_VIS_HIRES` (obs) | `geos.yml` |
| 7 | `observations-file-2015-2022-himawari8-IR-h512-v1.zarr` | `HIMAWARI_AHI_IR_HIRES` (obs) | `geos.yml` |
| 8 | `observations-file-2022-2024-himawari9-IR-o256-v1.zarr` | `HIMAWARI_AHI_IR_HIRES` (obs) | `geos.yml` |
| 9 | `observations-file-2015-2022-himawari8-VIS-h512-v1.zarr` | `HIMAWARI_AHI_VIS_HIRES` (obs) | `geos.yml` |
| 10 | `observations-file-2022-2024-himawari9-VIS-h512-v1.zarr` | `HIMAWARI_AHI_VIS_HIRES` (obs) | `geos.yml` |

That makes 10 files for 9 streams: both ERA5 streams read file 1, and each Himawari stream
reads two files (Himawari-8, then Himawari-9).

Time ranges in the config: training 2016-01-01 to 2022-12-31, validation 2023-10-01 to
2023-12-31, test 2023-06-01 to 2023-08-31. Converting 2016–2023 would cover all three. Later
hackathon configs (weathergen-configs `6a5d4ba8`) train from 2020-01-01.

Tooling per file:

- **File 1 (anemoi format):** use [`convert_range.py`](../convert_range.py) (it reads through
  `anemoi.datasets.open_dataset`, cuts a month range, and writes `data` with optional shards),
  then [`restore_zarr_attrs.py`](../restore_zarr_attrs.py) `--start-date`; see
  [`recipes/zarr-version-conversion-recipe.md`](../recipes/zarr-version-conversion-recipe.md).
  Then check the store with
  [`recipes/ZRB-E003-check-converted-stores.md`](../recipes/ZRB-E003-check-converted-stores.md).
- **Files 2–10 (WeatherGenerator obs format):** no converter yet. These stores are read by
  `DataReaderObs` with `zarr.open`, and each holds a 2-D `data` table (rows × columns, with
  attributes `colnames`, `means` and `vars`), a `dates` array and an hourly row index
  `idx_197001010000_1`. `convert_range.py` doesn't apply to them. A time slice would mean
  rewriting the row index, so converting them whole is simpler.

Streams in the same directory that are commented out, and whose files are **not** needed:

- `METOP_ABC_AVHRR_IASI` (`avhrr.yml`):
  `observations-ea-ofb-0001-2007-2021-metop-a-iasi-radiances-v1.zarr`,
  `observations-ea-ofb-0001-2013-2023-metop-b-iasi-radiances-v1.zarr`,
  `observations-ea-ofb-0001-2019-2023-metop-c-iasi-radiances-v1.zarr`
- `SurfaceCombined` (`synop.yml`, the whole file):
  `observations-ea-ofb-0001-1979-2025-combined-surface-v5.zarr`
