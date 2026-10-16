# WeatherGenerator Performance Analysis Tools

Tools for extracting and plotting scaling performance data from WeatherGenerator training runs.

This package is standalone: it is not a member of the repository's uv workspace, has its own
`uv.lock`, and does not depend on the rest of the repository. It only reads the
`<run_id>_train_metrics.json` files written by training runs.

## Installation

From this directory:

```bash
uv sync
```

## Scripts

### extract_scaling_data

Extracts scaling metrics from the train metrics of WeatherGenerator runs and writes parquet (and
csv) output.

```bash
uv run extract_scaling_data --run-ids RUN_ID1 RUN_ID2 \
  --shared-work-dir /path/to/shared_work --output scaling.parquet
```

`--shared-work-dir` is the directory containing `results/<run_id>/`. Run-ids can also be given as
a dict mapping the number of nodes to run-ids, e.g. `--run-ids '{1: run1, 4: run2}'`; otherwise
the number of nodes is read from the run's `logs/<run_id>/output.*.txt` (see `--logs-base-dir`).

Outputs:

- `scaling.parquet`: one row per run with `num_nodes`, `startup_time_seconds`, `training_time` and
  `loss_avg_mean`. `training_time` is the last `training_time_after_mini_epoch_seconds`, i.e. the
  training time from the start of training to the end of the last completed mini-epoch. For runs
  without that metric it falls back to the last `elapsed_training_time_seconds`.
- `scaling_detailed.parquet`: one row per metrics log step with `elapsed_training_time_seconds`,
  `total_num_samples`, `average_samples_per_second` and `loss_avg_mean`.
- `scaling_mini_epochs.parquet`: one row per completed mini-epoch with `completed_mini_epoch` and
  `training_time_after_mini_epoch_seconds`.

### generate_scaling_plots

Generates scaling plots and tables from the parquet/NDJSON files written by
`extract_scaling_data`.

```bash
uv run generate_scaling_plots standard --input scaling.parquet --type strong --y-scale log
uv run generate_scaling_plots loss --input scaling.parquet --type strong
uv run generate_scaling_plots detailed --input scaling_detailed.parquet
uv run generate_scaling_plots combined --strong-input strong.parquet --weak-input weak.parquet
```

The `standard`, `loss` and `combined` modes need a 1-node run as the reference for speedup and
efficiency.

## Suggested workflow

1. Extract the scaling data into a parquet file (on your HPC).
2. Copy the parquet file to your local machine.
3. Generate plots from the parquet file.
