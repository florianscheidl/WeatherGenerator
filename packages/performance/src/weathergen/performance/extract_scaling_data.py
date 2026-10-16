#!/usr/bin/env uv run python
"""Extract scaling data from WeatherGenerator runs.

Reads ``<shared-work-dir>/results/<run_id>/<run_id>_train_metrics.json`` for each run and writes
three tables (parquet + csv):

- ``<output>``: one row per run with num_nodes, startup_time_seconds, training_time and
  loss_avg_mean. ``training_time`` is the last ``training_time_after_mini_epoch_seconds``, i.e.
  the training time at the end of the last completed mini-epoch. Runs logged before that metric
  existed fall back to the last ``elapsed_training_time_seconds``.
- ``<output stem>_detailed``: one row per metrics log step with elapsed_training_time_seconds,
  total_num_samples, average_samples_per_second and loss_avg_mean.
- ``<output stem>_mini_epochs``: one row per completed mini-epoch with
  training_time_after_mini_epoch_seconds.
"""

import argparse
import ast
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MINI_EPOCH_TIME = "training_time_after_mini_epoch_seconds"
ELAPSED_TIME = "elapsed_training_time_seconds"

NUMERIC_COLUMNS = [
    "startup_time_seconds",
    "loss_avg_mean",
    "num_samples",
    "total_num_samples",
    "average_samples_per_second",
    ELAPSED_TIME,
    "completed_mini_epoch",
    MINI_EPOCH_TIME,
]


def extract_num_nodes_from_output(run_id: str, logs_base_dir: Path) -> int | None:
    """Extract num_nodes from output.*.txt file in the run directory.

    Looks for 'nNodes' pattern in output files.
    """
    run_log_dir = logs_base_dir / run_id
    if not run_log_dir.exists():
        return None

    # Look for output.*.txt files
    output_files = list(run_log_dir.glob("output.*.txt"))
    if not output_files:
        # Fallback to err files if no output files found
        output_files = list(run_log_dir.glob("weathergen.*.err"))

    for output_file in output_files:
        try:
            content = output_file.read_text()
        except OSError:
            continue
        # Look for nNodes pattern: "nNodes 128" (space-separated, as in NCCL logs)
        match = re.search(r"nNodes\s+(\d+)", content, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def load_train_metrics(run_id: str, shared_work_dir: Path) -> pd.DataFrame | None:
    """Load the train-stage rows of a run's metrics file.

    The train logger writes NaN/inf as strings, so the metric columns are coerced to numbers.
    """
    metrics_path = shared_work_dir / "results" / run_id / f"{run_id}_train_metrics.json"
    if not metrics_path.exists():
        logger.warning(f"Metrics file not found for {run_id}: {metrics_path}")
        return None
    try:
        df = pd.read_json(metrics_path, lines=True)
    except ValueError as e:
        logger.warning(f"Could not read metrics file for {run_id}: {e}")
        return None
    if len(df) == 0:
        return None

    if "stage" in df.columns:
        df = df[df["stage"] == "train"].reset_index(drop=True)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _first_value(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    val = df[col].dropna()
    return val.iloc[0] if len(val) > 0 else None


def _last_value(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    val = df[col].dropna()
    return val.iloc[-1] if len(val) > 0 else None


def extract_run_summary(df: pd.DataFrame) -> dict:
    """Extract run-level metrics: startup time, final training time and final loss."""
    training_time = _last_value(df, MINI_EPOCH_TIME)
    if training_time is None:
        # runs logged before training_time_after_mini_epoch_seconds was introduced
        training_time = _last_value(df, ELAPSED_TIME)

    return {
        "startup_time_seconds": _first_value(df, "startup_time_seconds"),
        "training_time": training_time,
        "loss_avg_mean": _last_value(df, "loss_avg_mean"),
    }


def extract_detailed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per metrics log step with elapsed training time and loss.

    Elapsed time and loss are logged in the same row. Older runs logged the elapsed time in a
    separate row after the loss row, in which case the closest preceding loss is used.
    """
    if ELAPSED_TIME not in df.columns or "loss_avg_mean" not in df.columns:
        return pd.DataFrame()

    loss = df["loss_avg_mean"].ffill()
    # older runs logged the number of samples as total_num_samples
    num_samples = df["total_num_samples"] if "total_num_samples" in df.columns else None
    if num_samples is None and "num_samples" in df.columns:
        num_samples = df["num_samples"]

    timing = df[ELAPSED_TIME].notna() & loss.notna()
    detailed = pd.DataFrame(
        {
            ELAPSED_TIME: df.loc[timing, ELAPSED_TIME],
            "total_num_samples": num_samples[timing] if num_samples is not None else None,
            "average_samples_per_second": (
                df.loc[timing, "average_samples_per_second"]
                if "average_samples_per_second" in df.columns
                else None
            ),
            "loss_avg_mean": loss[timing],
        }
    )
    return detailed.reset_index(drop=True)


def extract_mini_epoch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per completed mini-epoch with the training time at its end."""
    if MINI_EPOCH_TIME not in df.columns:
        return pd.DataFrame()

    rows = df[MINI_EPOCH_TIME].notna()
    mini_epochs = pd.DataFrame(
        {
            "completed_mini_epoch": (
                df.loc[rows, "completed_mini_epoch"].astype("Int64")
                if "completed_mini_epoch" in df.columns
                else None
            ),
            MINI_EPOCH_TIME: df.loc[rows, MINI_EPOCH_TIME],
        }
    )
    return mini_epochs.reset_index(drop=True)


def parse_run_ids(run_ids_str: list[str]) -> list[tuple[int | None, str]]:
    """Parse run-ids argument which can be:

    1. A list of run-ids (old format):
       ["run1", "run2"] -> [(None, "run1"), (None, "run2")]
    2. A dict mapping num_nodes to run-ids (new format):
       "{1: run1, 4: run2}" -> [(1, "run1"), (4, "run2")]

    Returns list of (num_nodes, run_id) tuples.
    """
    if len(run_ids_str) == 1:
        # Check if it looks like a dict: "{key: value, ...}"
        stripped = run_ids_str[0].strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            # Parse as dict format: {num_nodes: run_id, ...}
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    # Convert string keys to int if needed
                    result = []
                    for k, v in parsed.items():
                        key = int(k) if isinstance(k, str) and k.isdigit() else k
                        result.append((key, str(v)))
                    return result
            except (ValueError, SyntaxError):
                pass

        # Single run-id or comma-separated list
        run_ids = [r.strip() for r in run_ids_str[0].split(",") if r.strip()]
        return [(None, run_id) for run_id in run_ids]

    # Multiple arguments - treat as list of run-ids
    return [(None, run_id) for run_id in run_ids_str]


def _with_run_columns(df: pd.DataFrame, run_id: str, num_nodes: int | None) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "num_nodes", num_nodes)
    df.insert(0, "run_id", run_id)
    return df


def _write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Extract scaling data from WeatherGenerator runs. "
            "Run-ids can be provided as a list (--run-ids run1 run2) or as a dict "
            "mapping num_nodes to run-ids (--run-ids '{1: run1, 4: run2}'). "
            "If num_nodes is not provided in the dict, it will be extracted "
            "from output.*.txt files."
        )
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help=(
            "Run-ids to process. Can be: (1) list: run1 run2 run3, or "
            "(2) dict: '{1: run1, 4: run2, 8: run3}'"
        ),
    )
    parser.add_argument(
        "--logs-base-dir",
        type=Path,
        default=Path("logs"),
        help="Base directory for run logs (default: logs relative to current dir)",
    )
    parser.add_argument(
        "--shared-work-dir",
        type=Path,
        required=True,
        help="Base directory for shared work/results (contains results/<run_id>/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scaling_data.parquet"),
        help="Output parquet file path",
    )

    args = parser.parse_args()

    run_id_mapping = parse_run_ids(args.run_ids)
    if not run_id_mapping:
        sys.exit("Error: No run-ids provided")

    results = []
    all_detailed = []
    all_mini_epochs = []

    for num_nodes, run_id in run_id_mapping:
        # If num_nodes not provided, extract from output.*.txt file
        if num_nodes is None:
            num_nodes = extract_num_nodes_from_output(run_id, args.logs_base_dir)

        df = load_train_metrics(run_id, args.shared_work_dir)
        if df is None:
            continue

        results.append({"run_id": run_id, "num_nodes": num_nodes, **extract_run_summary(df)})

        detailed = extract_detailed_metrics(df)
        if len(detailed) > 0:
            all_detailed.append(_with_run_columns(detailed, run_id, num_nodes))

        mini_epochs = extract_mini_epoch_metrics(df)
        if len(mini_epochs) > 0:
            all_mini_epochs.append(_with_run_columns(mini_epochs, run_id, num_nodes))

        logger.info(
            f"{run_id} ({num_nodes} nodes): {len(detailed)} detailed metric entries, "
            f"{len(mini_epochs)} completed mini-epochs"
        )

    if not results:
        sys.exit("No data extracted")

    df = pd.DataFrame(results)
    if "num_nodes" in df.columns:
        df = df.sort_values("num_nodes").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_table(df, args.output)

    logger.info("\nSummary:")
    logger.info(f"  - Extracted {len(results)} run summaries to {args.output}")

    if all_detailed:
        detailed_output = args.output.with_name(f"{args.output.stem}_detailed{args.output.suffix}")
        detailed_df = pd.concat(all_detailed, ignore_index=True)
        _write_table(detailed_df, detailed_output)
        logger.info(
            f"  - Extracted {len(detailed_df)} detailed metric entries to {detailed_output}"
        )

    if all_mini_epochs:
        mini_epoch_output = args.output.with_name(
            f"{args.output.stem}_mini_epochs{args.output.suffix}"
        )
        mini_epoch_df = pd.concat(all_mini_epochs, ignore_index=True)
        _write_table(mini_epoch_df, mini_epoch_output)
        logger.info(f"  - Extracted {len(mini_epoch_df)} mini-epoch entries to {mini_epoch_output}")


if __name__ == "__main__":
    main()
