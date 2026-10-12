#!/usr/bin/env python3
"""Time DataReaderAnemoi._get over the temporal windows of one stream.

The reader is built exactly as the training data loader builds it (same config merge,
stream config, frequency subsampling, and time-window handler for the chosen stage), then
``_get`` is called once per window and timed. One JSON line is written per window, framed
by a header (dataset layout, versions, job info) and a summary line.

Typical use, comparing two encodings of the same dataset:

  uv run python scripts/benchmark_anemoi_reader.py --stream ERA5 --output v2.jsonl
  uv run python scripts/benchmark_anemoi_reader.py --stream ERA5 --output v3.jsonl \\
      --filename /path/to/dataset-v3.zarr

Under ``srun --ntasks N`` every task reads a disjoint, interleaved subset of the windows
and writes its own ``<output>.rank<k>.jsonl``, which measures concurrent reads from the
same dataset.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import resource
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anemoi.datasets as anemoi_datasets
import numpy as np
import zarr
from numpy.typing import NDArray
from omegaconf import DictConfig

from weathergen.common import config
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import ReaderData, TimeWindowHandler
from weathergen.train.utils import (
    Stage,
    cfg_keys_to_filter,
    filter_config_by_enabled,
    get_active_stage_config,
)

logger = logging.getLogger(__name__)

_ARRAY_METADATA_KEYS = (
    "zarr_format",
    "chunk_grid",
    "chunk_key_encoding",
    "codecs",
    "compressor",
    "filters",
    "dimension_separator",
    "order",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Same config arguments as the train/inference entry points.
    parser.add_argument("--private-config", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    parser.add_argument("--options", nargs="+", default=[])
    parser.add_argument(
        "--stream", default=None, help="Stream name (default: first anemoi stream in config)."
    )
    parser.add_argument(
        "--stage",
        choices=["train", "val", "test"],
        default="train",
        help="Stage whose start/end dates and time windows are used.",
    )
    parser.add_argument(
        "--filename",
        type=Path,
        default=None,
        help="Read this dataset instead of the stream's filenames[0] (e.g. a zarr v3 copy).",
    )
    parser.add_argument(
        "--channels",
        choices=["all", "source", "target"],
        default="all",
        help="Channels passed to _get: every dataset variable, or the stream's selection.",
    )
    parser.add_argument(
        "--order",
        choices=["sequential", "shuffled"],
        default="sequential",
        help="Window order; training uses shuffled access.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Use every n-th window index.")
    parser.add_argument(
        "--num-windows", type=int, default=None, help="Cap on windows across all tasks."
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Windows read per task before timing starts."
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Store a hash of each window's data, to check two encodings return equal values.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output (default: results/anemoi_reader_benchmark/<stream>_<time>.jsonl).",
    )
    return parser.parse_args()


def _stage_config(cf: DictConfig, stage: Stage) -> DictConfig:
    """Stage config as the trainer derives it: validation from training, test from validation."""
    training_cfg = filter_config_by_enabled(cf.get("training_config"), cfg_keys_to_filter)
    if stage == "train":
        return training_cfg
    validation_cfg = get_active_stage_config(
        training_cfg, cf.get("validation_config", {}), cfg_keys_to_filter
    )
    if stage == "val":
        return validation_cfg
    return get_active_stage_config(validation_cfg, cf.get("test_config", {}), cfg_keys_to_filter)


def _resolve_filename(cf: DictConfig, stream_info: DictConfig, override: Path | None) -> Path:
    """Resolve a dataset path like MultiStreamDataSampler._init_stream_datasets does."""
    filename = override if override is not None else Path(stream_info.filenames[0])
    if filename.exists():
        return filename
    candidates = [Path(path) / filename for path in cf.get("data_paths", [])]
    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve {filename}; tried {candidates}")
    return resolved


def _select_stream(cf: DictConfig, name: str | None) -> tuple[str, DictConfig]:
    if name is None:
        anemoi_streams = [k for k, v in cf.streams.items() if v.type == "anemoi"]
        if not anemoi_streams:
            raise ValueError(f"No anemoi stream in config; streams: {list(cf.streams)}")
        name = anemoi_streams[0]
    stream_info = cf.streams[name]
    if stream_info.type != "anemoi":
        raise ValueError(f"Stream {name} has type {stream_info.type}, not anemoi")
    return name, stream_info


def _array_metadata(filename: Path) -> dict[str, Any]:
    """Layout of the dataset's data array: what differs between zarr v2 and v3 encodings."""
    array = zarr.open_array(store=str(filename), path="data", mode="r")
    metadata = array.metadata.to_dict()
    return {
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "shards": list(array.shards) if array.shards is not None else None,
        "dtype": str(array.dtype),
        **{key: metadata[key] for key in _ARRAY_METADATA_KEYS if key in metadata},
    }


def _task_layout() -> tuple[int, int]:
    """(rank, num_tasks) from SLURM, (0, 1) outside srun."""
    return int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))


def _window_indices(reader: DataReaderAnemoi, args: argparse.Namespace) -> NDArray[np.int64]:
    index_range = reader.time_window_handler.get_index_range()
    indices = np.arange(args.start_index, index_range.end, args.stride, dtype=np.int64)
    if args.order == "shuffled":
        indices = np.random.default_rng(args.seed).permutation(indices)
    if args.num_windows is not None:
        indices = indices[: args.num_windows]
    return indices


def _output_path(args: argparse.Namespace, stream: str, rank: int, num_tasks: int) -> Path:
    output = args.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        job = os.environ.get("SLURM_JOB_ID")
        name = f"{stream}_{stamp}" + (f"_{job}" if job else "")
        output = Path("results/anemoi_reader_benchmark") / f"{name}.jsonl"
    if num_tasks > 1:
        output = output.with_suffix(f".rank{rank}.jsonl")
    return output


def _summary(durations_ns: list[int], decoded_bytes: int, output_bytes: int) -> dict[str, Any]:
    if not durations_ns:
        return {"num_timed_windows": 0}
    ms = np.array(durations_ns) / 1e6
    total_s = ms.sum() / 1e3
    return {
        "num_timed_windows": len(ms),
        "total_seconds": total_s,
        "mean_ms": ms.mean(),
        "median_ms": float(np.median(ms)),
        "p90_ms": float(np.percentile(ms, 90)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": ms.max(),
        "windows_per_second": len(ms) / total_s,
        "decoded_mib_per_second": decoded_bytes / 2**20 / total_s,
        "output_mib_per_second": output_bytes / 2**20 / total_s,
    }


def _output_nbytes(rd: ReaderData) -> int:
    return rd.coords.nbytes + rd.geoinfos.nbytes + rd.data.nbytes + rd.datetimes.nbytes


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rank, num_tasks = _task_layout()

    cf = config.load_merge_configs(
        args.private_config,
        None,
        None,
        args.base_config,
        *args.config,
        {},
        config.from_cli_arglist(args.options),
    )
    stage: Stage = args.stage
    stage_cfg = _stage_config(cf, stage)
    stream, stream_info = _select_stream(cf, args.stream)
    if stream_info.get("anemoi_config"):
        raise ValueError(f"Stream {stream} uses anemoi_config; only filenames are supported")
    stream_info["data_paths"] = cf.get("data_paths", [])
    filename = _resolve_filename(cf, stream_info, args.filename)

    tw_handler = TimeWindowHandler(
        stage_cfg.start_date,
        stage_cfg.end_date,
        stage_cfg.time_window_len,
        stage_cfg.time_window_step,
    )
    started = time.perf_counter()
    reader = DataReaderAnemoi(tw_handler, filename, stream_info, stage)
    init_seconds = time.perf_counter() - started
    if reader.ds is None:
        raise RuntimeError(f"Reader for {filename} is empty over {tw_handler}")

    channels = {
        "all": list(range(len(reader.ds.variables))),
        "source": [int(i) for i in reader.source_idx],
        "target": [int(i) for i in reader.target_idx],
    }[args.channels]
    all_indices = _window_indices(reader, args)
    indices = all_indices[rank::num_tasks]
    # _read_window decodes every variable of each timestep and selects channels afterwards.
    bytes_per_step = int(np.prod(reader.ds.shape[1:])) * reader.ds.dtype.itemsize

    header = {
        "stream": stream,
        "dataset": str(filename),
        "stage": stage,
        "time_window_handler": str(tw_handler),
        "frequency": str(reader.ds.frequency),
        "dataset_shape": list(reader.ds.shape),
        "array": _array_metadata(filename),
        "channels": args.channels,
        "num_channels": len(channels),
        "order": args.order,
        "seed": args.seed,
        "num_windows_total": len(all_indices),
        "num_windows_task": len(indices),
        "warmup": args.warmup,
        "init_seconds": init_seconds,
        "rank": rank,
        "num_tasks": num_tasks,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "zarr_version": zarr.__version__,
        "zarr_async_concurrency": zarr.config.get("async.concurrency"),
        "zarr_threading_max_workers": zarr.config.get("threading.max_workers"),
        "anemoi_datasets_version": getattr(anemoi_datasets, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "argv": vars(args),
    }
    output = _output_path(args, stream, rank, num_tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    logger.info("Writing %d windows of %s to %s", len(indices), filename, output)

    durations_ns: list[int] = []
    decoded_bytes = 0
    output_bytes = 0
    num_empty = 0
    with output.open("w") as out:
        out.write(json.dumps({"header": header}, default=str) + "\n")
        for position, idx in enumerate(indices):
            t_idxs, _ = reader._get_dataset_idxs(idx)
            started_ns = time.perf_counter_ns()
            rd = reader._get(idx, channels)
            duration_ns = time.perf_counter_ns() - started_ns

            warmup = position < args.warmup
            num_steps = len(t_idxs)
            record: dict[str, Any] = {
                "idx": int(idx),
                "position": position,
                "warmup": warmup,
                "window_start": str(tw_handler.window(idx).start),
                "dataset_steps": [int(t_idxs[0]), int(t_idxs[-1])] if num_steps else None,
                "num_steps": num_steps,
                "num_points": rd.len(),
                "duration_ns": duration_ns,
                "output_bytes": _output_nbytes(rd),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            if args.checksum:
                data = np.ascontiguousarray(rd.data)
                record["blake2b"] = hashlib.blake2b(data.view(np.uint8)).hexdigest()
            out.write(json.dumps({"window": record}) + "\n")

            if rd.is_empty():
                num_empty += 1
            elif not warmup:
                durations_ns.append(duration_ns)
                decoded_bytes += num_steps * bytes_per_step
                output_bytes += record["output_bytes"]
            if (position + 1) % args.log_every == 0:
                logger.info(
                    "%d/%d windows, last %.1f ms",
                    position + 1,
                    len(indices),
                    duration_ns / 1e6,
                )

        summary = _summary(durations_ns, decoded_bytes, output_bytes) | {"num_empty": num_empty}
        out.write(json.dumps({"summary": summary}, default=float) + "\n")
    logger.info("Summary: %s", json.dumps(summary, default=float, indent=2))


if __name__ == "__main__":
    main()
