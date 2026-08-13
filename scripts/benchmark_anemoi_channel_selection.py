#!/usr/bin/env python3
"""Benchmark channel-selection strategies against one Anemoi Zarr data array."""

import argparse
import hashlib
import json
import logging
import multiprocessing
import resource
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import anemoi.datasets as anemoi_datasets
import numpy as np
import zarr
from numpy.typing import NDArray
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import LocalStore, WrapperStore

from weathergen.common import config
from weathergen.datasets.anemoi_channel_selection import coalesced_ranges
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.train.utils import cfg_keys_to_filter, get_active_stage_config

logger = logging.getLogger(__name__)


class CountingStore(WrapperStore[LocalStore]):
    """Count buffers returned by a Zarr v3 store without copying their contents."""

    def __init__(self, store: LocalStore) -> None:
        super().__init__(store)
        self.keys: Counter[str] = Counter()
        self.bytes = 0

    def _record(self, key: str, value: Buffer | None) -> None:
        self.keys[key] += 1
        if value is not None:
            self.bytes += len(value.as_array_like())

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        value = await self._store.get(key, prototype, byte_range)
        self._record(key, value)
        return value

    async def _get_many(
        self,
        requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]],
    ) -> Any:
        async for key, value in self._store._get_many(requests):
            self._record(key, value)
            yield key, value

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        requests = list(key_ranges)
        values = await self._store.get_partial_values(prototype, requests)
        for (key, _), value in zip(requests, values, strict=True):
            self._record(key, value)
        return values

    def reset(self) -> None:
        self.keys.clear()
        self.bytes = 0


def _select_from_full(values: NDArray[Any], channels: list[int]) -> NDArray[Any]:
    return values[:, channels, 0, :]


def _read_method(
    array: zarr.Array,
    method: str,
    time_start: int,
    time_end: int,
    channels: list[int],
) -> NDArray[Any]:
    selection = (slice(time_start, time_end), channels, 0, slice(None))
    if method == "full":
        return _select_from_full(array[time_start:time_end, :, :, :], channels)
    if method == "expanded":
        return np.concatenate(
            [array[time_start:time_end, channel : channel + 1, 0, :] for channel in channels],
            axis=1,
        )
    if method == "orthogonal":
        return array.get_orthogonal_selection(selection)
    if method == "coalesced":
        ranges = coalesced_ranges(channels)
        parts = [array[time_start:time_end, start:end, 0, :] for start, end in ranges]
        joined = np.concatenate(parts, axis=1)
        range_channels = [channel for start, end in ranges for channel in range(start, end)]
        positions = {channel: position for position, channel in enumerate(range_channels)}
        return joined[:, [positions[channel] for channel in channels], :]
    raise ValueError(f"Unknown method: {method}")


def _trial(payload: dict[str, Any]) -> dict[str, Any]:
    store = CountingStore(LocalStore(payload["dataset"], read_only=True))
    array = zarr.open_array(store=store, path="data", mode="r")
    store.reset()
    started = time.perf_counter_ns()
    values = _read_method(
        array,
        payload["method"],
        payload["time_start"],
        payload["time_end"],
        payload["channels"],
    )
    duration_ns = time.perf_counter_ns() - started
    digest = hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()
    metadata_names = {"zarr.json", ".zarray", ".zattrs", ".zgroup"}
    chunk_keys = [key for key in store.keys if key.rsplit("/", 1)[-1] not in metadata_names]
    return {
        "method": payload["method"],
        "repetition": payload["repetition"],
        "duration_ns": duration_ns,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "output_bytes": values.nbytes,
        "sha256": digest,
        "store_gets": sum(store.keys.values()),
        "store_unique_keys": len(store.keys),
        "store_bytes": store.bytes,
        "chunk_gets": sum(store.keys[key] for key in chunk_keys),
        "unique_chunk_keys": len(chunk_keys),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _resolve_filename(cf: Any, stream_info: Any) -> Path:
    filename = Path(stream_info.filenames[0])
    if filename.exists():
        return filename
    candidates = [Path(path) / filename for path in cf.data_paths]
    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve {filename}; tried {candidates}")
    return resolved


def _build_reader(args: argparse.Namespace) -> tuple[Any, DataReaderAnemoi, Path]:
    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        None,
        {},
        cli_overwrite,
    )
    training_cfg = cf.training_config
    validation_cfg = get_active_stage_config(
        training_cfg, cf.get("validation_config", {}), cfg_keys_to_filter
    )
    test_cfg = get_active_stage_config(
        validation_cfg, cf.get("test_config", {}), cfg_keys_to_filter
    )
    tw_handler = TimeWindowHandler(
        test_cfg.start_date,
        test_cfg.end_date,
        test_cfg.time_window_len,
        test_cfg.time_window_step,
    )
    stream_info = cf.streams[args.stream]
    if stream_info.type != "anemoi":
        raise ValueError(f"{args.stream} uses {stream_info.type}, not the base Anemoi reader")
    if stream_info.get("anemoi_config"):
        raise ValueError("This benchmark currently requires a direct filename, not anemoi_config")
    stream_info["data_paths"] = cf.get("data_paths", [])
    filename = _resolve_filename(cf, stream_info)
    reader = DataReaderAnemoi(tw_handler, filename, stream_info, "test")
    return cf, reader, filename


def _array_metadata(filename: Path) -> dict[str, Any]:
    array = zarr.open_array(store=filename, path="data", mode="r")
    metadata = array.metadata.to_dict()
    return {
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "shards": list(array.shards) if array.shards is not None else None,
        "dtype": str(array.dtype),
        "zarr_format": metadata.get("zarr_format"),
        "chunk_grid": metadata.get("chunk_grid"),
        "codecs": metadata.get("codecs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run-id", default="nlwqm5vw")
    parser.add_argument("--mini-epoch", type=int, default=-1)
    parser.add_argument("--private-config", type=Path)
    parser.add_argument("--stream", default="ERA5")
    parser.add_argument("--temporal-index", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--options", nargs="*", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    _, reader, filename = _build_reader(args)
    if reader.ds is None:
        raise RuntimeError("Resolved reader is empty")
    time_indices, _ = reader._get_dataset_idxs(args.temporal_index)
    dates = reader.ds.dates[time_indices]
    raw = anemoi_datasets.open_dataset(filename)
    raw_indices = [int(np.searchsorted(raw.dates, date)) for date in dates]
    if raw_indices != list(range(raw_indices[0], raw_indices[-1] + 1)):
        raise ValueError(f"Native benchmark requires contiguous raw time indices: {raw_indices}")
    channels = list(dict.fromkeys([*map(int, reader.target_idx), *map(int, reader.geoinfo_idx)]))
    if not channels:
        raise ValueError("No target or geoinfo channels selected")

    methods = ["full", "expanded", "orthogonal", "coalesced"]
    header = {
        "schema_version": 1,
        "experiment": "INF-E013b",
        "from_run_id": args.from_run_id,
        "stream": args.stream,
        "dataset": str(filename),
        "temporal_index": args.temporal_index,
        "raw_time_indices": raw_indices,
        "selected_channels": channels,
        "selected_channel_names": [raw.variables[index] for index in channels],
        "coalesced_ranges": coalesced_ranges(channels),
        "array": _array_metadata(filename),
        "anemoi_tree": str(raw.tree()),
        "anemoi_version": getattr(anemoi_datasets, "__version__", "unknown"),
        "zarr_version": zarr.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    records = []
    for repetition in range(args.repetitions):
        order = methods[repetition % len(methods) :] + methods[: repetition % len(methods)]
        for method in order:
            payload = {
                "dataset": str(filename),
                "time_start": raw_indices[0],
                "time_end": raw_indices[-1] + 1,
                "channels": channels,
                "method": method,
                "repetition": repetition,
            }
            with ctx.Pool(1) as pool:
                record = pool.apply(_trial, (payload,))
            records.append(record)
            with args.output.open("a") as output:
                output.write(json.dumps({"header": header, "result": record}) + "\n")

    hashes = {record["sha256"] for record in records}
    if len(hashes) != 1:
        raise RuntimeError(f"Selection methods returned different values: {hashes}")
    logger.info(
        "Benchmark completed: %s",
        json.dumps({"output": str(args.output), "header": header, "results": records}, indent=2),
    )


if __name__ == "__main__":
    main()
