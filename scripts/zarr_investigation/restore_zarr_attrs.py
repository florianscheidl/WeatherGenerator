"""Restore what a zarr v2 -> v3 conversion dropped, using the original v2 store as source.

1. Arrays/groups present in v2 but absent from v3 (no `zarr.json`) are copied over as
   native v3 nodes, with their dtype, chunking, fill value and attributes. Arrays larger
   than --max-copy-gib are refused (they belong in a proper conversion, not this fixup).
2. For every node with a v2 `.zattrs`, those attributes are merged into the `attributes`
   field of the matching v3 `zarr.json`. Keys already present in v3 win.
`_ARRAY_DIMENSIONS` (v2/xarray convention) becomes v3 `dimension_names` if unset.
If the v3 root carries consolidated metadata, it is rebuilt afterwards so readers
don't keep seeing the stale copy.

Everything restored describes the v2 `data` array, so the script refuses to touch a v3 store
whose `data` has a different shape (e.g. one month sliced out of a 45-year dataset: the v2
`dates`, `start_date` and `end_date` would be wrong for it). After restoring, it checks that
the v3 metadata matches its `data` (dates and frequency against the time axis, variables and
statistics against the variable axis, coordinates against the grid) and fails otherwise.
--check-only runs both checks without writing anything, e.g. on an already restored store.

Usage: python restore_zarr_attrs.py OLD_V2.zarr NEW_V3.zarr [--dry-run | --check-only]
       [--max-copy-gib N]
Requires zarr>=3.1.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import zarr

# Anemoi datasets layout: data is (time, variable, ensemble, grid point).
DATA = "data"
PER_VARIABLE = {"mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"}
PER_POINT = {"latitudes", "longitudes"}
_FREQUENCY_UNITS = {"s": "s", "m": "m", "h": "h", "d": "D"}


def v2_attrs(v2: Path) -> dict[str, dict]:
    """Map node path ('' = root) -> attributes. Prefer .zmetadata: avoids walking chunk files."""
    zmeta = v2 / ".zmetadata"
    if zmeta.exists():
        entries = json.loads(zmeta.read_text())["metadata"]
        return {
            k.removesuffix(".zattrs").rstrip("/"): v
            for k, v in entries.items()
            if k.endswith(".zattrs") and v
        }
    out = {}
    for dirpath, dirnames, filenames in os.walk(v2):
        here = Path(dirpath)
        if ".zarray" in filenames:
            dirnames[:] = []  # arrays have no children; skip chunk subdirs
        if ".zattrs" in filenames:
            attrs = json.loads((here / ".zattrs").read_text())
            if attrs:
                rel = here.relative_to(v2).as_posix()
                out["" if rel == "." else rel] = attrs
    return out


def split_dims(attrs: dict) -> tuple[dict, list[str] | None]:
    attrs = dict(attrs)
    return attrs, attrs.pop("_ARRAY_DIMENSIONS", None)


def copy_missing_nodes(v2: Path, v3: Path, dry_run: bool, max_copy_bytes: int) -> list[str]:
    """Copy v2 groups/arrays that have no v3 counterpart. Returns their paths."""
    src = zarr.open_group(str(v2), mode="r", zarr_format=2)
    missing = [
        (path, node)
        for path, node in sorted(src.members(max_depth=None))
        if not (v3 / path / "zarr.json").exists()
    ]
    too_big = [
        f"{p} ({n.nbytes / 2**30:.1f} GiB)"
        for p, n in missing
        if isinstance(n, zarr.Array) and n.nbytes > max_copy_bytes
    ]
    if too_big:
        raise SystemExit(
            f"refusing to copy arrays above --max-copy-gib: {too_big}\n"
            f"No {v3}/<name>/zarr.json found for them. This script completes a conversion; "
            "it does not do one. Check that the v3 path is right and that the converter "
            "wrote these arrays where expected."
        )
    if not missing:
        return []

    dst = zarr.open_group(str(v3), mode="r+", zarr_format=3, use_consolidated=False)
    for path, node in missing:
        attrs, dims = split_dims(node.attrs.asdict())
        if isinstance(node, zarr.Group):
            print(f"copy group {path}")
            if not dry_run:
                dst.create_group(path, attributes=attrs)
            continue
        print(
            f"copy array {path}: shape={node.shape} dtype={node.dtype} ({node.nbytes / 2**20:.1f} MiB)"
        )
        if dry_run:
            continue
        new = dst.create_array(
            path,
            shape=node.shape,
            dtype=node.dtype,
            chunks=node.chunks,
            fill_value=node.fill_value,
            attributes=attrs,
            dimension_names=dims,
        )
        new[...] = node[...]
    return [path for path, _ in missing]


def patch_attributes(v2: Path, v3: Path, dry_run: bool, skip: set[str]) -> int:
    """Merge v2 attributes into existing v3 nodes. Returns the number of nodes changed."""
    patched = 0
    for node, attrs in sorted(v2_attrs(v2).items()):
        if node in skip:
            continue
        zj = v3 / node / "zarr.json"
        if not zj.exists():
            print(f"WARNING: {node or '/'}: no zarr.json in v3 store, skipped")
            continue
        meta = json.loads(zj.read_text())
        attrs, dims = split_dims(attrs)
        new_attrs = {**attrs, **meta.get("attributes", {})}
        added = sorted(set(new_attrs) - set(meta.get("attributes", {})))
        set_dims = meta.get("node_type") == "array" and dims and not meta.get("dimension_names")
        if not added and not set_dims:
            continue
        changes = ([f"+{added}"] if added else []) + (
            [f"dimension_names={dims}"] if set_dims else []
        )
        print(f"attrs {node or '/'}: {', '.join(changes)}")
        if dry_run:
            continue
        meta["attributes"] = new_attrs
        if set_dims:
            meta["dimension_names"] = dims
        tmp = zj.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(zj)  # atomic on POSIX
        patched += 1
    return patched


def source_mismatch(v2: Path, v3: Path) -> list[str]:
    """Problems that make the v2 metadata wrong for the v3 store: a differently shaped `data`."""
    src = zarr.open_group(str(v2), mode="r", zarr_format=2)
    dst = zarr.open_group(str(v3), mode="r", zarr_format=3, use_consolidated=False)
    if DATA not in src or DATA not in dst:
        print(f"WARNING: '{DATA}' missing from v2 or v3 store, source shape check skipped")
        return []
    if src[DATA].shape != dst[DATA].shape:
        return [
            f"v3 {DATA} has shape {dst[DATA].shape}, v2 {DATA} has {src[DATA].shape}: the v3 "
            "store holds a different slice, so the v2 dates, start/end dates and statistics "
            "don't describe it"
        ]
    return []


def _parse_frequency(frequency: str) -> np.timedelta64:
    match = re.fullmatch(r"(\d+)([smhd])", frequency)
    if match is None:
        raise ValueError(f"Unsupported frequency {frequency!r}")
    return np.timedelta64(int(match[1]), _FREQUENCY_UNITS[match[2]])


def check_metadata(v3: Path) -> list[str]:
    """Problems where the v3 metadata doesn't match the shape of its `data` array."""
    group = zarr.open_group(str(v3), mode="r", zarr_format=3, use_consolidated=False)
    if DATA not in group:
        return [f"no '{DATA}' array"]
    shape = group[DATA].shape
    if len(shape) != 4:
        return [f"{DATA} has shape {shape}, expected (time, variable, ensemble, grid point)"]
    num_steps, num_variables, _, num_points = shape
    attrs = group.attrs.asdict()
    errors = []

    def expect(what: str, actual: object, expected: object) -> None:
        if actual != expected:
            errors.append(f"{what} is {actual}, expected {expected} from {DATA} {shape}")

    start = np.datetime64(attrs["start_date"]) if "start_date" in attrs else None
    end = np.datetime64(attrs["end_date"]) if "end_date" in attrs else None
    frequency = _parse_frequency(attrs["frequency"]) if "frequency" in attrs else None
    if start is not None and end is not None and frequency is not None:
        expect(
            "number of steps from start_date/end_date/frequency",
            (end - start) // frequency + 1,
            num_steps,
        )
    if "dates" in group:
        dates = group["dates"][:]
        expect("len(dates)", len(dates), num_steps)
        if len(dates) > 0 and start is not None:
            expect("start_date", start, dates[0])
        if len(dates) > 0 and end is not None:
            expect("end_date", end, dates[-1])
        steps = np.unique(np.diff(dates))
        if frequency is not None and len(steps) > 0 and (len(steps) > 1 or steps[0] != frequency):
            errors.append(f"dates steps are {steps[:5]}, expected only {frequency}")
    if "variables" in attrs:
        expect("len(variables)", len(attrs["variables"]), num_variables)
    for name, array in group.arrays():
        if name in PER_VARIABLE or name.startswith("statistics_tendencies_"):
            expect(f"{name} shape", array.shape, (num_variables,))
        elif name in PER_POINT:
            expect(f"{name} shape", array.shape, (num_points,))
    return errors


def fail_if(errors: list[str], v3: Path) -> None:
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(f"{v3}: metadata does not match the data ({len(errors)} problem(s))")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("v2", type=Path)
    p.add_argument("v3", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--check-only", action="store_true", help="Only check the v3 store against v2 and its data."
    )
    p.add_argument("--max-copy-gib", type=float, default=1.0)
    a = p.parse_args()

    mismatch = source_mismatch(a.v2, a.v3)
    if a.check_only:
        fail_if(mismatch + check_metadata(a.v3), a.v3)
        print("metadata matches the data")
        return
    # Refuse before writing anything: restoring would put the v2 time axis on a different slice.
    fail_if(mismatch, a.v3)

    copied = copy_missing_nodes(a.v2, a.v3, a.dry_run, int(a.max_copy_gib * 2**30))
    # Copied nodes already carry their attributes; in a dry run they don't exist yet,
    # so leave them out of the attribute pass to avoid spurious warnings.
    patched = patch_attributes(a.v2, a.v3, a.dry_run, skip=set(copied))

    root = json.loads((a.v3 / "zarr.json").read_text())
    if root.get("consolidated_metadata") and (copied or patched) and not a.dry_run:
        zarr.consolidate_metadata(str(a.v3), zarr_format=3)
        print("rebuilt consolidated metadata")
    suffix = " (dry run)" if a.dry_run else ""
    print(f"copied {len(copied)} node(s), patched attributes on {patched} node(s){suffix}")
    if not a.dry_run:
        fail_if(check_metadata(a.v3), a.v3)
        print("metadata matches the data")


if __name__ == "__main__":
    main()
