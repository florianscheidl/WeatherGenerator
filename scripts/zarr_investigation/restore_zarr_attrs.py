"""Restore what a zarr v2 -> v3 conversion dropped, using the original v2 store as source.

1. Arrays/groups present in v2 but absent from v3 (no `zarr.json`) are copied over as
   native v3 nodes, with their dtype, chunking, fill value and attributes. Arrays larger
   than --max-copy-gib are refused (they belong in a proper conversion, not this fixup).
2. For every node with a v2 `.zattrs`, those attributes are merged into the `attributes`
   field of the matching v3 `zarr.json`. Keys already present in v3 win.
`_ARRAY_DIMENSIONS` (v2/xarray convention) becomes v3 `dimension_names` if unset.
If the v3 root carries consolidated metadata, it is rebuilt afterwards so readers
don't keep seeing the stale copy.

Usage: python restore_zarr_attrs.py OLD_V2.zarr NEW_V3.zarr [--dry-run] [--max-copy-gib N]
Requires zarr>=3.1.
"""

import argparse
import json
import os
from pathlib import Path

import zarr


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("v2", type=Path)
    p.add_argument("v3", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-copy-gib", type=float, default=1.0)
    a = p.parse_args()

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


if __name__ == "__main__":
    main()
