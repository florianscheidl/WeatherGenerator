# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Count the operations that make the host wait for the device (host-device syncs), per call site.

Whether an operation syncs depends on the operation, not on the GPU: reading a device value on
the host (`.item()`, `int(t)`, `if t:`), an output shape that depends on device data (boolean-mask
indexing, `nonzero`, `repeat_interleave` without `output_size`), and copies between host and
device. `SyncCounter` is a `TorchDispatchMode` that sees every aten op of the main thread and of
the autograd threads (backward), so the counts are exact and free of timing noise, and the same
on MPS as on CUDA. `torch.cuda.set_sync_debug_mode` only exists on CUDA; use it there to check
these counts.

Kinds counted (the first three always block the host; the copies do on CUDA):

- `scalar_read`: `aten._local_scalar_dense`, the device value of a 0-dim tensor read on the host.
- `data_dependent_shape`: ops whose output size is computed from device data.
- `d2h_copy`: copies from the device to the host.
- `h2d_copy`: copies from host memory to the device; `pinned` tells whether the source was
  pinned (a non_blocking copy from pinned memory does not block on CUDA; MPS has no pinned
  memory, so there every copy counts as unpinned).
- `h2d_tensor_ctor`: `torch.tensor(data, device=...)`, including indexing with a Python list.
  The host copy itself is not visible as an op; the device tensor appears via `lift_fresh`.

Not seen on MPS: a scalar setitem (`t[i] = 0.0`) creates the scalar on the device there, while
on CUDA it is a pageable host-to-device copy (sync-barrier trigger catalogue). Counts of the
copy kinds can therefore differ between devices; the first three kinds do not.

Sites are the innermost frame of weathergen code on the stack, or the autograd node for ops in
the backward pass. Counting costs tens of microseconds per op, so it is for counting runs, not
for timing runs.
"""

import dataclasses
import json
import sys
import threading
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

_DATA_DEPENDENT_SHAPE_OPS = {
    "aten.nonzero.default",
    "aten.masked_select.default",
    "aten._unique2.default",
    "aten.unique_dim.default",
    "aten.unique_consecutive.default",
}
_REPEAT_INTERLEAVE_OPS = {"aten.repeat_interleave.Tensor", "aten.repeat_interleave.self_Tensor"}
_INDEX_OPS = {"aten.index.Tensor", "aten.index_put.default", "aten.index_put_.default"}
_COPY_OPS = {"aten._to_copy.default", "aten.copy_.default"}

_THIS_FILE = __file__
_SRC_MARKER = f"{Path('src') / 'weathergen'}"
_CHECKPOINT_FILE = f"{Path('torch') / 'utils' / 'checkpoint.py'}"


@dataclasses.dataclass(frozen=True)
class SyncEvent:
    kind: str
    op: str
    site: str


def _code_site() -> tuple[str | None, bool]:
    """
    Innermost frame of weathergen code on the stack (outside this module), if any, and whether
    the op runs inside torch.utils.checkpoint (the recomputation of a checkpointed forward).
    """
    code_site, in_checkpoint = None, False
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if code_site is None and _SRC_MARKER in filename and filename != _THIS_FILE:
            relative = filename[filename.index(_SRC_MARKER) :]
            code_site = f"{relative}:{frame.f_lineno} ({frame.f_code.co_name})"
        if filename.endswith(_CHECKPOINT_FILE):
            in_checkpoint = True
        frame = frame.f_back
    return code_site, in_checkpoint


def _site() -> str:
    """
    Innermost frame of weathergen code, or the autograd node for ops of the backward pass.

    In the backward pass, a checkpointed forward is recomputed: its ops are attributed to their
    frame in the forward code, marked as recomputation.
    """
    code_site, in_checkpoint = _code_site()
    node = torch._C._current_autograd_node()
    if node is None:
        return code_site or "<outside weathergen>"
    node_name = type(node).__name__
    if in_checkpoint and code_site is not None:
        return f"backward ({node_name}), recompute: {code_site}"
    return f"backward: {node_name}"


def _is_bool_index(index) -> bool:
    return isinstance(index, torch.Tensor) and index.dtype in (torch.bool, torch.uint8)


class SyncCounter(TorchDispatchMode):
    """
    Count host-device syncs while active, per step (`next_step`) and per call site.

    Only ops on tensors of `device_types` count (by default every device but the CPU; pass
    {"cpu"} to count on the CPU, e.g. in tests).
    """

    def __init__(self, device_types: set[str] | None = None) -> None:
        super().__init__()
        self.device_types = device_types
        self.events: Counter[SyncEvent] = Counter()
        self.steps: list[Counter[str]] = [Counter()]
        self._lock = threading.Lock()

    def _on_device(self, tensor: torch.Tensor) -> bool:
        if self.device_types is None:
            return tensor.device.type != "cpu"
        return tensor.device.type in self.device_types

    def _classify(self, func, args, kwargs) -> tuple[str, str] | None:
        """(kind, op) of a syncing op, or None."""
        name = str(func)
        tensors = [a for a in tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)]
        if not any(self._on_device(t) for t in tensors) and name not in _COPY_OPS:
            return None

        if name == "aten._local_scalar_dense.default":
            return "scalar_read", name
        if name in _DATA_DEPENDENT_SHAPE_OPS:
            return "data_dependent_shape", name
        if name in _REPEAT_INTERLEAVE_OPS and kwargs.get("output_size") is None:
            return "data_dependent_shape", name
        if name in _INDEX_OPS and any(_is_bool_index(i) for i in tree_leaves(args[1])):
            return "data_dependent_shape", f"{name} (bool mask)"
        if name == "aten.lift_fresh.default":
            return "h2d_tensor_ctor", name
        if name in _COPY_OPS:
            return self._classify_copy(name, args, kwargs)
        return None

    def _classify_copy(self, name, args, kwargs) -> tuple[str, str] | None:
        if name == "aten.copy_.default":
            dst, src = args[0], args[1]
            dst_device = dst.device
        else:
            src = args[0]
            dst_device = kwargs.get("device") or src.device
            dst_device = torch.device(dst_device)
        src_on, dst_on = self._on_device(src), self._on_device_type(dst_device)
        if src_on and not dst_on and dst_device.type == "cpu":
            return "d2h_copy", name
        if not src_on and src.device.type == "cpu" and dst_on:
            pinned = "pinned" if src.is_pinned() else "unpinned"
            return "h2d_copy", f"{name} ({pinned})"
        return None

    def _on_device_type(self, device: torch.device) -> bool:
        if self.device_types is None:
            return device.type != "cpu"
        return device.type in self.device_types

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        classified = self._classify(func, args, kwargs)
        if classified is not None:
            kind, op = classified
            event = SyncEvent(kind, op, _site())
            with self._lock:  # backward ops run on autograd threads
                self.events[event] += 1
                self.steps[-1][kind] += 1
        return func(*args, **kwargs)

    def next_step(self) -> None:
        """Start counting the next step."""
        self.steps.append(Counter())

    def report(self) -> dict:
        """Counts per step and per (kind, op, site), most frequent first."""
        steps = [dict(s) for s in self.steps if s]
        num_steps = max(len(steps), 1)
        sites = [
            {
                "kind": e.kind,
                "op": e.op,
                "site": e.site,
                "count": n,
                "per_step": n / num_steps,
            }
            for e, n in self.events.most_common()
        ]
        totals = Counter()
        for s in self.steps:
            totals.update(s)
        return {"num_steps": len(steps), "totals": dict(totals), "steps": steps, "sites": sites}


def format_report(report: dict, max_sites: int = 40) -> str:
    """Totals per kind and the most frequent call sites, per step."""
    num_steps = max(report["num_steps"], 1)
    lines = [f"host-device syncs over {report['num_steps']} steps (per step):"]
    for kind, n in sorted(report["totals"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {kind:22s} {n / num_steps:8.1f}")
    lines.append("call sites:")
    for s in report["sites"][:max_sites]:
        lines.append(f"  {s['per_step']:8.1f}  {s['kind']:20s} {s['op']:45s} {s['site']}")
    if len(report["sites"]) > max_sites:
        lines.append(f"  ... {len(report['sites']) - max_sites} more sites")
    return "\n".join(lines)


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
