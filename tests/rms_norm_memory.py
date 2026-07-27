# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Does swapping LayerNorm for RMSNorm actually save activation memory on this backend?

Run it on the target hardware -- the answer is a property of the build, not of the model.
Use the environment that is already synced rather than `uv run --extra gpu`, which may go
off and re-resolve (and rebuild flash-attn) before it prints anything:

    srun -n1 .venv/bin/python tests/rms_norm_memory.py

Nothing here imports flash_attn, so it needs only torch and a visible GPU.

`aten::rms_norm` has no autocast registration, so unlike `layer_norm` it is not promoted
to float32. That alone only helps if the call also reaches the *fused* kernel: the backward
of `aten::_fused_rms_norm` recomputes the float32 upcast from the bfloat16 input, whereas
the composite fallback stores float32 copies and gives back most of the saving. A backend
with no fused kernel falls back silently, so this checks which op actually ran.

Reports per norm: bytes the autograd graph retains, output dtype, whether dispatch reached
the fused kernel, and agreement with a float32 reference. Where it did not, expect RMSNorm
to retain about as much as the LayerNorm it replaces -- and the swap to be pointless.
"""

import logging
import warnings

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from weathergen.model.norms import RMSNorm

_logger = logging.getLogger(__name__)

N, D = 8192, 2048


class SavedBytes(torch.autograd.graph.saved_tensors_hooks):
    """Distinct storages the autograd graph packs away, ignoring pre-existing tensors."""

    def __init__(self, ignore=()):
        self.storages: dict[int, tuple[int, str]] = {}
        self.ignore = {t.untyped_storage().data_ptr() for t in ignore}
        super().__init__(self._pack, lambda t: t)

    def _pack(self, t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in self.ignore:
            self.storages[ptr] = (t.untyped_storage().nbytes(), str(t.dtype))
        return t

    @property
    def total(self) -> int:
        return sum(nbytes for nbytes, _ in self.storages.values())


class RecordOps(TorchDispatchMode):
    """Names of the aten ops a call actually dispatches to."""

    def __init__(self):
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def dispatched_ops(fn) -> list[str]:
    """Did the call reach `aten::_fused_rms_norm`, or decompose into the composite fallback?

    The fallback also emits a "Cannot dispatch to fused implementation" warning, but only
    for a dtype mismatch -- a backend with no fused kernel at all falls back silently, so
    the op that actually ran is the only trustworthy signal.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with RecordOps() as rec:
            fn()
    return rec.ops


def measure(label: str, module: torch.nn.Module, x: torch.Tensor, unit: int, device: str) -> None:
    acct = SavedBytes(ignore=(x, *module.parameters()))
    with torch.autocast(device_type=device, dtype=torch.bfloat16), acct:
        out = module(x)
    ops = dispatched_ops(lambda: module(x)) if isinstance(module, RMSNorm) else None
    fused = None if ops is None else any("_fused_rms_norm" in op for op in ops)
    _logger.info(
        "  %-38s retained %5.2fu  out %-8s %s",
        label,
        acct.total / unit,
        str(out.dtype).replace("torch.", ""),
        "" if fused is None else f"fused={fused}",
    )
    if fused is False:
        # the composite fallback is what makes the swap a regression, so name what ran
        _logger.info("      fell back to: %s", ", ".join(dict.fromkeys(ops)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    # emitted before anything touches the GPU, so a silent run means the environment is
    # still resolving rather than the measurement being slow
    _logger.info("torch %s, cuda available: %s", torch.__version__, torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        _logger.warning(
            "No CUDA device: layer_norm is not promoted to float32 under CPU autocast, so "
            "these numbers do not reflect the GPU behaviour this compares."
        )

    unit = N * D * 2  # one bfloat16 activation
    x = torch.randn(N, D, device=device, dtype=torch.bfloat16, requires_grad=True)

    _logger.info("N=%d D=%d  (u = %.0f MiB, one bfloat16 activation)", N, D, unit / 2**20)
    _logger.info("per-norm, over the full embedding dimension:")
    measure(
        "LayerNorm (elementwise_affine=False)",
        torch.nn.LayerNorm(D, elementwise_affine=False).to(device),
        x,
        unit,
        device,
    )
    measure(
        "RMSNorm (elementwise_affine=False)",
        RMSNorm(D, elementwise_affine=False).to(device),
        x,
        unit,
        device,
    )
    measure("RMSNorm (elementwise_affine=True)", RMSNorm(D).to(device), x, unit, device)

    _logger.info("agreement with a float32 reference (max relative):")
    ref = torch.nn.functional.rms_norm(x.float(), [D], None, 1e-6)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        got = RMSNorm(D, elementwise_affine=False).to(device)(x)
    rel = ((got.float() - ref) / ref.abs().clamp_min(1e-3)).abs().max().item()
    _logger.info("  RMSNorm vs float32 rms_norm            %.2e", rel)


if __name__ == "__main__":
    main()
