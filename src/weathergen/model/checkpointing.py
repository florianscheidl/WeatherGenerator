# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""Activation-checkpointing policy: which checkpoints of the model are active.

The model checkpoints at two nested levels (all non-reentrant):

  outer   a whole engine in `EncoderModule.forward` (embedding, local assimilation with the
          local-to-global adapter and the query aggregation, global assimilation)
  blocks  every block inside an engine (and the stream embedding, decoder, forecasting engine)

A block inside an outer checkpoint runs its forward three times per step (forward, outer
recompute, block recompute): the outer recompute only exists to rebuild the block inputs.
Switching an outer checkpoint off keeps those inter-block tensors in memory and saves one forward
of the engine in the backward pass. Checkpointing never changes values: with any setting the
outputs and gradients are bitwise equal (dropout RNG state is preserved by `checkpoint`).

Config section (all keys optional, all default to true = the behaviour without this section):

  activation_checkpointing:
    outer_engines: true          # or false, or a dict with the keys of OUTER_REGIONS
    blocks: true                 # or false, or a dict with the keys of BLOCK_REGIONS
"""

from collections.abc import Mapping

from torch.utils.checkpoint import checkpoint

OUTER_REGIONS = ("embedding", "assimilate_local", "global_assimilation")
BLOCK_REGIONS = (
    "embedding",  # stream embedding: embed, layers, readout
    "local_assimilation",
    "local2global",
    "aggregation",
    "global_assimilation",
    "forecasting",
    "prediction",  # decoder (target prediction engine)
    "target_coords",  # embedding of the target coordinates
    "latent_head",
)


def _resolve(value, regions: tuple[str, ...], what: str) -> dict[str, bool]:
    if isinstance(value, bool):
        return dict.fromkeys(regions, value)
    if value is None:
        return dict.fromkeys(regions, True)
    if not isinstance(value, Mapping) and not hasattr(value, "items"):
        raise ValueError(f"activation_checkpointing.{what} must be a bool or a mapping: {value!r}")
    out = dict.fromkeys(regions, True)
    for k, v in value.items():
        if k not in out:
            raise ValueError(
                f"activation_checkpointing.{what}: unknown region {k!r}, use {regions}"
            )
        out[k] = bool(v)
    return out


class CheckpointPolicy:
    """Resolved switches of the `activation_checkpointing` config section."""

    def __init__(self, outer_engines=True, blocks=True):
        self._outer = _resolve(outer_engines, OUTER_REGIONS, "outer_engines")
        self._blocks = _resolve(blocks, BLOCK_REGIONS, "blocks")

    @classmethod
    def from_config(cls, cf) -> "CheckpointPolicy":
        section = cf.get("activation_checkpointing", None) if cf is not None else None
        if section is None:
            return cls()
        unknown = set(section.keys()) - {"outer_engines", "blocks"}
        if unknown:
            raise ValueError(f"activation_checkpointing: unknown keys {sorted(unknown)}")
        return cls(section.get("outer_engines", True), section.get("blocks", True))

    def outer(self, region: str) -> bool:
        return self._outer[region]

    def blocks(self, region: str) -> bool:
        return self._blocks[region]


def maybe_checkpoint(enabled: bool, function, *args, **kwargs):
    """`checkpoint(function, ...)` (non-reentrant) if enabled, else the plain call."""
    if enabled:
        kwargs.setdefault("use_reentrant", False)
        return checkpoint(function, *args, **kwargs)
    kwargs.pop("use_reentrant", None)
    return function(*args, **kwargs)
