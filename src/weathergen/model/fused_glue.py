# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Config options `rope_impl` and `adaln_fused` (experiment O5): fewer elementwise kernels.

- `rope_impl` (eager | cached | fused | fused_flash): see `positional_encoding.py`.
- `adaln_fused`: every `AdaLayerNorm` becomes a `FusedAdaLayerNorm` (class swap: same
  parameters, same state dict), whose modulation `norm(x) * (1 + scale) + shift` is one
  kernel per pass (`fused_kernels.AdaLNModulate`) instead of three in the forward and five in
  the backward. Where the only consumer of the output is one autocast Linear
  (`MultiCrossAttentionHeadVarlen.lnorm_in_q`, the norm of an `MLP`), the kernel writes the
  autocast dtype directly, so the Linear's input cast and its backward mirror disappear too.
  Elsewhere (self-attention: q, k and v Linears; anything else) the output keeps the eager
  dtype, because the three bfloat16 input gradients would be summed in bfloat16 instead of
  float32. Bitwise equal to the eager modulation followed by the consumer's cast.
"""

import logging

import torch

from weathergen.model.attention import MultiCrossAttentionHeadVarlen
from weathergen.model.fused_kernels import AdaLNModulate, adaln_supported
from weathergen.model.layers import MLP
from weathergen.model.norms import AdaLayerNorm
from weathergen.model.positional_encoding import set_rope_impl

logger = logging.getLogger(__name__)


class FusedAdaLayerNorm(AdaLayerNorm):
    """`AdaLayerNorm` with the modulation as one fused op (see module doc)."""

    # write the autocast dtype instead of float32: only where one autocast Linear consumes y
    autocast_dtype_out = False

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.embed_aux:
            aux = block(aux)
        ln = self.norm(x)
        out_dtype = torch.promote_types(ln.dtype, aux.dtype)
        dev = ln.device.type
        if (
            self.autocast_dtype_out
            and out_dtype == torch.float32
            and torch.is_autocast_enabled(dev)
        ):
            out_dtype = torch.get_autocast_dtype(dev)
        if not adaln_supported(ln, aux):
            scale, shift = aux.split(aux.shape[-1] // 2, dim=-1)
            return (ln * (1 + scale) + shift).to(out_dtype)
        return AdaLNModulate.apply(ln, aux, out_dtype)


def _single_linear_consumer(model: torch.nn.Module, name: str) -> bool:
    parts = name.split(".")
    if len(parts) < 2:
        return False
    parent = model.get_submodule(".".join(parts[:-1]))
    if isinstance(parent, MultiCrossAttentionHeadVarlen) and parts[-1] == "lnorm_in_q":
        return True  # x_q -> proj_heads_q only
    if len(parts) >= 3 and parts[-1] == "0":
        owner = model.get_submodule(".".join(parts[:-2]))
        if isinstance(owner, MLP) and parts[-2] == "layers":
            return isinstance(owner.layers[1], torch.nn.Linear)
    return False


def use_fused_adaln(model: torch.nn.Module) -> dict[str, int]:
    """Turn every `AdaLayerNorm` of `model` into a `FusedAdaLayerNorm`, in place."""
    counts = {"autocast_dtype_out": 0, "eager_dtype_out": 0}
    for name, module in list(model.named_modules()):
        if type(module) is not AdaLayerNorm:
            continue
        module.__class__ = FusedAdaLayerNorm
        module.autocast_dtype_out = _single_linear_consumer(model, name)
        counts["autocast_dtype_out" if module.autocast_dtype_out else "eager_dtype_out"] += 1
    return counts


def configure_fused_glue(model: torch.nn.Module, cf) -> None:
    """Apply `rope_impl` and `adaln_fused` from the config to a built model (before sharding)."""
    rope_impl = cf.get("rope_impl", "eager")
    set_rope_impl(rope_impl)
    msg = f"rope_impl={rope_impl}"
    if cf.get("adaln_fused", False):
        msg += f", adaln_fused: {use_fused_adaln(model)}"
    logger.info(msg)
