# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Config option `layernorm_bf16`: LayerNorms that run in the autocast dtype.

`use_autocast_dtype_layernorms` turns selected `torch.nn.LayerNorm` instances of a built model
into `AutocastDtypeLayerNorm` (a class swap: same parameters, same state dict). Selected are
the norms whose output feeds only a Linear or flash attention, where the autocast-dtype output
is what the consumer computed with before:

- attention modules: `lnorm`, `lnorm_in_q` (both not for DiT blocks, whose output feeds the
  noise conditioning), `lnorm_in_kv`, `lnorm_q`, `lnorm_k` (the q/k norms feed
  `.to(attention_dtype)`, which becomes a no-op);
- `MLP`: the pre-norm `layers[0]`;
- `StreamEmbedTransformer` in `channels` mode: `ln_final` (feeds the unembedding Linear).

Left in float32 under autocast:

- AdaLN norms (`AdaLayerNorm.norm`, `AdaLNZero.norm`, `AdaLayerNormLayer.ln`): their output
  feeds the modulation `x * (1 + scale) + shift`, which runs in float32 today; with
  `layernorm_bf16_adaln` they are converted too and the modulation runs in bfloat16;
- the encoder's final `ln`: its output feeds float32 consumers (target prediction engines
  with float32 parameters, latent losses);
- everything else, e.g. `KwargLayerNorm` (writes the residual stream), `StreamEmbedTransformer`
  `ln_final` in `columns` mode (feeds a float16 cast), the target-engine input norms.
"""

import torch

from weathergen.model.attention import (
    MultiCrossAttentionHead,
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
    MultiSelfAttentionHeadVarlenFlex,
)
from weathergen.model.embeddings import StreamEmbedTransformer
from weathergen.model.layers import MLP
from weathergen.model.norms import (
    AdaLayerNorm,
    AdaLayerNormLayer,
    AdaLNZero,
    AutocastDtypeLayerNorm,
)

_ATTENTION_TYPES = (
    MultiCrossAttentionHead,
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
    MultiSelfAttentionHeadVarlenFlex,
)


def _site(model: torch.nn.Module, name: str, with_adaln: bool) -> str | None:
    """Site class of the LayerNorm `name` if it is converted, else None."""
    parts = name.split(".")
    parent = model.get_submodule(".".join(parts[:-1]))
    attr = parts[-1]

    if isinstance(parent, _ATTENTION_TYPES):
        if attr in ("lnorm_in_kv", "lnorm_q", "lnorm_k"):
            return "attention"
        if attr in ("lnorm", "lnorm_in_q") and not getattr(parent, "is_dit", False):
            return "attention"
        return None

    if isinstance(parent, StreamEmbedTransformer):
        return "embed" if attr == "ln_final" and parent.mode == "channels" else None

    if with_adaln and (
        (isinstance(parent, AdaLayerNorm | AdaLNZero) and attr == "norm")
        or (isinstance(parent, AdaLayerNormLayer) and attr == "ln")
    ):
        return "adaln"

    if isinstance(parent, torch.nn.ModuleList) and len(parts) >= 2:
        owner = model.get_submodule(".".join(parts[:-2]))
        list_attr = parts[-2]
        if isinstance(owner, MLP) and list_attr == "layers" and attr == "0":
            return "mlp"
        if (
            isinstance(owner, StreamEmbedTransformer)
            and list_attr == "ln_final"
            and owner.mode == "channels"
        ):
            return "embed"

    return None


def use_autocast_dtype_layernorms(
    model: torch.nn.Module, with_adaln: bool = False
) -> tuple[dict[str, int], list[str]]:
    """Turn the selected LayerNorms of `model` into `AutocastDtypeLayerNorm`, in place.

    Call on the built model, before `fully_shard`. Returns the number of converted norms per
    site class and the names of the `torch.nn.LayerNorm` instances left as they are.
    """
    counts: dict[str, int] = {}
    left: list[str] = []
    for name, module in list(model.named_modules()):
        if type(module) is not torch.nn.LayerNorm:
            continue
        site = _site(model, name, with_adaln)
        if site is None:
            left.append(name)
            continue
        module.__class__ = AutocastDtypeLayerNorm
        counts[site] = counts.get(site, 0) + 1
    return counts, left
