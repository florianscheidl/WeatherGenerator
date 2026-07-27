# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    """Root-mean-square normalisation, on `torch.nn.functional.rms_norm`.

    Two properties of that op are the reason for preferring it over both a hand-rolled
    implementation and `LayerNorm`, and both concern what the backward pass has to keep:

    * `aten::rms_norm` carries no autocast registration, so it runs in the dtype it is
      given. `layer_norm` instead has a float32 cast policy, and under autocast it
      materialises a float32 copy of a bfloat16 input -- twice the bytes of the tensor
      itself -- which the graph then holds until the backward pass.
    * Where a fused kernel is available it dispatches to `aten::_fused_rms_norm`, whose
      backward takes `(grad_out, input, normalized_shape, rstd, weight)`. The graph keeps
      the bfloat16 input and a per-row `rstd`, and the float32 upcast is redone inside the
      backward rather than stored. Without a fused kernel the composite fallback runs
      instead, which does save float32 copies and gives up most of the benefit.

    `weight` is cast to the dtype of the input on every call because a mismatch between the
    two drops the call back onto the composite path -- with only a warning to say so -- and
    master weights are float32 under mixed precision. Whether a given build has the fused
    kernel at all is a property of the backend, and there the fallback is silent, so
    measure it where it matters with `tests/rms_norm_memory.py` rather than assuming.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability.
            Default is 1e-6.
            elementwise_affine (bool, optional): Whether to learn a scaling parameter. Set this
            to False to mirror the parameter-free `LayerNorm` used in the attention blocks.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter | None): Learnable scaling parameter, absent when
            `elementwise_affine` is False.

        """
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm, in the dtype of `x`.

        """
        weight = None if self.weight is None else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class AdaLayerNorm(torch.nn.Module):
    """
    AdaLayerNorm for embedding auxiliary information
    """

    def __init__(
        self,
        dim_embed_x,
        dim_aux,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        norm_type: str = "LayerNorm",
    ):
        super().__init__()

        # simple 2-layer MLP for embedding auxiliary information
        self.embed_aux = torch.nn.ModuleList()
        self.embed_aux.append(torch.nn.Linear(dim_aux, 4 * dim_aux))
        self.embed_aux.append(torch.nn.SiLU())
        self.embed_aux.append(torch.nn.Linear(4 * dim_aux, 2 * dim_embed_x))

        # follows the caller's norm_type so the conditioned blocks are not left on
        # LayerNorm, and with it the float32 promotion, when the rest moves to RMSNorm
        if norm_type == "LayerNorm":
            self.norm = torch.nn.LayerNorm(dim_embed_x, norm_eps, norm_elementwise_affine)
        else:
            self.norm = RMSNorm(dim_embed_x, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.embed_aux:
            aux = block(aux)
        scale, shift = aux.split(aux.shape[-1] // 2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift

        return x


def norm_in_input_dtype(norm: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply `norm` to `x` and return the result in the dtype of `x`.

    Under autocast, `layer_norm` carries a float32 cast policy, so its result comes back as
    float32 even for a bfloat16 input. Where that result feeds more than one projection --
    q, k and v off the same normalised tensor -- every projection would otherwise make and
    retain its own bfloat16 copy of it. Casting once here leaves a single copy for all of
    them to share, which is bit-identical to the cast autocast would have applied to each.

    Outside autocast this is a no-op: the norm already returns the dtype of its input.

    Note this deliberately does *not* stop the norm itself from running in float32. Doing
    that needs `torch.autocast(enabled=False)`, i.e. mutating thread-local autocast state
    inside the module, and every one of these prologues runs inside `torch.utils.checkpoint`
    -- which replays the function under an autocast context it *reconstructs* from
    `is_autocast_enabled(device_type)` sampled at checkpoint-call time. The two passes then
    disagree about what the norm saved and the recompute fails its metadata check.
    """

    return norm(x).to(x.dtype)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x2 * F.silu(x1)


class AdaLayerNormLayer(torch.nn.Module):
    """
    AdaLayerNorm for embedding auxiliary information as done in DiT (Peebles & Xie) with zero
    initialisation https://arxiv.org/pdf/2212.09748

    This module thus wraps a layer (e.g. self-attention or feedforward nn) and applies LayerNorm
    followed by scale and shift before the layer and a final scaling after the layer as well as the
    final residual layer.

    layer is a function that takes 2 arguments the first the latent and the second is the
    conditioning signal
    """

    def __init__(
        self,
        dim,
        dim_aux,
        layer,
        norm_eps: float = 1e-6,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_aux, 3 * dim, bias=True))

        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.layer = layer

        # Initialize weights to zero for modulation and gating layers
        self.initialise_weights()

    def initialise_weights(self):
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, x_lens, **kwargs) -> torch.Tensor:
        # the -1 in torch.repeat_interleave(..) is because x_lens is designed for use with flash
        # attention and thus has a spurious 0 at the beginning to satisfy the flash attention api
        shift, scale, gate = self.adaLN_modulation(c)[torch.repeat_interleave(x_lens) - 1].chunk(
            3, dim=1
        )
        kwargs["x_lens"] = x_lens
        return (
            gate
            * self.layer(
                modulate(
                    self.ln(x),
                    shift,
                    scale,
                ),
                **kwargs,
            )
            + x
        )


class SaturateEncodings(nn.Module):
    """A common alternative to a KL regularisation prevent outliers in the latent space when
    learning an auto-encoder for latent generative model, an example value for the scale factor is 5
    """

    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor_squared = scale_factor**2

    def forward(self, x):
        return x / torch.sqrt(1 + (x**2 / self.scale_factor_squared))
