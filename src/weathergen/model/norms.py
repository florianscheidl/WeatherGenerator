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


# from https://github.com/meta-llama/llama/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability.
            Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        """Re-initialize the learnable scale to ones (matches ``__init__``)."""
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AutocastDtypeLayerNorm(torch.nn.LayerNorm):
    """`torch.nn.LayerNorm` that runs in the autocast dtype instead of float32 under autocast.

    `layer_norm` is on autocast's float32 list (CUDA, MPS): a bfloat16 input is cast up, the
    norm returns float32, and every consumer casts it back down (each Linear, the q/k
    `.to(attention_dtype)`), with the mirror casts in the backward. Here the norm runs on the
    input cast to the autocast dtype, with the affine parameters cast to that dtype, inside an
    autocast-disabled region, so it returns the autocast dtype. The bfloat16 kernels compute
    the statistics in float32 and round the output once, as the consumer's cast did before.

    The output dtype does not depend on the input dtype: during the recompute of a functional
    checkpoint aligned with an FSDP2 unit, FSDP2 skips the cast of the unit inputs to
    `param_dtype`, so the same norm can see bfloat16 in the forward and float32 in the
    recompute; casting to the autocast dtype here keeps the saved tensors identical in both.

    Without autocast on the input's device this is `torch.nn.LayerNorm`. Swapped in for
    selected `torch.nn.LayerNorm` instances by `use_autocast_dtype_layernorms` (`layernorm_bf16`
    in the config); the parameters and the state dict are those of `torch.nn.LayerNorm`.

    `impl = "triton"` (set by `layernorm_impl: triton`, see `layernorm_triton.py`) runs the
    multi-row Triton kernel instead of `F.layer_norm` on CUDA; same dtypes, same autocast
    handling.
    """

    impl = "torch"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = x.device.type
        if not torch.is_autocast_enabled(device_type):
            return super().forward(x)
        dtype = torch.get_autocast_dtype(device_type)
        with torch.autocast(device_type=device_type, enabled=False):
            weight = None if self.weight is None else self.weight.to(dtype)
            bias = None if self.bias is None else self.bias.to(dtype)
            if self.impl == "triton" and x.is_cuda and len(self.normalized_shape) == 1:
                from weathergen.model.layernorm_triton import fast_layer_norm

                return fast_layer_norm(x, weight, bias, self.eps, dtype)
            return F.layer_norm(x.to(dtype), self.normalized_shape, weight, bias, self.eps)


class AdaLNZero(torch.nn.Module):
    """
    AdaLayerNorm with zero initialization and with additional gate parameter
    """

    def __init__(
        self, dim_embed_x, dim_aux, norm_elementwise_affine: bool = False, norm_eps: float = 1e-5
    ):
        super().__init__()

        # simple 2-layer MLP for embedding auxiliary information
        self.embed_aux = torch.nn.ModuleList()
        self.embed_aux.append(torch.nn.Linear(dim_aux, 6 * dim_aux))
        self.embed_aux.append(torch.nn.SiLU())
        self.embed_aux.append(torch.nn.Linear(6 * dim_aux, 3 * dim_embed_x))

        self.norm = torch.nn.LayerNorm(
            dim_embed_x,
            eps=norm_eps,
            elementwise_affine=norm_elementwise_affine,
        )
        # Zero-initialize the final modulation layer.
        nn.init.zeros_(self.embed_aux[-1].weight)
        nn.init.zeros_(self.embed_aux[-1].bias)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.embed_aux:
            aux = block(aux)
        scale, shift, gate = aux.chunk(3, dim=-1)

        x = self.norm(x) * (1 + scale) + shift

        return x, gate


class AdaLayerNorm(torch.nn.Module):
    """
    AdaLayerNorm for embedding auxiliary information
    """

    def __init__(
        self, dim_embed_x, dim_aux, norm_elementwise_affine: bool = False, norm_eps: float = 1e-5
    ):
        super().__init__()

        # simple 2-layer MLP for embedding auxiliary information
        self.embed_aux = torch.nn.ModuleList()
        self.embed_aux.append(torch.nn.Linear(dim_aux, 4 * dim_aux))
        self.embed_aux.append(torch.nn.SiLU())
        self.embed_aux.append(torch.nn.Linear(4 * dim_aux, 2 * dim_embed_x))

        self.norm = torch.nn.LayerNorm(dim_embed_x, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.embed_aux:
            aux = block(aux)
        scale, shift = aux.split(aux.shape[-1] // 2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift

        return x


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
        # expand the per-sequence conditioning to per-token; equivalent to
        # torch.repeat_interleave(x_lens) - 1, but with a static output shape taken from
        # x (repeat_interleave's data-dependent output size would force a device sync);
        # the -1 is because x_lens has a spurious 0 at the beginning to satisfy the
        # flash attention api
        token_idxs = torch.arange(x.shape[0], device=x.device)
        seg_idxs = torch.searchsorted(x_lens.cumsum(0).to(token_idxs.dtype), token_idxs, right=True)
        shift, scale, gate = self.adaLN_modulation(c)[seg_idxs - 1].chunk(3, dim=1)
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
