# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""All performance options on together: `embed_readout_batched`, `layernorm_bf16`,
`rope_impl=fused`, `adaln_fused` (with `layernorm_bf16_adaln` off).

CPU, bfloat16 autocast with the CUDA float32 policy for `layer_norm` (fixture of
`layernorm_bf16_test`), non-reentrant checkpoint around every unit, as the model runs on the GPU.
Checks that forward, backward and recompute run without dtype errors or `CheckpointError`, that
checkpointed equals plain, that the AdaLN norms stay float32 and the fused modulation sees
(float32 ln, bfloat16 aux), and that the result is close to all options off.
"""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from weathergen.model import attention as attention_module
from weathergen.model import fused_kernels as fk
from weathergen.model import positional_encoding as pe
from weathergen.model.attention import MultiSelfAttentionHead
from weathergen.model.embeddings import StreamEmbedTransformer
from weathergen.model.fused_glue import FusedAdaLayerNorm, configure_fused_glue
from weathergen.model.layernorm_bf16 import use_autocast_dtype_layernorms
from weathergen.model.layernorm_bf16_test import _fp32_layer_norm
from weathergen.model.layers import MLP
from weathergen.model.norms import AdaLayerNorm, AutocastDtypeLayerNorm

BF16 = torch.bfloat16
D = 32
AUX = 6
N = 48  # patches
C = 6  # channels
TOKEN = 8


@pytest.fixture
def cuda_like_autocast_cpu():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("layer_norm", _fp32_layer_norm, "AutocastCPU")
    yield
    lib._destroy()
    pe.set_rope_impl("eager")


@pytest.fixture
def sdpa_for_flash(monkeypatch):
    def fn(q, k, v, softcap=0.0, dropout_p=0.0):
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    monkeypatch.setattr(attention_module, "flash_attn_func", fn)


class Block(torch.nn.Module):
    """StreamEmbedTransformer readout plus an AdaLN attention block and an AdaLN MLP."""

    def __init__(self):
        super().__init__()
        self.embed = StreamEmbedTransformer(
            mode="channels",
            num_tokens=1,
            token_size=TOKEN,
            num_channels=C,
            dim_embed=D,
            dim_out=C * 5,
            num_blocks=1,
            num_heads=2,
            unembed_mode="block",
            readout_batched=True,
            dropout_final_rate=0.0,
        )
        self.attn = MultiSelfAttentionHead(
            D, 4, dim_aux=AUX, with_2d_rope=True, with_flash=True, attention_dtype=BF16
        )
        self.mlp = MLP(D, D, with_residual=True, dim_aux=AUX)
        self.head = torch.nn.Linear(C * 5, D)

    def forward(self, x_in, x, coords, aux, wrap):
        e = self.embed(x_in)  # (N, 1, C * 5)
        h = self.head(e.squeeze(1)).unsqueeze(0) + x
        h = wrap(lambda t: self.attn(t, coords=coords, ada_ln_aux=aux), h)
        h = wrap(lambda t: self.mlp(t, ada_ln_aux=aux), h)
        return h


def _plain(fn, t):
    return fn(t)


def _ckpt(fn, t):
    return checkpoint(fn, t, use_reentrant=False)


def _inputs():
    g = torch.Generator().manual_seed(1)
    x_in = torch.randn(N, TOKEN, C, generator=g)
    x = torch.randn(1, N, D, generator=g)
    lat = (torch.rand(1, N, generator=g) - 0.5) * torch.pi
    lon = torch.rand(1, N, generator=g) * 2 * torch.pi
    coords = torch.stack([lat, lon], -1).to(BF16)
    aux = torch.randn(1, N, AUX, generator=g)
    grad = torch.randn(1, N, D, generator=g)
    return x_in, x, coords, aux, grad


def _make(options_on):
    torch.manual_seed(0)
    m = Block()
    with torch.no_grad():
        for ln in m.embed.ln_final:
            ln.weight.normal_(1.0, 0.2)
            ln.bias.normal_(0.0, 0.2)
        # non-zero modulation so that the AdaLN output depends on aux
        for mod in (m.attn.lnorm, m.mlp.layers[0]):
            mod.embed_aux[-1].weight.normal_(0, 0.2)
    if options_on:
        use_autocast_dtype_layernorms(m, with_adaln=False)
        configure_fused_glue(m, {"rope_impl": "fused", "adaln_fused": True})
    return m


def _run(m, wrap, x_dtype=torch.float32):
    x_in, x, coords, aux, grad = _inputs()
    x = x.to(x_dtype).requires_grad_()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=BF16):
        out = m(x_in, x, coords, aux, wrap)
    out.backward(grad.to(out.dtype))
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    grads["x"] = x.grad.clone()
    return out.detach(), grads


def test_selection_and_dtypes_at_fused_adaln(cuda_like_autocast_cpu, sdpa_for_flash, monkeypatch):
    m = _make(options_on=True)
    # AdaLN norms stay float32 LayerNorms, wrapped by the fused modulation
    for ada in (m.attn.lnorm, m.mlp.layers[0]):
        assert type(ada) is FusedAdaLayerNorm and isinstance(ada, AdaLayerNorm)
        assert type(ada.norm) is torch.nn.LayerNorm
    # the final readout norms are converted (but the batched readout does not call them)
    assert all(type(ln) is AutocastDtypeLayerNorm for ln in m.embed.ln_final)

    seen = []
    orig = fk.AdaLNModulate.apply

    def spy(ln, aux, out_dtype):
        out = orig(ln, aux, out_dtype)
        seen.append((ln.dtype, aux.dtype, out_dtype, out.dtype))
        return out

    monkeypatch.setattr(fk.AdaLNModulate, "apply", spy)
    x_seen = []
    for ada in (m.attn.lnorm, m.mlp.layers[0]):
        ada.norm.register_forward_pre_hook(lambda _, args: x_seen.append(args[0].dtype))
    _run(m, _plain)
    assert x_seen == [torch.float32, torch.float32]
    # attention (eager dtype out) and MLP (single Linear consumer: autocast dtype out)
    assert seen[0] == (torch.float32, BF16, torch.float32, torch.float32)
    assert seen[1] == (torch.float32, BF16, BF16, BF16)


@pytest.mark.parametrize("x_dtype", [torch.float32, BF16])
def test_all_options_forward_backward_recompute(cuda_like_autocast_cpu, sdpa_for_flash, x_dtype):
    m = _make(options_on=True)
    out_p, g_p = _run(m, _plain, x_dtype)
    out_c, g_c = _run(m, _ckpt, x_dtype)  # raises CheckpointError on a recompute mismatch
    assert out_c.dtype == out_p.dtype
    assert torch.equal(out_c, out_p)
    for k, v in g_p.items():
        assert v.dtype == g_c[k].dtype
        assert torch.equal(v, g_c[k]), k
    assert all(torch.isfinite(v).all() for v in g_p.values())


def test_all_options_close_to_off(cuda_like_autocast_cpu, sdpa_for_flash):
    off = _make(options_on=False)
    out_off, g_off = _run(off, _ckpt)
    on = _make(options_on=True)
    on.load_state_dict(off.state_dict())
    out_on, g_on = _run(on, _ckpt)
    scale = out_off.abs().max()
    assert (out_on - out_off).abs().max() / scale < 2e-2
    for k, v in g_off.items():
        s = v.abs().max().clamp_min(1e-6)
        assert (g_on[k] - v).abs().max() / s < 5e-2, k
