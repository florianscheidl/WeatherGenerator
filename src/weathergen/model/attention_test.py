import pytest
import torch

from weathergen.model import attention
from weathergen.model.attention import MultiCrossAttentionHeadVarlen, MultiSelfAttentionHeadVarlen
from weathergen.model.attention_sdpa import sdpa_func, sdpa_varlen_func


def _reference_attention(q, k, v):
    """Naive attention for one sequence, layout (seq, heads, dim)."""
    scores = torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1] ** 0.5
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def _reference_varlen(q, k, v, lens_q, lens_k):
    outs = []
    for q_i, k_i, v_i in zip(
        q.split(lens_q.tolist()), k.split(lens_k.tolist()), v.split(lens_k.tolist()), strict=True
    ):
        outs.append(_reference_attention(q_i, k_i, v_i) if len(k_i) > 0 else torch.zeros_like(q_i))
    return torch.cat(outs)


def _cu_seqlens(lens):
    return torch.nn.functional.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0))


def _packed(lens, num_heads=2, dim=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(int(lens.sum()), num_heads, dim, generator=gen, requires_grad=True)


@pytest.mark.parametrize(
    "lens_q, lens_k",
    [([3, 5, 1], [3, 5, 1]), ([2, 4, 3], [5, 1, 7]), ([2, 0, 3], [4, 2, 0])],
)
def test_sdpa_varlen_matches_reference(lens_q, lens_k):
    lens_q, lens_k = torch.tensor(lens_q), torch.tensor(lens_k)
    q, k, v = _packed(lens_q, seed=0), _packed(lens_k, seed=1), _packed(lens_k, seed=2)

    out = sdpa_varlen_func(
        q, k, v, _cu_seqlens(lens_q), _cu_seqlens(lens_k), lens_q.max(), lens_k.max()
    )
    ref = _reference_varlen(q, k, v, lens_q, lens_k)
    torch.testing.assert_close(out, ref)

    grads = torch.autograd.grad(out.square().sum(), (q, k, v))
    grads_ref = torch.autograd.grad(ref.square().sum(), (q, k, v))
    for g, g_ref in zip(grads, grads_ref, strict=True):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g, g_ref)


def test_sdpa_dense_matches_reference():
    q, k, v = (torch.randn(2, 6, 2, 8) for _ in range(3))
    out = sdpa_func(q, k, v)
    ref = torch.stack([_reference_attention(*qkv) for qkv in zip(q, k, v, strict=True)])
    torch.testing.assert_close(out, ref)


def test_sdpa_rejects_softcap():
    q = torch.randn(1, 4, 2, 8)
    with pytest.raises(NotImplementedError):
        sdpa_func(q, q, q, softcap=1.0)


def test_fallback_without_flash_attn(monkeypatch):
    monkeypatch.setattr(attention, "flash_attn_varlen_func", None)
    monkeypatch.setattr(attention, "get_device_type", lambda: "cpu")
    assert attention.get_attention_fcts(True) == (sdpa_func, sdpa_varlen_func)

    monkeypatch.setattr(attention, "get_device_type", lambda: "cuda")
    with pytest.raises(ImportError):
        attention.get_attention_fcts(True)
    # explicitly disabling flash attention works on CUDA without flash_attn
    assert attention.get_attention_fcts(False) == (sdpa_func, sdpa_varlen_func)


def test_varlen_heads_run_without_flash_attn(monkeypatch):
    monkeypatch.setattr(attention, "flash_attn_varlen_func", None)
    monkeypatch.setattr(attention, "get_device_type", lambda: "cpu")

    # the heads expect lens with a leading zero (they cumsum without prepending one)
    lens = torch.tensor([0, 3, 5, 2])
    x = torch.randn(int(lens.sum()), 16)
    self_att = MultiSelfAttentionHeadVarlen(16, 2, attention_dtype=torch.float32)
    assert self_att(x, lens).shape == x.shape

    lens_kv = torch.tensor([0, 4, 1, 6])
    x_kv = torch.randn(int(lens_kv.sum()), 8)
    cross_att = MultiCrossAttentionHeadVarlen(16, 8, 2, attention_dtype=torch.float32)
    out = cross_att(x, x_kv, x_q_lens=lens, x_kv_lens=lens_kv)
    assert out.shape == x.shape
    out.sum().backward()


@pytest.mark.skipif(
    attention.flash_attn_varlen_func is None or not torch.cuda.is_available(),
    reason="needs flash_attn and a CUDA/ROCm device",
)
def test_sdpa_varlen_matches_flash_attn():
    lens_q, lens_k = torch.tensor([17, 64, 3, 40]), torch.tensor([33, 64, 1, 8])
    q, k, v = (
        torch.randn(int(lens.sum()), 4, 32, device="cuda", dtype=torch.bfloat16)
        for lens in (lens_q, lens_k, lens_k)
    )
    cu_q, cu_k = _cu_seqlens(lens_q).cuda(), _cu_seqlens(lens_k).cuda()
    args = (q, k, v, cu_q, cu_k, lens_q.max(), lens_k.max())
    out_flash = attention.flash_attn_varlen_func(*args)
    out_sdpa = sdpa_varlen_func(*args)
    torch.testing.assert_close(out_sdpa, out_flash, atol=2e-2, rtol=2e-2)
