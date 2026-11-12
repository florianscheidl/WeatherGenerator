"""O5: cached RoPE tables, fused RoPE and fused AdaLN modulation against the eager ops.

CPU tests run the PyTorch references of the fused ops (`fused_kernels.*_reference`) and must
be bitwise equal to eager. CUDA tests (skipped without a GPU and Triton) run the Triton kernels
on production shapes and must be bitwise equal to eager too.
"""

import copy

import pytest
import torch

from weathergen.model import fused_kernels as fk
from weathergen.model import positional_encoding as pe
from weathergen.model.attention import MultiCrossAttentionHeadVarlen, MultiSelfAttentionHeadVarlen
from weathergen.model.fused_glue import FusedAdaLayerNorm, use_fused_adaln
from weathergen.model.layers import MLP
from weathergen.model.norms import AdaLayerNorm

BF16 = torch.bfloat16
CUDA = torch.cuda.is_available() and fk.HAVE_TRITON


@pytest.fixture
def rope_impl():
    yield pe.set_rope_impl
    pe.set_rope_impl("eager")


def _coords(n, gen, device="cpu"):
    lat = (torch.rand(n, generator=gen) - 0.5) * torch.pi
    lon = torch.rand(n, generator=gen) * 2 * torch.pi
    return torch.stack([lat, lon], -1).to(BF16).to(device)


def _rope_eager_and(impl, q, k, coords, unsqueeze_dim, rope_impl, g_q, g_k):
    out = []
    for mode in ("eager", impl):
        rope_impl(mode)
        q_, k_ = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
        qe, ke = pe.rotary_pos_emb_2d(q_, k_, coords, unsqueeze_dim=unsqueeze_dim)
        torch.autograd.backward([qe, ke], [g_q, g_k])
        out.append((qe.detach(), ke.detach(), q_.grad, k_.grad))
    return out


@pytest.mark.parametrize("autocast", [False, True])
def test_rope_tables_cached_bitwise(autocast):
    gen = torch.Generator().manual_seed(0)
    coords = _coords(96, gen).unsqueeze(0)
    with torch.autocast("cpu", dtype=BF16, enabled=autocast):
        cos, sin = pe.rotary_embedding_2d(coords, 64)
        t1 = pe.rotary_tables_2d_cached(coords, 64, BF16)
        t2 = pe.rotary_tables_2d_cached(coords, 64, BF16)
    assert t1 is t2  # cache hit
    assert torch.equal(t1[0], cos.to(BF16)) and torch.equal(t1[1], sin.to(BF16))
    assert torch.equal(t1[2], cos[..., :32].to(BF16)) and torch.equal(t1[3], sin[..., 32:].to(BF16))
    with torch.no_grad():
        coords.add_(0.5)  # in-place update: new version, new tables
    t3 = pe.rotary_tables_2d_cached(coords, 64, BF16)
    assert t3 is not t1
    assert torch.equal(t3[0], pe.rotary_embedding_2d(coords, 64)[0].to(BF16))


@pytest.mark.parametrize("impl", ["cached", "fused"])
@pytest.mark.parametrize(
    "shape, unsqueeze_dim",
    [((2, 40, 4, 64), 2), ((40, 4, 64), 1), ((40, 4, 32), 1)],
    ids=["BTHD (global/forecast)", "THD (varlen)", "THD head dim 32"],
)
def test_rope_impl_bitwise(impl, shape, unsqueeze_dim, rope_impl):
    gen = torch.Generator().manual_seed(1)
    t = shape[1] if len(shape) == 4 else shape[0]
    coords = _coords(t, gen)
    coords = coords.unsqueeze(0) if len(shape) == 4 else coords
    q, k, g_q, g_k = (torch.randn(shape, generator=gen).to(BF16) for _ in range(4))
    (qa, ka, gqa, gka), (qb, kb, gqb, gkb) = _rope_eager_and(
        impl, q, k, coords, unsqueeze_dim, rope_impl, g_q, g_k
    )
    for a, b in ((qa, qb), (ka, kb), (gqa, gqb), (gka, gkb)):
        assert torch.equal(a, b)


def test_rope_single_rounding_differs_by_rounding_only():
    """exact=False (as flash-attn's kernel) differs from eager by at most bf16 rounding."""
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(1, 64, 8, 64, generator=gen).to(BF16)
    cos_h = torch.rand(64, 32, generator=gen).to(BF16)
    sin_h = torch.rand(64, 32, generator=gen).to(BF16)
    exact = fk.rope_reference(x, cos_h, sin_h, exact=True)
    single = fk.rope_reference(x, cos_h, sin_h, exact=False)
    # bound by the size of the two products (cancellation makes relative errors large)
    xf, c, s = x.float(), cos_h.float()[None, :, None, :], sin_h.float()[None, :, None, :]
    mag = xf[..., :32].abs() * c + xf[..., 32:].abs() * s
    mag = torch.cat([mag, xf[..., 32:].abs() * c + xf[..., :32].abs() * s], -1)
    diff = (exact.float() - single.float()).abs()
    assert (diff <= 2**-7 * mag + 1e-30).all()  # at most ~2 bf16 ulps of the larger product
    assert not torch.equal(exact, single)


@pytest.mark.parametrize(
    "ln_dtype, aux_dtype, out_dtype",
    [
        (torch.float32, BF16, BF16),  # autocast, consumer is one Linear (cast fused in)
        (torch.float32, BF16, torch.float32),  # autocast, eager output dtype
        (BF16, BF16, BF16),  # layernorm_bf16_adaln (O2) on
        (torch.float32, torch.float32, torch.float32),  # no autocast
    ],
)
def test_adaln_modulate_bitwise(ln_dtype, aux_dtype, out_dtype):
    gen = torch.Generator().manual_seed(3)
    n, d = 257, 48
    x = torch.randn(n, d, generator=gen) * 3 + 1
    aux = torch.randn(n, 2 * d, generator=gen).to(aux_dtype)
    g = torch.randn(n, d, generator=gen).to(out_dtype)

    def run(fused):
        x_ = x.clone().requires_grad_()
        aux_ = aux.clone().requires_grad_()
        ln = torch.nn.functional.layer_norm(x_, (d,)).to(ln_dtype)
        if fused:
            y = fk.AdaLNModulate.apply(ln, aux_, out_dtype)
        else:
            scale, shift = aux_.split(d, dim=-1)
            y = (ln * (1 + scale) + shift).to(out_dtype)  # .to: the consumer Linear's cast
        y.backward(g)
        return y.detach(), x_.grad, aux_.grad

    for a, b in zip(run(False), run(True), strict=True):
        assert a.dtype == b.dtype
        assert torch.equal(a, b)


@pytest.mark.parametrize("autocast_dtype_out", [False, True])
def test_fused_adaln_module_bitwise(autocast_dtype_out):
    gen = torch.Generator().manual_seed(4)
    torch.manual_seed(4)
    n, d, d_aux = 64, 32, 6
    eager = AdaLayerNorm(d, d_aux)
    fused = copy.deepcopy(eager)
    fused.__class__ = FusedAdaLayerNorm
    fused.autocast_dtype_out = autocast_dtype_out
    x = torch.randn(n, d, generator=gen)
    aux = torch.randn(n, d_aux, generator=gen)
    w = torch.randn(d, 16, generator=gen)

    def run(m):
        x_ = x.clone().requires_grad_()
        with torch.autocast("cpu", dtype=BF16):
            y = m(x_, aux)
            out = y @ w.to(BF16) if y.dtype == BF16 else y.to(BF16) @ w.to(BF16)
        out.float().pow(2).sum().backward()
        return [y.detach().to(BF16), x_.grad] + [p.grad for p in m.parameters()]

    for a, b in zip(run(eager), run(fused), strict=True):
        assert torch.equal(a, b)


def test_use_fused_adaln_sites():
    model = torch.nn.ModuleDict(
        {
            "mlp": MLP(32, 32, with_residual=True, dim_aux=6),
            "cross": MultiCrossAttentionHeadVarlen(32, 32, 4, dim_aux=6),
            "self": MultiSelfAttentionHeadVarlen(32, 4, dim_aux=6),
        }
    )
    counts = use_fused_adaln(model)
    assert counts == {"autocast_dtype_out": 2, "eager_dtype_out": 1}
    assert model["mlp"].layers[0].autocast_dtype_out
    assert model["cross"].lnorm_in_q.autocast_dtype_out
    assert not model["self"].lnorm.autocast_dtype_out
    assert all(type(m) is FusedAdaLayerNorm for m in model.modules() if isinstance(m, AdaLayerNorm))


# ------------------------------------------------------------------------------- CUDA (Triton)


@pytest.mark.skipif(not CUDA, reason="needs CUDA and Triton")
@pytest.mark.parametrize("impl", ["cached", "fused"])
@pytest.mark.parametrize("shape, unsqueeze_dim", [((1, 12288, 32, 64), 2), ((12288, 32, 64), 1)])
def test_rope_cuda_bitwise(impl, shape, unsqueeze_dim, rope_impl):
    gen = torch.Generator().manual_seed(5)
    t = shape[1] if len(shape) == 4 else shape[0]
    coords = _coords(t, gen, "cuda")
    coords = coords.unsqueeze(0) if len(shape) == 4 else coords
    q, k, g_q, g_k = (torch.randn(shape, generator=gen).to(BF16).cuda() for _ in range(4))
    with torch.autocast("cuda", dtype=BF16):
        (qa, ka, gqa, gka), (qb, kb, gqb, gkb) = _rope_eager_and(
            impl, q, k, coords, unsqueeze_dim, rope_impl, g_q, g_k
        )
    for a, b in ((qa, qb), (ka, kb), (gqa, gqb), (gka, gkb)):
        assert torch.equal(a, b)


@pytest.mark.skipif(not CUDA, reason="needs CUDA and Triton")
def test_rope_cuda_triton_matches_reference():
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(1, 12288, 32, 64, generator=gen).to(BF16).cuda()
    cos_h = torch.rand(12288, 32, generator=gen).to(BF16).cuda()
    sin_h = torch.rand(12288, 32, generator=gen).to(BF16).cuda()
    for conj in (False, True):
        for exact in (True, False):
            a = fk.rope_triton(x, cos_h, sin_h, conj, exact)
            b = fk.rope_reference(x, cos_h, sin_h, conj, exact)
            assert torch.equal(a, b) if exact else torch.allclose(a.float(), b.float(), rtol=2**-7)


@pytest.mark.skipif(not CUDA, reason="needs CUDA and Triton")
@pytest.mark.parametrize(
    "ln_dtype, out_dtype",
    [(torch.float32, BF16), (torch.float32, torch.float32), (BF16, BF16)],
)
def test_adaln_cuda_bitwise(ln_dtype, out_dtype):
    gen = torch.Generator().manual_seed(7)
    n, d = 542_080 // 8, 512  # ERA5 target points (one eighth), decoder width
    x = (torch.randn(n, d, generator=gen) * 3 + 1).cuda()
    aux = torch.randn(n, 2 * d, generator=gen).to(BF16).cuda()
    g = torch.randn(n, d, generator=gen).to(out_dtype).cuda()

    def run(fused):
        x_ = x.clone().requires_grad_()
        aux_ = aux.clone().requires_grad_()
        ln = torch.nn.functional.layer_norm(x_, (d,)).to(ln_dtype)
        if fused:
            y = fk.AdaLNModulate.apply(ln, aux_, out_dtype)
        else:
            scale, shift = aux_.split(d, dim=-1)
            y = (ln * (1 + scale) + shift).to(out_dtype)
        y.backward(g)
        return y.detach(), x_.grad, aux_.grad

    for a, b in zip(run(False), run(True), strict=True):
        assert torch.equal(a, b)
