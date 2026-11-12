"""Tests for `qk_norm: rmsnorm` (rmsnorm_triton.py).

CPU tests: the reference path against `F.rms_norm`, the module, the LayerNorm equivalence after
centring the projections. CUDA tests (skipped without a GPU and Triton): the Triton kernels.
"""

import pytest
import torch
import torch.nn.functional as F

from weathergen.model import rmsnorm_triton as rt
from weathergen.model.fused_kernels import HAVE_TRITON

BF16 = torch.bfloat16
CUDA = torch.cuda.is_available() and HAVE_TRITON
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a GPU and Triton")
SITE_DS = [64, 128, 256, 512, 2048]


def _ref(x, dy, eps):
    x = x.detach().float().requires_grad_()
    y = F.rms_norm(x, (x.shape[-1],), None, eps)
    y.backward(dy.float())
    return y.detach(), x.grad


def _check(device, d, rows, dtype, atol, rtol, eps=1e-4):
    g = torch.Generator().manual_seed(d)
    x = (torch.randn(rows, d, generator=g) * 2 + 0.5).to(dtype).to(device)
    dy = torch.randn(rows, d, generator=g).to(dtype).to(device)
    xr = x.clone().requires_grad_()
    y = rt.fast_rms_norm(xr, eps, dtype)
    y.backward(dy)
    y_ref, dx_ref = _ref(x, dy, eps)
    assert y.dtype == dtype and xr.grad.dtype == dtype
    torch.testing.assert_close(y.float(), y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(xr.grad.float(), dx_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("d", SITE_DS + [96])
@pytest.mark.parametrize("rows", [1, 37, 1000])
def test_reference_fp32(d, rows):
    _check("cpu", d, rows, torch.float32, 1e-5, 1e-5)


@pytest.mark.parametrize("d", [64, 128])
def test_reference_bf16(d):
    _check("cpu", d, 100, BF16, 5e-2, 2e-2)


def test_module_matches_f_rms_norm_cpu():
    m = rt.QKRMSNorm(64, 1e-4)
    x = torch.randn(10, 4, 64)
    torch.testing.assert_close(m(x), F.rms_norm(x, (64,), None, 1e-4))
    assert not list(m.parameters()) and not m.state_dict()


def test_module_autocast_dtype_independent_of_input_cpu():
    m = rt.QKRMSNorm(64, 1e-4)
    x = torch.randn(10, 4, 64)
    with torch.autocast("cpu", dtype=BF16):
        y32 = m(x)
        y16 = m(x.to(BF16))
    assert y32.dtype == BF16 and y16.dtype == BF16
    torch.testing.assert_close(y32, y16)


def test_unit_rms_and_bound():
    x = torch.randn(50, 8, 64) * 7 + 3
    y = rt.QKRMSNorm(64, 1e-12)(x)
    torch.testing.assert_close(y.pow(2).sum(-1), torch.full((50, 8), 64.0), atol=1e-3, rtol=1e-4)
    # same norm as LayerNorm without affine, so |q.k| / sqrt(D) <= sqrt(D) for both
    yl = F.layer_norm(x, (64,), None, None, 1e-12)
    torch.testing.assert_close(yl.pow(2).sum(-1), y.pow(2).sum(-1), atol=1e-3, rtol=1e-4)


def test_layernorm_equals_rmsnorm_after_centring():
    class Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dim_head_proj = 16
            self.proj_heads_q = torch.nn.Linear(32, 4 * 16, bias=False)
            self.proj_heads_k = torch.nn.Linear(32, 4 * 16, bias=False)
            self.lnorm_q = torch.nn.LayerNorm(16, elementwise_affine=False, eps=1e-5)
            self.lnorm_k = torch.nn.LayerNorm(16, elementwise_affine=False, eps=1e-5)

    ln = Blk()
    rms = Blk()
    rms.load_state_dict(ln.state_dict())
    assert rt.use_qk_rmsnorm(rms) == 2
    assert rt.centre_qk_projections(rms) == 2
    x = torch.randn(7, 32)
    for proj, norm in (("proj_heads_q", "lnorm_q"), ("proj_heads_k", "lnorm_k")):
        a = getattr(ln, norm)(getattr(ln, proj)(x).reshape(7, 4, 16))
        b = getattr(rms, norm)(getattr(rms, proj)(x).reshape(7, 4, 16))
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_use_qk_rmsnorm_skips_affine_and_identity():
    class Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lnorm_q = torch.nn.LayerNorm(16)
            self.lnorm_k = torch.nn.Identity()

    b = Blk()
    assert rt.use_qk_rmsnorm(b) == 0
    assert isinstance(b.lnorm_q, torch.nn.LayerNorm)


def test_default_config_is_layernorm():
    from pathlib import Path

    import yaml

    root = Path(rt.__file__).resolve().parents[3]
    cf = yaml.safe_load((root / "config" / "default_config.yml").read_text())
    assert cf["qk_norm"] == "layernorm" and cf["qk_norm_impl"] == "torch"


@needs_cuda
@pytest.mark.parametrize("d", SITE_DS + [96, 4096])
@pytest.mark.parametrize("rows", [1, 1000, 4099, 12293, 393216])
@pytest.mark.parametrize("dtype", [torch.float32, BF16])
def test_triton_matches_torch(d, rows, dtype):
    tol = (1e-4, 1e-4) if dtype == torch.float32 else (6e-2, 2e-2)
    _check("cuda", d, rows, dtype, *tol)


@needs_cuda
def test_triton_strided_view():
    big = torch.randn(5000, 3, 4, 64, device="cuda", dtype=BF16)
    q = big[:, 0]  # [N, H, dh], rows not flattenable with one stride
    y = rt.fast_rms_norm(q, 1e-4, BF16)
    torch.testing.assert_close(
        y.float(), F.rms_norm(q.float(), (64,), None, 1e-4), atol=5e-2, rtol=2e-2
    )


@needs_cuda
@pytest.mark.parametrize("impl", ["torch", "triton"])
def test_module_cuda_autocast_and_checkpoint(impl):
    m = rt.QKRMSNorm(64, 1e-4, impl)
    x = torch.randn(4096, 32, 64, device="cuda", dtype=BF16, requires_grad=True)
    with torch.autocast("cuda", dtype=BF16):
        y32 = m(x.float())
        y16 = m(x)
    assert y32.dtype == BF16 and y16.dtype == BF16
    torch.testing.assert_close(y32, y16)
    ref = F.rms_norm(x.float(), (64,), None, 1e-4)
    torch.testing.assert_close(y16.float(), ref, atol=5e-2, rtol=2e-2)

    def f(t):
        with torch.autocast("cuda", dtype=BF16):
            return m(t).float().square().sum()

    g1 = torch.autograd.grad(f(x), x)[0]
    g2 = torch.autograd.grad(torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False), x)[0]
    torch.testing.assert_close(g1, g2)
