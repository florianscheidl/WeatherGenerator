"""Tests for the multi-row LayerNorm (`layernorm_impl: triton`).

CPU tests run `FastLayerNormFn` on its PyTorch reference path (same formulas as the kernels)
against `F.layer_norm`. CUDA tests (skipped without a GPU and Triton) run the Triton kernels on
every normalised size of the production sites against `F.layer_norm`, forward and backward.
"""

import pytest
import torch
import torch.nn.functional as F

from weathergen.model import layernorm_triton as lt
from weathergen.model.fused_kernels import HAVE_TRITON
from weathergen.model.norms import AutocastDtypeLayerNorm

BF16 = torch.bfloat16
CUDA = torch.cuda.is_available() and HAVE_TRITON
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a GPU and Triton")
SITE_DS = [64, 128, 256, 512, 2048]


def _run(fn, x, w, b, dy):
    x = x.detach().clone().requires_grad_()
    w = None if w is None else w.detach().clone().requires_grad_()
    b = None if b is None else b.detach().clone().requires_grad_()
    y = fn(x, w, b)
    y.backward(dy)
    return y.detach(), x.grad, None if w is None else w.grad, None if b is None else b.grad


def _check(device, d, rows, dtype, affine, atol, rtol, out_dtype=None):
    g = torch.Generator().manual_seed(d)
    x = (torch.randn(rows, d, generator=g) * 2 + 0.5).to(dtype).to(device)
    w = torch.randn(d, generator=g).to(dtype).to(device) if affine else None
    b = torch.randn(d, generator=g).to(dtype).to(device) if affine else None
    od = out_dtype or dtype
    dy = torch.randn(rows, d, generator=g).to(od).to(device)
    ref_x = x.float()
    ref = _run(
        lambda x_, w_, b_: F.layer_norm(
            x_, (d,), None if w_ is None else w_, None if b_ is None else b_, 1e-5
        ),
        ref_x,
        None if w is None else w.float(),
        None if b is None else b.float(),
        dy.float(),
    )
    got = _run(lambda x_, w_, b_: lt.fast_layer_norm(x_, w_, b_, 1e-5, od), x, w, b, dy)
    names = ["y", "dx", "dw", "db"]
    for n, r, o in zip(names, ref, got, strict=True):
        if r is None:
            assert o is None
            continue
        a, t = atol, rtol
        if n == "dx" and dtype == BF16:  # dx has the input dtype: one bf16 rounding (2^-8)
            a, t = max(a, 0.02), max(t, 0.01)
        torch.testing.assert_close(o.float(), r, atol=a, rtol=t, msg=lambda m, n=n: f"{n}: {m}")


@pytest.mark.parametrize("d", SITE_DS + [96])
@pytest.mark.parametrize("affine", [False, True])
def test_reference_fp32_matches_torch(d, affine):
    _check("cpu", d, 37, torch.float32, affine, 1e-4, 1e-4)


@pytest.mark.parametrize("d", [64, 2048])
@pytest.mark.parametrize("affine", [False, True])
def test_reference_bf16_within_rounding(d, affine):
    # inputs are bf16 values, so the reference only differs from fp32 torch by output rounding
    _check("cpu", d, 37, BF16, affine, 0.15, 0.03)


def test_bf16_in_fp32_out():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 64, generator=g).to(BF16)
    y = lt.fast_layer_norm(x, None, None, 1e-5, torch.float32)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, F.layer_norm(x.float(), (64,), eps=1e-5), atol=1e-5, rtol=1e-5)


def test_residual_fused():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(10, 128, generator=g).to(BF16).requires_grad_()
    r = torch.randn(10, 128, generator=g).to(BF16).requires_grad_()
    w = torch.randn(128, generator=g).to(BF16).requires_grad_()
    y, s = lt.fast_layer_norm(x, w, None, 1e-5, BF16, residual=r)
    s_ref = x + r
    assert torch.equal(s, s_ref)
    y_ref = F.layer_norm(s_ref.float(), (128,), w.float(), None, 1e-5)
    torch.testing.assert_close(y.float(), y_ref, atol=0.1, rtol=0.03)
    gy, gs = torch.randn_like(y), torch.randn_like(s)
    (y.float() * gy.float()).sum().backward(retain_graph=True)
    (s.float() * gs.float()).sum().backward()
    # reference: same graph with eager ops
    x2, r2, w2 = (t.detach().clone().requires_grad_() for t in (x, r, w))
    s2 = x2 + r2
    y2 = F.layer_norm(s2.float(), (128,), w2.float(), None, 1e-5)
    ((y2 * gy.float()).sum() + (s2.float() * gs.float()).sum()).backward()
    torch.testing.assert_close(x.grad.float(), x2.grad.float(), atol=0.1, rtol=0.05)
    torch.testing.assert_close(x.grad, r.grad)  # both inputs of the add get the same gradient


def test_only_residual_sum_used():
    x = torch.randn(4, 64, requires_grad=True)
    r = torch.randn(4, 64, requires_grad=True)
    _, s = lt.fast_layer_norm(x, None, None, 1e-5, torch.float32, residual=r)
    s.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(4, 64))


def test_strided_rows_view():
    base = torch.randn(16, 3 * 64)
    xv = base[:, 64:128]  # row stride 192, last dim contiguous
    x2, stride = lt._rows_2d(xv)
    assert stride == 192 and x2.data_ptr() == xv.data_ptr()
    y = lt.fast_layer_norm(xv, None, None, 1e-5)
    torch.testing.assert_close(y, F.layer_norm(xv, (64,), eps=1e-5), atol=1e-5, rtol=1e-5)
    xt = torch.randn(4, 8, 64).transpose(0, 1)  # not flattenable: copy
    y = lt.fast_layer_norm(xt, None, None, 1e-5)
    torch.testing.assert_close(y, F.layer_norm(xt, (64,), eps=1e-5), atol=1e-5, rtol=1e-5)


def test_modules_and_switch():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = AutocastDtypeLayerNorm(64)
            self.b = torch.nn.LayerNorm(64)
            self.c = torch.nn.Linear(64, 64)

    m = M()
    counts = lt.use_triton_layernorms(m)
    assert counts == {"autocast_dtype": 1, "float32": 1}
    assert m.a.impl == "triton" and AutocastDtypeLayerNorm.impl == "torch"
    assert isinstance(m.b, lt.TritonLayerNorm) and isinstance(m.b, torch.nn.LayerNorm)
    x = torch.randn(5, 64)
    # CPU: both fall back to torch.nn.LayerNorm.forward semantics
    torch.testing.assert_close(m.a(x), F.layer_norm(x, (64,), m.a.weight, m.a.bias, m.a.eps))
    torch.testing.assert_close(m.b(x), F.layer_norm(x, (64,), m.b.weight, m.b.bias, m.b.eps))
    assert set(m.state_dict()) == {"a.weight", "a.bias", "b.weight", "b.bias", "c.weight", "c.bias"}


@needs_cuda
@pytest.mark.parametrize("d", SITE_DS + [96, 4096])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("rows", [1, 1000, 12288 + 5])
def test_cuda_fp32(d, affine, rows):
    _check("cuda", d, rows, torch.float32, affine, 2e-4, 2e-4)


@needs_cuda
@pytest.mark.parametrize("d", SITE_DS + [96])
@pytest.mark.parametrize("affine", [False, True])
def test_cuda_bf16(d, affine):
    # reference is fp32 torch on the bf16 inputs; kernel output is rounded to bf16
    _check("cuda", d, 4099, BF16, affine, 0.08, 0.02)


@needs_cuda
@pytest.mark.parametrize("d", [64, 512])
def test_cuda_bf16_in_fp32_out(d):
    _check("cuda", d, 999, BF16, False, 2e-4, 2e-4, out_dtype=torch.float32)


@needs_cuda
def test_cuda_large_and_strided():
    d, rows = 64, 393216
    x = torch.randn(rows, d, device="cuda").to(BF16)
    y = lt.fast_layer_norm(x, None, None, 1e-4)
    torch.testing.assert_close(
        y.float(), F.layer_norm(x.float(), (d,), eps=1e-4), atol=0.03, rtol=0.01
    )
    base = torch.randn(1000, 3 * 64, device="cuda").to(BF16)
    xv = base[:, 64:128]
    y = lt.fast_layer_norm(xv, None, None, 1e-4)
    torch.testing.assert_close(
        y.float(), F.layer_norm(xv.float(), (64,), eps=1e-4), atol=0.03, rtol=0.01
    )


@needs_cuda
def test_cuda_residual():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(777, 256, generator=g).to(BF16).cuda().requires_grad_()
    r = torch.randn(777, 256, generator=g).to(BF16).cuda().requires_grad_()
    y, s = lt.fast_layer_norm(x, None, None, 1e-5, BF16, residual=r)
    assert torch.equal(s, x + r)
    torch.testing.assert_close(
        y.float(), F.layer_norm(s.float(), (256,), eps=1e-5), atol=0.04, rtol=0.01
    )
    (y.float().sum() * 0.5 + s.float().sum()).backward()
    assert torch.equal(x.grad, r.grad)


@needs_cuda
def test_cuda_autocast_module_matches_torch():
    d = 128
    ref = AutocastDtypeLayerNorm(d, elementwise_affine=False, eps=1e-4).cuda()
    fast = AutocastDtypeLayerNorm(d, elementwise_affine=False, eps=1e-4).cuda()
    fast.impl = "triton"
    x = torch.randn(5000, 4, d, device="cuda").to(BF16)
    outs = []
    for m in (ref, fast):
        xi = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=BF16):
            y = m(xi)
        assert y.dtype == BF16
        y.float().pow(2).sum().backward()
        outs.append((y.detach().float(), xi.grad.float()))
    torch.testing.assert_close(outs[1][0], outs[0][0], atol=0.03, rtol=0.01)
    torch.testing.assert_close(outs[1][1], outs[0][1], atol=0.5, rtol=0.05)
