"""Tests for `AutocastDtypeLayerNorm` (config option `layernorm_bf16`).

CPU autocast keeps `layer_norm` in the input dtype, unlike CUDA (and MPS), where it is on the
float32 list. `cuda_like_autocast_cpu` registers the CUDA policy for `aten::layer_norm` under
CPU autocast, so the CPU tests see the casts the model sees on the GPU.
"""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointError, checkpoint

from weathergen.model.norms import AutocastDtypeLayerNorm

BF16 = torch.bfloat16
D = 64
HEADS = 4


def _fp32_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
    keys = torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCPU)
    with torch._C._ExcludeDispatchKeyGuard(keys):

        def up(t):
            return t.float() if t is not None and t.is_floating_point() else t

        return torch.ops.aten.layer_norm(
            up(x), normalized_shape, up(weight), up(bias), eps, cudnn_enable
        )


@pytest.fixture
def cuda_like_autocast_cpu():
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("layer_norm", _fp32_layer_norm, "AutocastCPU")
    yield
    lib._destroy()


class _CastCounter(TorchDispatchMode):
    """Counts dtype casts of tensors with at least `min_numel` elements."""

    def __init__(self, min_numel=1024):
        super().__init__()
        self.min_numel = min_numel
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype):
            src = args[0]
            if src.numel() >= self.min_numel and out.dtype != src.dtype:
                self.n += 1
        return out


class _Prologue(torch.nn.Module):
    """Attention prologue as in `MultiSelfAttentionHeadVarlen.forward`: norm, q/k/v, q/k norm."""

    def __init__(self, norm_cls):
        super().__init__()
        self.lnorm = norm_cls(D, elementwise_affine=False, eps=1e-4)
        self.proj_q = torch.nn.Linear(D, D, bias=False)
        self.proj_k = torch.nn.Linear(D, D, bias=False)
        self.proj_v = torch.nn.Linear(D, D, bias=False)
        self.lnorm_q = norm_cls(D // HEADS, elementwise_affine=False, eps=1e-4)
        self.lnorm_k = norm_cls(D // HEADS, elementwise_affine=False, eps=1e-4)
        self.mlp_norm = norm_cls(D, eps=1e-5)
        self.mlp = torch.nn.Linear(D, D)

    def forward(self, x):
        s = [x.shape[0], HEADS, D // HEADS]
        h = self.lnorm(x)
        qs = self.lnorm_q(self.proj_q(h).reshape(s)).to(BF16)
        ks = self.lnorm_k(self.proj_k(h).reshape(s)).to(BF16)
        vs = self.proj_v(h).reshape(s)
        out = x + (qs * ks * vs).flatten(1)
        return out + self.mlp(self.mlp_norm(out))


def _twins(param_dtype=BF16, seed=0):
    torch.manual_seed(seed)
    ref = _Prologue(torch.nn.LayerNorm)
    torch.nn.init.normal_(ref.mlp_norm.weight, 1.0, 0.1)
    torch.nn.init.normal_(ref.mlp_norm.bias, 0.0, 0.1)
    ref = ref.to(param_dtype)
    new = _Prologue(AutocastDtypeLayerNorm).to(param_dtype)
    new.load_state_dict(ref.state_dict())
    return ref, new


def _run(module, x, grad_out, wrap=None):
    x = x.detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=BF16):
        out = wrap(module, x) if wrap else module(x)
    out.backward(grad_out)
    grads = {n: p.grad for n, p in module.named_parameters()}
    return out, x.grad, grads


def _assert_bf16_close(a, b):
    # within bf16 rounding: the reference rounds the float32 norm output once at the consumer
    torch.testing.assert_close(a.float(), b.float(), rtol=2**-7, atol=2**-7)


def test_cpu_autocast_policy_emulated(cuda_like_autocast_cpu):
    x = torch.randn(8, D, dtype=BF16)
    with torch.autocast("cpu", dtype=BF16):
        assert torch.nn.LayerNorm(D)(x).dtype == torch.float32
        assert AutocastDtypeLayerNorm(D)(x).dtype == BF16


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("param_dtype", [torch.float32, BF16])
@pytest.mark.parametrize("in_dtype", [BF16, torch.float32])
def test_matches_fp32_norm_then_cast(cuda_like_autocast_cpu, affine, param_dtype, in_dtype):
    torch.manual_seed(1)
    ref = torch.nn.LayerNorm(D, elementwise_affine=affine)
    if affine:
        torch.nn.init.normal_(ref.weight, 1.0, 0.1)
        torch.nn.init.normal_(ref.bias, 0.0, 0.1)
    ref = ref.to(param_dtype)
    new = AutocastDtypeLayerNorm(D, elementwise_affine=affine).to(param_dtype)
    new.load_state_dict(ref.state_dict())
    x = (3.0 * torch.randn(512, D) + 1.0).to(in_dtype)
    grad_out = torch.randn(512, D, dtype=BF16)

    out_ref, gx_ref, gp_ref = _run(ref, x, grad_out, wrap=lambda m, x: m(x).to(BF16))
    out_new, gx_new, gp_new = _run(new, x, grad_out)

    assert out_ref.dtype == out_new.dtype == BF16
    assert gx_new.dtype == in_dtype
    _assert_bf16_close(out_new, out_ref)
    _assert_bf16_close(gx_new, gx_ref)
    for name, g in gp_ref.items():
        assert gp_new[name].dtype == param_dtype
        # CPU's bfloat16 layer_norm backward rounds the weight/bias reductions in bfloat16
        # (~1 % of the largest entry over 512 rows; MPS accumulates in float32: ~0.2 %)
        atol = 2**-5 * g.abs().max().item()
        torch.testing.assert_close(gp_new[name].float(), g.float(), rtol=0, atol=atol)


def test_forward_and_backward_cast_count_drops(cuda_like_autocast_cpu):
    ref, new = _twins()
    x = torch.randn(256, D, dtype=BF16, requires_grad=True)
    counts = {}
    for label, module in (("ref", ref), ("new", new)):
        with _CastCounter() as fwd, torch.autocast("cpu", dtype=BF16):
            out = module(x)
        with _CastCounter() as bwd:
            out.sum().backward()
        counts[label] = (fwd.n, bwd.n)
    # ref: 4 norms cast up (4), 4 Linears cast the fp32 norm output down (4), q/k .to (2)
    assert counts["ref"] == (10, 10)
    assert counts["new"] == (0, 0)


def _nested(module, x):
    return checkpoint(lambda y: checkpoint(module, y, use_reentrant=False), x, use_reentrant=False)


@pytest.mark.parametrize("wrap", ["checkpoint", "nested"])
def test_checkpoint_recompute_matches(cuda_like_autocast_cpu, wrap):
    _, new = _twins()
    x = torch.randn(256, D, dtype=BF16)
    grad_out = torch.randn(256, D, dtype=BF16)
    fn = _nested if wrap == "nested" else lambda m, y: checkpoint(m, y, use_reentrant=False)

    out_plain, gx_plain, gp_plain = _run(new, x, grad_out)
    new.zero_grad(set_to_none=True)
    out_ckpt, gx_ckpt, gp_ckpt = _run(new, x, grad_out, wrap=fn)

    assert torch.equal(out_ckpt, out_plain)
    assert torch.equal(gx_ckpt, gx_plain)
    for name, g in gp_plain.items():
        assert torch.equal(gp_ckpt[name], g)


def _norm_in_input_dtype(norm, x):
    """The July 2026 attempt (e79d23edc): output in the dtype of the input."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return norm(x).to(x.dtype)


class _SkippedUnitCast(torch.nn.Module):
    """A unit whose input cast runs in the forward but not in the checkpoint recompute.

    FSDP2 casts unit inputs to `param_dtype` in its pre-forward hook, which returns early when
    the unit is already in PRE_BACKWARD, as it is in the recompute of a functional checkpoint
    aligned with the unit (docs/fsdp2-activation-checkpointing.md on d274b18d2).
    """

    def __init__(self, norm_fn):
        super().__init__()
        self.norm_fn = norm_fn
        self.proj = torch.nn.Linear(D, D, bias=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls == 1:
            x = x.to(BF16)
        return self.proj(self.norm_fn(x))


def test_fsdp_skipped_input_cast_in_recompute(cuda_like_autocast_cpu):
    norm = torch.nn.LayerNorm(D, elementwise_affine=False)
    x = torch.randn(64, D, requires_grad=True)  # float32, as at a root-unit -> unit boundary

    # autocast's float32 norm hides the skipped cast: float32 out in both passes
    unit = _SkippedUnitCast(norm)
    with torch.autocast("cpu", dtype=BF16):
        out = checkpoint(unit, x, use_reentrant=False)
    out.sum().backward()

    # the earlier attempt replays fine when the unit input has the same dtype in both passes:
    # checkpoint restores the outer autocast state and the inner disabled region re-runs
    unit = _SkippedUnitCast(lambda y: _norm_in_input_dtype(norm, y))
    with torch.autocast("cpu", dtype=BF16):
        out = _nested(unit, x.to(BF16))
    out.sum().backward()

    # ... and fails the recompute's metadata check when FSDP2 skips the input cast
    unit = _SkippedUnitCast(lambda y: _norm_in_input_dtype(norm, y))
    with torch.autocast("cpu", dtype=BF16):
        out = checkpoint(unit, x, use_reentrant=False)
    with pytest.raises(CheckpointError, match="different metadata"):
        out.sum().backward()

    # the autocast-dtype norm saves bfloat16 in both passes
    norm.__class__ = AutocastDtypeLayerNorm
    unit = _SkippedUnitCast(norm)
    with torch.autocast("cpu", dtype=BF16):
        out = checkpoint(unit, x, use_reentrant=False)
    out.sum().backward()
    assert unit.calls == 2


def test_no_autocast_is_layer_norm():
    torch.manual_seed(2)
    ref = torch.nn.LayerNorm(D)
    torch.nn.init.normal_(ref.weight, 1.0, 0.1)
    new = AutocastDtypeLayerNorm(D)
    new.load_state_dict(ref.state_dict())
    for dtype in (torch.float32, BF16):
        x = torch.randn(32, D, dtype=dtype)
        assert torch.equal(new.to(dtype)(x), ref.to(dtype)(x))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_autocast():
    """MPS autocast puts layer_norm on its float32 list, like CUDA."""
    ref, new = _twins()
    ref, new = ref.to("mps"), new.to("mps")
    x = torch.randn(256, D, dtype=BF16, device="mps")
    with torch.autocast("mps", dtype=BF16):
        assert ref.lnorm(x).dtype == torch.float32
        out_ref, out_new = ref(x), new(x)
        assert new.lnorm(x).dtype == BF16
    _assert_bf16_close(out_new, out_ref)


def test_selection():
    try:
        from weathergen.model.layernorm_bf16 import use_autocast_dtype_layernorms
        from weathergen.model.layers import MLP
        from weathergen.model.norms import AdaLayerNormLayer
    except ImportError as e:  # flash_attn on machines without CUDA
        pytest.skip(f"model imports: {e}")
    from weathergen.model.attention import MultiCrossAttentionHeadVarlen

    model = torch.nn.ModuleDict(
        {
            "attn": MultiCrossAttentionHeadVarlen(D, D, HEADS, dim_aux=8),
            "mlp": MLP(D, D),
            "mlp_ada": MLP(D, D, dim_aux=8),
            "adaln": AdaLayerNormLayer(D, 8, torch.nn.Identity()),
            "ln": torch.nn.LayerNorm(D),
        }
    )
    state = {k: v.clone() for k, v in model.state_dict().items()}

    counts, left = use_autocast_dtype_layernorms(model)
    assert counts == {"attention": 3, "mlp": 1}
    assert sorted(left) == [
        "adaln.ln",
        "attn.lnorm_in_q.norm",
        "ln",
        "mlp_ada.layers.0.norm",
    ]
    assert type(model["attn"].lnorm_in_kv) is AutocastDtypeLayerNorm
    assert type(model["mlp"].layers[0]) is AutocastDtypeLayerNorm
    assert model.state_dict().keys() == state.keys()

    counts, left = use_autocast_dtype_layernorms(model, with_adaln=True)
    assert counts == {"adaln": 3}
    assert left == ["ln"]
