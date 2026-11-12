# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Full stack: combined options, `layernorm_impl=triton` and the checkpoint policy K1.

K1 = `activation_checkpointing.outer_engines.assimilate_local=False` and
`activation_checkpointing.blocks.global_assimilation=False`. CPU, bfloat16 autocast, Triton
LayerNorm through its reference path (`FORCE_REFERENCE_ON_CPU`). Non-reentrant checkpoint where
the forward input is float32 and the recompute input bfloat16 (FSDP2 skips its input cast in the
aligned recompute). Must run without `CheckpointError`, and the K1 setting must give the same
output and gradients as full checkpointing.
"""

import pytest
import torch

from weathergen.model import layernorm_triton as lt
from weathergen.model.checkpointing import CheckpointPolicy, maybe_checkpoint
from weathergen.model.combined_options_test import (  # noqa: F401
    Block,
    _inputs,
    _make,
    cuda_like_autocast_cpu,
    sdpa_for_flash,
)

BF16 = torch.bfloat16


@pytest.fixture
def triton_reference_on_cpu(monkeypatch):
    monkeypatch.setattr(lt, "FORCE_REFERENCE_ON_CPU", True)


def _run_k1(m, policy, recompute_dtype):
    x_in, x, coords, aux, grad = _inputs()
    x = x.requires_grad_()
    m.zero_grad(set_to_none=True)
    calls = {}

    def make_fn(name, fn):
        def wrapped(t):
            calls[name] = calls.get(name, 0) + 1
            # forward gets float32, the recompute gets the other dtype
            return fn(t if calls[name] == 1 else t.to(recompute_dtype))

        return wrapped

    def wrap(fn, t):
        # the first call site is the attention block (checkpointed), the second the global
        # assimilation block (K1: not checkpointed)
        name = f"site{len(calls_sites)}"
        calls_sites.append(name)
        enabled = policy._blocks["local_assimilation"] if name == "site0" else (
            policy._blocks["global_assimilation"]
        )
        return maybe_checkpoint(enabled, make_fn(name, fn), t, use_reentrant=False)

    calls_sites = []
    with torch.autocast("cpu", dtype=BF16):
        out = m(x_in, x, coords, aux, wrap)
    out.backward(grad.to(out.dtype))
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return out.detach(), grads, calls


@pytest.mark.parametrize("recompute_dtype", [torch.float32, BF16])
def test_k1_triton_checkpoint_fp32_forward_bf16_recompute(
    cuda_like_autocast_cpu, sdpa_for_flash, triton_reference_on_cpu, recompute_dtype
):
    m = _make(options_on=True)
    counts = lt.use_triton_layernorms(m)
    assert counts["float32"] > 0  # the AdaLN norms are TritonLayerNorm now
    k1 = CheckpointPolicy(
        outer_engines={"assimilate_local": False}, blocks={"global_assimilation": False}
    )
    full = CheckpointPolicy()
    out_full, g_full, calls_full = _run_k1(m, full, recompute_dtype)  # no CheckpointError
    out_k1, g_k1, calls_k1 = _run_k1(m, k1, recompute_dtype)
    # full: both sites run twice; K1: only the checkpointed site runs twice
    assert calls_full == {"site0": 2, "site1": 2}
    assert calls_k1 == {"site0": 2, "site1": 1}
    if recompute_dtype == torch.float32:
        assert torch.equal(out_full, out_k1)
        for k, v in g_full.items():
            assert torch.equal(v, g_k1[k]), k
    else:
        # the bfloat16 recompute input rounds the recomputed activations: close, not bitwise
        assert torch.equal(out_full, out_k1)
        for k, v in g_full.items():
            s = v.abs().max().clamp_min(1e-6)
            assert (g_k1[k] - v).abs().max() / s < 5e-2, k
    assert all(torch.isfinite(v).all() for v in g_full.values())
