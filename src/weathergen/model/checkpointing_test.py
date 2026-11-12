"""Activation-checkpointing policy: defaults keep today's behaviour, every setting is bitwise
equal in outputs and gradients, and the forward counts follow the nesting (CPU)."""

import collections
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.model import attention as attention_module
from weathergen.model.checkpointing import (
    BLOCK_REGIONS,
    OUTER_REGIONS,
    CheckpointPolicy,
    maybe_checkpoint,
)
from weathergen.model.engines import GlobalAssimilationEngine

D = 16
NUM_BLOCKS = 3  # attention + MLP pairs: 6 blocks


def _cf(**ckpt):
    cf = {
        "ae_global_att_dense_rate": 1.0,
        "ae_global_num_blocks": NUM_BLOCKS,
        "ae_global_dim_embed": D,
        "ae_global_num_heads": 2,
        "ae_global_dropout_rate": 0.1,
        "ae_global_with_qk_lnorm": True,
        "ae_global_mlp_hidden_factor": 2,
        "ae_local_num_queries": 1,
        "with_flash_attention": True,
        "norm_type": "LayerNorm",
        "norm_eps": 1e-6,
        "mlp_norm_eps": 1e-6,
        "attention_dtype": "fp32",
    }
    if ckpt:
        cf["activation_checkpointing"] = ckpt
    return OmegaConf.create(cf)


def _run(outer: bool, blocks: bool):
    """Global engine under an encoder-style outer checkpoint; returns out, grads, counts."""
    torch.manual_seed(0)
    cf = _cf(blocks={"global_assimilation": blocks})
    engine = GlobalAssimilationEngine(cf, num_healpix_cells=4)
    engine.train()
    counts = collections.Counter()
    for i, block in enumerate(engine.ae_global_blocks):
        block.register_forward_pre_hook(lambda m, a, i=i: counts.update([i]))
    x = torch.randn(1, 8, D, generator=torch.Generator().manual_seed(1), requires_grad=True)
    torch.manual_seed(123)  # dropout stream
    out, _ = maybe_checkpoint(outer, engine, x, use_reentrant=False)
    grad = torch.randn(out.shape, generator=torch.Generator().manual_seed(2))
    (out * grad).sum().backward()
    grads = {n: p.grad.clone() for n, p in engine.named_parameters()}
    grads["x"] = x.grad.clone()
    return out.detach(), grads, counts


@pytest.fixture(autouse=True)
def sdpa_for_flash(monkeypatch):
    def fn(q, k, v, softcap=0.0, dropout_p=0.0):
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p
        ).transpose(1, 2)

    monkeypatch.setattr(attention_module, "flash_attn_func", fn)


@pytest.fixture
def reference():
    return _run(outer=True, blocks=True)


@pytest.mark.parametrize("outer", [True, False])
@pytest.mark.parametrize("blocks", [True, False])
def test_every_setting_is_bitwise_equal(reference, outer, blocks):
    out, grads, _ = _run(outer, blocks)
    ref_out, ref_grads, _ = reference
    assert torch.equal(out, ref_out)
    assert grads.keys() == ref_grads.keys()
    for name, g in grads.items():
        assert torch.equal(g, ref_grads[name]), name


def test_forward_counts_follow_the_nesting():
    n = 2 * NUM_BLOCKS
    both = _run(True, True)[2]
    # forward + outer recompute + block recompute; the last block is not rebuilt by the outer
    # recompute (early stop: nothing after it is needed)
    assert [both[i] for i in range(n)] == [3] * (n - 1) + [2]
    assert [_run(False, True)[2][i] for i in range(n)] == [2] * n
    assert [_run(True, False)[2][i] for i in range(n)] == [2] * n
    assert [_run(False, False)[2][i] for i in range(n)] == [1] * n


def test_policy_defaults_and_parsing():
    p = CheckpointPolicy.from_config(_cf())
    assert all(p.outer(r) for r in OUTER_REGIONS) and all(p.blocks(r) for r in BLOCK_REGIONS)
    p = CheckpointPolicy.from_config(_cf(outer_engines=False))
    assert not any(p.outer(r) for r in OUTER_REGIONS) and all(p.blocks(r) for r in BLOCK_REGIONS)
    p = CheckpointPolicy.from_config(_cf(outer_engines={"embedding": False}, blocks=False))
    assert not p.outer("embedding") and p.outer("assimilate_local")
    assert not any(p.blocks(r) for r in BLOCK_REGIONS)
    assert CheckpointPolicy.from_config(None).outer("embedding")


@pytest.mark.parametrize(
    "bad", [{"outer_engines": {"nope": False}}, {"blocks": {"x": 1}}, {"other": True}]
)
def test_policy_rejects_unknown_keys(bad):
    with pytest.raises(ValueError):
        CheckpointPolicy.from_config(_cf(**bad))


def test_default_config_section_is_all_on():
    path = Path(__file__).parents[3] / "config" / "default_config.yml"
    section = OmegaConf.load(path).activation_checkpointing
    assert set(section.outer_engines) == set(OUTER_REGIONS)
    assert set(section.blocks) == set(BLOCK_REGIONS)
    p = CheckpointPolicy.from_config(OmegaConf.create({"activation_checkpointing": section}))
    assert all(p.outer(r) for r in OUTER_REGIONS) and all(p.blocks(r) for r in BLOCK_REGIONS)
    # both override forms of the command line merge into the defaults
    for dotlist, off in [
        (["activation_checkpointing.outer_engines=False"], set(OUTER_REGIONS)),
        (["activation_checkpointing.outer_engines.embedding=False"], {"embedding"}),
    ]:
        cf = OmegaConf.merge(
            OmegaConf.create({"activation_checkpointing": section}), OmegaConf.from_dotlist(dotlist)
        )
        p = CheckpointPolicy.from_config(cf)
        assert {r for r in OUTER_REGIONS if not p.outer(r)} == off
