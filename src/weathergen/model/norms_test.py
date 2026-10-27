import pytest
import torch

from weathergen.model.norms import AdaLayerNormLayer, modulate
from weathergen.utils.sync_counter import SyncCounter


def _reference(layer: AdaLayerNormLayer, x, c, x_lens):
    """repeat_interleave formulation the searchsorted one replaced (syncs on a device)."""
    shift, scale, gate = layer.adaLN_modulation(c)[torch.repeat_interleave(x_lens) - 1].chunk(
        3, dim=1
    )
    return gate * layer.layer(modulate(layer.ln(x), shift, scale), x_lens=x_lens) + x


@pytest.mark.parametrize("seed", range(4))
def test_ada_layer_norm_matches_repeat_interleave(seed):
    gen = torch.Generator().manual_seed(seed)
    lens = torch.randint(0, 4, (7,), generator=gen, dtype=torch.int32)
    lens[[1, 4]] = 0  # empty sequences
    x_lens = torch.cat([torch.zeros(1, dtype=torch.int32), lens])  # leading 0 as for flash
    x = torch.randn(int(lens.sum()), 8, generator=gen)
    c = torch.randn(len(lens), 5, generator=gen)
    layer = AdaLayerNormLayer(8, 5, lambda x, **kwargs: 2.0 * x)
    torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, generator=gen)

    with SyncCounter({"cpu"}) as counter:
        out = layer(x, c, x_lens)

    assert torch.equal(out, _reference(layer, x, c, x_lens))
    assert counter.report()["totals"] == {}
