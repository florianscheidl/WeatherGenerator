import torch

from weathergen.model.utils import maybe_checkpoint, set_inline_checkpointing


class _ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 1


def test_maybe_checkpoint_calls_module_directly_when_disabled():
    module = _ToyModule()
    set_inline_checkpointing(module, enabled=False)

    x = torch.tensor([1.0], requires_grad=True)
    y = maybe_checkpoint(module, x, use_reentrant=False)

    assert module.calls == 1
    assert torch.equal(y, torch.tensor([2.0]))


def test_maybe_checkpoint_defaults_to_enabled():
    module = _ToyModule()

    x = torch.tensor([1.0], requires_grad=True)
    y = maybe_checkpoint(module, x, use_reentrant=False)

    assert torch.equal(y, torch.tensor([2.0]))
