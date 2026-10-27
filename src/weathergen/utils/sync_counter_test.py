import pytest
import torch

from weathergen.utils.sync_counter import SyncCounter, format_report

requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _kinds(counter: SyncCounter) -> dict[str, int]:
    return counter.report()["totals"]


def test_scalar_reads_count_as_syncs():
    x = torch.arange(4.0)
    with SyncCounter({"cpu"}) as counter:
        x.sum().item()
        int(x.max())
        bool(x.sum() > 0)
        torch.arange(x.max())
    assert _kinds(counter) == {"scalar_read": 4}


def test_data_dependent_shapes_count_as_syncs():
    x = torch.arange(4.0, requires_grad=True)
    lens = torch.tensor([1, 3])
    with SyncCounter({"cpu"}) as counter:
        y = x[x > 1.0]  # forward index, and index_put in its backward
        torch.repeat_interleave(lens)
        (x > 1.0).nonzero()
        y.sum().backward()
    assert _kinds(counter) == {"data_dependent_shape": 4}
    sites = {s["site"] for s in counter.report()["sites"]}
    assert "backward: IndexBackward0" in sites


def test_sync_free_formulations_do_not_count():
    lens = torch.tensor([1, 3])
    x = torch.arange(4.0)
    with SyncCounter({"cpu"}) as counter:
        torch.repeat_interleave(lens, output_size=4)
        torch.searchsorted(lens.cumsum(0), torch.arange(4), right=True)
        x.index_select(0, torch.arange(2))
        torch.where(x > 1.0, x, 0.0)
    assert _kinds(counter) == {}


def test_counts_per_step_and_site():
    x = torch.arange(4.0)
    with SyncCounter({"cpu"}) as counter:
        for _ in range(3):
            x.sum().item()
            counter.next_step()
    report = counter.report()
    assert report["num_steps"] == 3
    assert report["steps"] == [{"scalar_read": 1}] * 3
    (site,) = report["sites"]
    assert site["per_step"] == 1.0
    assert site["site"].startswith("src/weathergen/utils/sync_counter_test.py:")
    assert "scalar_read" in format_report(report)


def test_cpu_ops_do_not_count_by_default():
    x = torch.arange(4.0)
    with SyncCounter() as counter:
        x.sum().item()
        x[x > 1.0]
    assert _kinds(counter) == {}


@requires_mps
def test_host_device_copies():
    device = torch.device("mps")
    with SyncCounter() as counter:
        torch.ones(3).to(device)
        torch.tensor(1.0, device=device)
        torch.ones(3, device=device).cpu()
        torch.ones(3, device=device).tolist()
        # a scalar setitem copies from the host on CUDA, but not on MPS (see the module doc)
        torch.ones(3, device=device)[0] = 2.0
    assert _kinds(counter) == {"h2d_copy": 1, "h2d_tensor_ctor": 1, "d2h_copy": 2}
