# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from weathergen.train.loss_modules.loss_functions import lp_loss, spatial_lp_loss
from weathergen.train.loss_modules.loss_module_physical import LossPhysical


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor(
        [
            [0.5, float("nan"), -1.0],
            [1.5, 0.0, 2.0],
            [-0.5, 1.0, 0.5],
            [2.0, -1.0, float("nan")],
            [0.0, 1.5, 3.0],
            [1.0, -0.5, -2.0],
            [3.0, 2.5, 1.0],
        ],
        dtype=torch.float64,
    )
    pred = torch.arange(42, dtype=torch.float64).reshape(2, 7, 3) / 11.0 - 1.0
    weights_channels = torch.tensor([0.5, 1.25, 2.0], dtype=torch.float64)
    weights_points = torch.tensor([0.75, 1.0, 1.5, 0.25, 2.0, 1.25, 0.5], dtype=torch.float64)
    return target, pred, weights_channels, weights_points


def _check_distributed_equivalence(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        target, pred_data, weights_channels, weights_points = _inputs()
        bounds = [(0, 3), (3, 3), (3, 7)]
        start, end = bounds[rank]
        cases = [
            (1, False, True, False),
            (2, False, True, True),
            (2, True, True, True),
            (2, False, False, True),
        ]

        for p_norm, with_p_root, with_mean, with_weights in cases:
            pred_full = pred_data.clone().requires_grad_()
            reference, reference_chs = lp_loss(
                target,
                pred_full,
                p_norm=p_norm,
                with_p_root=with_p_root,
                with_mean=with_mean,
                weights_channels=weights_channels if with_weights else None,
                weights_points=weights_points if with_weights else None,
            )
            reference.backward()

            pred_local = pred_data[:, start:end].clone().requires_grad_()
            actual, actual_chs = spatial_lp_loss(
                target[start:end],
                pred_local,
                p_norm=p_norm,
                with_p_root=with_p_root,
                with_mean=with_mean,
                weights_channels=weights_channels if with_weights else None,
                weights_points=weights_points[start:end] if with_weights else None,
                spatial_group=dist.group.WORLD,
            )
            actual.backward()

            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(actual_chs, reference_chs, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(
                pred_local.grad,
                pred_full.grad[:, start:end],
                rtol=1e-12,
                atol=1e-12,
            )

        # FSDP/DDP averages parameter gradients across all ranks. Scaling each
        # local contribution by the spatial size restores the full-sample sum.
        feature = pred_data[0, :, :1]
        parameter_reference = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
        pred_reference = (parameter_reference * feature).unsqueeze(0)
        loss_reference, _ = lp_loss(target[:, :1], pred_reference, p_norm=2)
        loss_reference.backward()

        parameter_local = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
        pred_local = (parameter_local * feature[start:end]).unsqueeze(0)
        loss_local, _ = spatial_lp_loss(
            target[start:end, :1],
            pred_local,
            p_norm=2,
            spatial_group=dist.group.WORLD,
            local_gradient_scale=float(world_size),
        )
        loss_local.backward()
        dist.all_reduce(parameter_local.grad, op=dist.ReduceOp.SUM)
        parameter_local.grad /= world_size
        torch.testing.assert_close(
            parameter_local.grad,
            parameter_reference.grad,
            rtol=1e-12,
            atol=1e-12,
        )

        # Tokenized-spacetime streams must enter the same sequence of loss
        # collectives even when one rank has no local timestamps.
        module = LossPhysical.__new__(LossPhysical)
        module.device = "cpu"
        module.spatial_local_loss = True
        module.spatial_parallel_size = world_size
        module.spatial_parallel_group = dist.group.WORLD
        all_times = np.array(
            ["2026-01-01T00", "2026-01-01T01", "2026-01-01T01", "2026-01-01T02"],
            dtype="datetime64[ns]",
        )
        time_bounds = [(0, 1), (1, 1), (1, 4)]
        time_start, time_end = time_bounds[rank]
        masks = module._get_substep_masks(
            {"tokenize_spacetime": True},
            output_step=0,
            target_times=all_times[time_start:time_end],
        )
        assert len(masks) == 3
        assert sum(int(mask.sum()) for mask in masks) == time_end - time_start
    finally:
        dist.destroy_process_group()


def test_spatial_lp_loss_matches_full_loss_and_local_gradients(tmp_path: Path) -> None:
    rendezvous = tmp_path / "spatial-loss-rendezvous"
    mp.spawn(
        _check_distributed_equivalence,
        args=(3, str(rendezvous)),
        nprocs=3,
        join=True,
    )


def test_spatial_lp_loss_none_is_an_exact_local_path() -> None:
    target, pred_data, weights_channels, weights_points = _inputs()
    expected = lp_loss(
        target,
        pred_data,
        p_norm=2,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )
    actual = spatial_lp_loss(
        target,
        pred_data,
        p_norm=2,
        weights_channels=weights_channels,
        weights_points=weights_points,
    )

    assert actual[0] == pytest.approx(expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_spatial_lp_loss_requires_distributed_for_a_group() -> None:
    target, pred_data, _, _ = _inputs()

    with pytest.raises(RuntimeError, match="initialized torch.distributed"):
        spatial_lp_loss(
            target,
            pred_data,
            p_norm=2,
            spatial_group=object(),  # type: ignore[arg-type]
        )
