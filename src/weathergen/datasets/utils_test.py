import astropy_healpix as hp
import numpy as np
import pytest
import torch
from torch import tensor

from weathergen.datasets.utils import (
    cell_partition_mask,
    locs_to_cell_coords_ctrs,
    locs_to_ctr_coords,
    vecs_to_rots,
)


def _locs_to_cell_coords_ctrs(
    healpix_centers_rots: torch.Tensor, locs: list[torch.Tensor]
) -> torch.Tensor:
    return torch.cat(
        [
            torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1)
            if len(s) > 0
            else torch.tensor([])
            for _, (R, s) in enumerate(zip(healpix_centers_rots, locs, strict=False))
        ]
    )


def test_locs_to_cell_coords_ctrs():
    locs = [
        tensor(
            [
                [0.7235, -0.6899, -0.0245],
                [0.7178, -0.6951, -0.0408],
                [0.7288, -0.6835, -0.0408],
                [0.7229, -0.6886, -0.0571],
            ]
        ),
        tensor(
            [
                [0.6899, -0.7235, -0.0245],
                [0.6835, -0.7288, -0.0408],
                [0.6951, -0.7178, -0.0408],
                [0.6886, -0.7229, -0.0571],
            ]
        ),
        tensor([]),
    ]
    hp_centers_rots = tensor(
        [
            [
                [7.0711e-01, 7.0711e-01, 6.1232e-17],
                [-7.0711e-01, 7.0711e-01, -2.5363e-17],
                [-6.1232e-17, -2.5363e-17, 1.0000e00],
            ],
            [
                [6.8939e-01, 7.2409e-01, 2.0833e-02],
                [-7.2409e-01, 6.8965e-01, -8.9294e-03],
                [-2.0833e-02, -8.9294e-03, 9.9974e-01],
            ],
            [
                [7.2409e-01, 6.8939e-01, 2.0833e-02],
                [-6.8939e-01, 7.2434e-01, -8.3304e-03],
                [-2.0833e-02, -8.3304e-03, 9.9975e-01],
            ],
            [
                [7.0649e-01, 7.0649e-01, 4.1667e-02],
                [-7.0649e-01, 7.0751e-01, -1.7250e-02],
                [-4.1667e-02, -1.7250e-02, 9.9898e-01],
            ],
        ]
    )
    torch.testing.assert_close(
        _locs_to_cell_coords_ctrs(hp_centers_rots, locs),
        locs_to_cell_coords_ctrs(hp_centers_rots, locs),
    )


# def _tcs_simpled(target_coords: list[Tensor]) -> tuple[list[Tensor], Tensor]:
#     tcs = [
#         (
#             s2tor3(
#                 torch.deg2rad(90.0 - t[..., 0]),
#                 torch.deg2rad(180.0 + t[..., 1]),
#             )
#             if len(t) > 0
#             else torch.tensor([])
#         )
#         for t in target_coords
#     ]
#     cat_target_coords = torch.cat(target_coords)
#     return tcs, cat_target_coords


# def test_tcs():
#     target_coords = [
#         tensor(
#             [[2.3377, -135.0000], [1.4026, -135.4545], [1.4026, -134.5455], [0.4675, -135.0000]]
#         ),
#         tensor(
#             [[3.2727, -133.6082], [2.3377, -134.0816], [2.3377, -133.1633], [1.4026, -133.6364]]
#         ),
#     ]
#     tcs_ref, cat_tcs_ref = _tcs_simpled(target_coords)
#     tcs_opt, cat_tcs_opt = tcs_optimized(target_coords)
#     assert len(tcs_ref) == len(tcs_opt)
#     torch.testing.assert_close(cat_tcs_ref, cat_tcs_opt)
#     torch.testing.assert_close(tcs_ref, tcs_opt, atol=1e-8, rtol=1e-5)


def _locs_to_ctr_coords(ctrs_r3, locs: list[torch.Tensor]) -> list[torch.Tensor]:
    ctrs_rots = vecs_to_rots(ctrs_r3).to(torch.float32)

    ## express each centroid in local coordinates w.r.t to healpix center
    #  by rotating center to origin
    return [
        (
            torch.matmul(R, s.transpose(-1, -2)).transpose(-2, -1)
            if len(s) > 0
            else torch.zeros([0, 3])
        )
        for i, (R, s) in enumerate(zip(ctrs_rots, locs, strict=False))
    ]


def test_locs_to_ctr_coords():
    locs = [
        tensor(
            [
                [0.7235, -0.6899, -0.0245],
                [0.7178, -0.6951, -0.0408],
                [0.7288, -0.6835, -0.0408],
                [0.7229, -0.6886, -0.0571],
            ]
        ),
        tensor(
            [
                [0.6899, -0.7235, -0.0245],
                [0.6835, -0.7288, -0.0408],
                [0.6951, -0.7178, -0.0408],
                [0.6886, -0.7229, -0.0571],
            ]
        ),
        tensor([]),
    ]
    ctrs_r3 = tensor(
        [
            [7.2425e-01, 6.8954e-01, 6.1232e-17],
            [7.0695e-01, 7.0695e-01, 2.0833e-02],
            [7.4079e-01, 6.7141e-01, 2.0833e-02],
        ]
    )
    torch.testing.assert_close(
        locs_to_ctr_coords(ctrs_r3, locs),
        _locs_to_ctr_coords(ctrs_r3, locs),
    )


def test_cell_partition_mask_is_a_partition():
    """Over all shards the masks must be disjoint and cover the grid."""
    hl_data, hl_split, num_shards = 5, 3, 4

    masks = [
        cell_partition_mask(hl_data, hl_split, num_shards, i).numpy() for i in range(num_shards)
    ]

    assert np.array(masks).sum(axis=0).tolist() == [1] * (12 * 4**hl_data)


def test_cell_partition_mask_balances_cells():
    hl_data, hl_split, num_shards = 5, 3, 4
    num_cells = 12 * 4**hl_data

    for i in range(num_shards):
        mask = cell_partition_mask(hl_data, hl_split, num_shards, i)
        assert int(mask.sum()) == num_cells // num_shards


def test_cell_partition_mask_matches_nested_child_ordering():
    """The projection to the data level assumes nested ordering -- check it holds."""
    hl_data, hl_split, num_shards, shard_idx = 5, 3, 4, 1

    mask = cell_partition_mask(hl_data, hl_split, num_shards, shard_idx)

    # map the selected data-level cells back to their coarse cell via astropy, rather than
    # by repeating the index arithmetic the implementation itself uses
    data_cells = np.flatnonzero(mask.numpy())
    lon, lat = hp.healpix_to_lonlat(data_cells, nside=2**hl_data, order="nested")
    parents = hp.lonlat_to_healpix(lon, lat, nside=2**hl_split, order="nested")

    expected = np.flatnonzero(np.arange(12 * 4**hl_split) % num_shards == shard_idx)
    assert sorted(set(np.unique(parents).tolist())) == expected.tolist()


def test_cell_partition_mask_split_at_data_level():
    """healpix_level_split == healpix_level selects every num_shards-th cell."""
    mask = cell_partition_mask(3, 3, num_shards=4, shard_idx=2)

    assert np.flatnonzero(mask.numpy()).tolist() == list(range(2, 12 * 4**3, 4))


def test_cell_partition_mask_split_at_base_resolution():
    """healpix_level_split == 0 partitions the 12 base cells."""
    mask = cell_partition_mask(2, 0, num_shards=4, shard_idx=0)

    # base cells 0, 4, 8, each covering 4**2 children
    assert np.flatnonzero(mask.numpy()).tolist() == (
        list(range(0, 16)) + list(range(64, 80)) + list(range(128, 144))
    )


def test_cell_partition_mask_rejects_split_finer_than_data():
    with pytest.raises(AssertionError, match="healpix_level_split"):
        cell_partition_mask(3, 5, num_shards=4, shard_idx=0)


def test_cell_partition_mask_rejects_uneven_partition():
    # 12 * 4**3 = 768 cells do not divide evenly across 5 shards
    with pytest.raises(AssertionError, match="evenly"):
        cell_partition_mask(5, 3, num_shards=5, shard_idx=0)


def test_cell_partition_mask_rejects_out_of_range_shard():
    with pytest.raises(AssertionError, match="shard_idx"):
        cell_partition_mask(5, 3, num_shards=4, shard_idx=4)
