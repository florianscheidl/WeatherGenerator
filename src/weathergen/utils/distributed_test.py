import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from weathergen.utils.distributed import ddp_average


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _average_on_rank(rank: int, world_size: int, port: int, results) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        results[rank] = ddp_average(torch.tensor([float(rank), 2.0 * rank])).tolist()
    finally:
        dist.destroy_process_group()


def test_ddp_average_with_gloo():
    """Gloo (CPU) has no ReduceOp.AVG; ddp_average must still average."""
    world_size = 2
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_average_on_rank, args=(world_size, _free_port(), results), nprocs=world_size)
        assert dict(results) == {0: [0.5, 1.0], 1: [0.5, 1.0]}


def test_ddp_average_without_distributed():
    assert ddp_average(torch.tensor([1.0, 3.0])).tolist() == [1.0, 3.0]
