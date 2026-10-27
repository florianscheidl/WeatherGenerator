# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Compare two model checkpoints tensor by tensor: bitwise equal, or the largest differences.

Used as an equivalence check for changes that must not change what the model computes (e.g.
removing host-device syncs): train a few steps deterministically (CPU, fp32, fixed seed) from
each code state and compare the checkpoints. Every parameter depends on every step's losses
and gradients, so bitwise-equal checkpoints mean equal training.

    uv run python scripts/compare_checkpoints.py A.chkpt B.chkpt [--atol 0]

Exits with 0 if all tensors are equal within --atol (bitwise by default), else 1.

Training is bitwise reproducible on the CPU only single-threaded: multi-threaded reductions
sum in a varying order (differences around 1e-9 after a few steps). For each code state:

    OMP_NUM_THREADS=1 WEATHERGEN_PRIVATE_CONF=./private_config.yaml uv run train \\
      --base-config ./config/era5_local.yml --config ./config/era5_local_forecasting.yml \\
      --run-id <id> --options device=cpu with_mixed_precision=False attention_dtype=fp32 \\
      data_loading.rng_seed=42 data_loading.num_workers=0 \\
      training_config.samples_per_mini_epoch=4 validation_config.samples_per_mini_epoch=1

then compare models/<id A>/<id A>_chkpt00001.chkpt with the one of B.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def compare(a: dict, b: dict, atol: float) -> list[str]:
    """Differences between the state dicts `a` and `b`, one line each; empty if equal."""
    problems = []
    for key in sorted(a.keys() - b.keys()):
        problems.append(f"only in A: {key}")
    for key in sorted(b.keys() - a.keys()):
        problems.append(f"only in B: {key}")
    for key in sorted(a.keys() & b.keys()):
        ta, tb = a[key], b[key]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            if ta != tb:
                problems.append(f"{key}: {ta!r} != {tb!r}")
            continue
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            problems.append(f"{key}: {ta.dtype}{list(ta.shape)} vs {tb.dtype}{list(tb.shape)}")
            continue
        if torch.equal(ta, tb):
            continue
        diff = (ta.double() - tb.double()).abs().max().item()
        if diff > atol:
            problems.append(f"{key}: max abs diff {diff:.3e}")
    return problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--atol", type=float, default=0.0, help="0 means bitwise equal")
    args = parser.parse_args()

    a = torch.load(args.a, map_location="cpu", weights_only=True)
    b = torch.load(args.b, map_location="cpu", weights_only=True)
    problems = compare(a, b, args.atol)
    for line in problems:
        logger.info(line)
    criterion = "bitwise" if args.atol == 0 else f"within atol={args.atol}"
    logger.info(
        f"{len(a)} vs {len(b)} entries: "
        + ("equal " + criterion if not problems else f"{len(problems)} differ ({criterion})")
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
