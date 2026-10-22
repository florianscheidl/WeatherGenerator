"""
CPU smoke test of the train and inference entry points on a tiny synthetic anemoi dataset.

The entry points run in subprocesses, since they set process-wide state (multiprocessing start
method, selected device). On an exception they print the traceback and drop into pdb, which
returns on the closed stdin, so the checks are on the outputs rather than the exit code.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
import zarr

REPO_ROOT = Path(__file__).parents[2]
BASE_CONFIG = REPO_ROOT / "config" / "era5_local.yml"

# tiny model on a coarse HEALPix grid, fp32 on the CPU
SMOKE_CONFIG = {
    "device": "cpu",
    "with_mixed_precision": False,
    "attention_dtype": "fp32",
    "healpix_level": 2,
    "ae_local_dim_embed": 32,
    "ae_local_num_heads": 2,
    "ae_local_num_blocks": 1,
    "ae_adapter_num_heads": 2,
    "ae_adapter_embed": 16,
    "ae_global_dim_embed": 32,
    "ae_global_num_heads": 2,
    "ae_global_num_blocks": 1,
    "ae_aggregation_num_heads": 2,
    "ae_aggregation_num_blocks": 1,
    "fe_num_heads": 2,
    "fe_num_blocks": 1,
    "data_loading": {"num_workers": 0},
    "training_config": {
        "samples_per_mini_epoch": 2,
        "start_date": "2020-01-01T00:00",
        "end_date": "2020-01-07T00:00",
    },
    "validation_config": {
        "samples_per_mini_epoch": 1,
        "start_date": "2020-01-08T00:00",
        "end_date": "2020-01-10T00:00",
    },
    "test_config": {
        "samples_per_mini_epoch": 1,
        "start_date": "2020-01-08T00:00",
        "end_date": "2020-01-10T00:00",
    },
}

STREAM_CONFIG = {
    "ERA5": {
        "type": "anemoi",
        "stream_id": 0,
        "filenames": ["synthetic.zarr"],
        "source": ["t_850", "z_850"],
        "target": ["t_850"],
        "masking_rate": 0.6,
        "masking_rate_none": 0.05,
        "token_size": 32,
        "tokenize_spacetime": True,
        "max_num_targets": -1,
        "embed": {
            "net": "transformer",
            "num_tokens": 1,
            "num_heads": 2,
            "dim_embed": 16,
            "num_blocks": 1,
        },
        "embed_target_coords": {"net": "linear", "dim_embed": 16},
        "target_readout": {"num_layers": 1, "num_heads": 2},
        "pred_head": {"ens_size": 1, "num_layers": 1},
    }
}


def write_synthetic_anemoi_dataset(
    path: Path, variables=("t_850", "z_850"), num_points=2000, num_dates=40, seed=0
) -> None:
    """Random fields at random locations, 6-hourly from 2020-01-01, in the anemoi zarr layout."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2020-01-01T00:00:00", "s")
    dates = start + np.arange(num_dates) * np.timedelta64(6, "h")
    data = rng.standard_normal((num_dates, len(variables), 1, num_points)).astype(np.float32)
    per_variable = data.transpose(1, 0, 2, 3).reshape(len(variables), -1).astype(np.float64)

    root = zarr.open_group(path, mode="w", zarr_format=2)
    root.create_array("data", data=data, chunks=(1, len(variables), 1, num_points))
    root.create_array("dates", data=dates)
    root.create_array("latitudes", data=np.degrees(np.arcsin(rng.uniform(-1, 1, num_points))))
    root.create_array("longitudes", data=rng.uniform(0, 360, num_points))
    for name, values in [
        ("mean", per_variable.mean(1)),
        ("stdev", per_variable.std(1)),
        ("minimum", per_variable.min(1)),
        ("maximum", per_variable.max(1)),
    ]:
        root.create_array(name, data=values)
    root.attrs.update(
        variables=list(variables),
        # channel selection needs an entry per variable, and no constant fields
        variables_metadata={v: {} for v in variables},
        constant_fields=[],
        frequency="6h",
        start_date=str(dates[0]),
        end_date=str(dates[-1]),
        resolution="synthetic",
        field_shape=[num_points],
        ensemble_dimension=1,
        flatten_grid=True,
        missing_dates=[],
        version="0.30",
    )


def _run(entry_point: str, args: list[str], private_config: Path, cwd: Path) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", f"from weathergen.run_train import {entry_point}; {entry_point}()"]
        + args,
        env={**os.environ, "WEATHERGEN_PRIVATE_CONF": str(private_config)},
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = proc.stdout + proc.stderr
    assert "Traceback" not in output, output[-5000:]
    assert proc.returncode == 0, output[-5000:]
    return output


def test_train_and_inference_on_cpu(tmp_path):
    data_dir, streams_dir, work_dir = (tmp_path / d for d in ("data", "streams", "work"))
    for d in (data_dir, streams_dir, work_dir):
        d.mkdir()
    write_synthetic_anemoi_dataset(data_dir / "synthetic.zarr")
    (streams_dir / "era5.yml").write_text(yaml.safe_dump(STREAM_CONFIG))
    private_config = tmp_path / "private_config.yml"
    private_config.write_text(
        yaml.safe_dump(
            {"path_shared_working_dir": str(work_dir), "data_path_anemoi": str(data_dir)}
        )
    )
    smoke_config = tmp_path / "smoke_config.yml"
    smoke_config.write_text(yaml.safe_dump({**SMOKE_CONFIG, "streams_directory": str(streams_dir)}))

    run_id = "smoketrn"
    output = _run(
        "train",
        ["--base-config", str(BASE_CONFIG), "--config", str(smoke_config), "--run-id", run_id],
        private_config,
        tmp_path,
    )
    assert "Loss is 0.0" not in output
    checkpoint = work_dir / "models" / run_id / f"{run_id}_chkpt00001.chkpt"
    assert checkpoint.exists(), output[-5000:]

    inference_run_id = "smokeinf"
    output = _run(
        "inference",
        [
            "--from-run-id",
            run_id,
            "--mini-epoch",
            "1",
            "--config",
            str(smoke_config),
            "--run-id",
            inference_run_id,
        ],
        private_config,
        tmp_path,
    )
    assert "Finished inference run" in output, output[-5000:]
    assert list((work_dir / "results" / inference_run_id).glob("validation_*")), output[-5000:]
