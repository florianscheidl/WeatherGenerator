# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import types

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from weathergen.common.io import ZarrWriteConfig
from weathergen.model.engines import LatentState
from weathergen.model.model import ModelOutput
from weathergen.utils import validation_io as vio


def _latent_only_inputs(skip_unwritten_decoders: bool):
    """A one-sample chunk (forecast step 5) with a latent prediction and no physical one."""
    stream = types.SimpleNamespace(sample_idx=0, source_raw=[None])
    sample = types.SimpleNamespace(streams_data={"ERA5": stream})
    source_samples = types.SimpleNamespace(get_samples=lambda: [sample])
    batch = types.SimpleNamespace(
        get_source_samples=lambda: source_samples, get_output_idxs=lambda: list(range(1, 41))
    )
    model_output = ModelOutput([5], 1, None)
    tokens = torch.arange(12 * 4, dtype=torch.float32).reshape(1, 12, 4)
    model_output.add_latent_prediction(
        model_output.chunk_idx(5), "latent_state", LatentState(None, None, None, tokens)
    )
    cf = OmegaConf.create(
        {"streams": {"ERA5": {"val_target_channels": ["t"], "val_source_channels": ["t"]}}}
    )
    val_cfg = OmegaConf.create(
        {
            "losses": {"physical": {"type": "LossPhysical"}},
            "with_targets": False,
            "output": {"streams": ["latent"], "skip_unwritten_decoders": skip_unwritten_decoders},
            "start_date": np.datetime64("2023-06-01T06:00"),
            "end_date": np.datetime64("2023-07-12T22:00"),
            "time_window_len": np.timedelta64(6, "h"),
            "time_window_step": np.timedelta64(6, "h"),
        },
        flags={"allow_objects": True},
    )
    return cf, val_cfg, batch, model_output, tokens


def test_latents_are_written_without_physical_predictions(tmp_path):
    cf, val_cfg, batch, model_output, tokens = _latent_only_inputs(skip_unwritten_decoders=True)

    staged = vio.stage_output(
        cf,
        val_cfg,
        1,
        0,
        0,
        lambda sname, x: x,
        batch,
        model_output,
        {"physical": None},
        store_path=tmp_path / "out.zarr",
    )

    assert staged is not None
    assert staged.preds == [[]] and staged.batch_data_args["streams"] == {}
    staged.write_config = ZarrWriteConfig()
    staged.write()
    root = zarr.open(str(tmp_path / "out.zarr"), mode="r")
    np.testing.assert_array_equal(root["0/latent/5/tokens/data"][:], tokens[0].numpy())


def test_without_the_option_an_output_without_physical_predictions_is_not_written():
    cf, val_cfg, batch, model_output, _ = _latent_only_inputs(skip_unwritten_decoders=False)

    # unchanged behaviour: the physical predictions are collected, and an output without any is
    # skipped (with the decoders running, as without the option, they are never empty)
    staged = vio.stage_output(
        cf, val_cfg, 1, 0, 0, lambda sname, x: x, batch, model_output, {"physical": None}
    )

    assert staged is None
