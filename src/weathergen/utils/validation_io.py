# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import numpy.typing as npt
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.utils.distributed import (
    all_gather_vlen,
    get_encoder_spatial_parallel_group,
)

_logger = logging.getLogger(__name__)


def _reassemble_validation_rows(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    coords: list[torch.Tensor],
    times_ns: list[torch.Tensor],
    row_ids: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, npt.NDArray[np.datetime64]]:
    """Concatenate rank-local rows and restore stable ReaderData row order."""

    pred = torch.cat(predictions, dim=1)
    target = torch.cat(targets, dim=0)
    target_coords = torch.cat(coords, dim=0)
    target_times_ns = torch.cat(times_ns, dim=0)
    stable_ids = torch.cat(row_ids, dim=0)

    if stable_ids.numel() != torch.unique(stable_ids).numel():
        raise ValueError("rank-local validation shards contain duplicate stable row ids")
    order = torch.argsort(stable_ids, stable=True)
    target_times = target_times_ns[order].cpu().numpy().astype("datetime64[ns]")
    return pred[:, order], target[order], target_coords[order], target_times


def _gather_and_reassemble_validation_rows(cf, pred, target, coords, times, row_ids):
    if row_ids is None:
        raise ValueError("rank-local validation output requires stable row ids")

    device = pred.device
    coords = coords.to(device)
    row_ids = row_ids.to(device)
    times_ns = torch.as_tensor(
        times.astype("datetime64[ns]").astype(np.int64), device=device, dtype=torch.int64
    )
    group, _ = get_encoder_spatial_parallel_group(cf)
    return _reassemble_validation_rows(
        all_gather_vlen(pred, group=group),
        all_gather_vlen(target, group=group),
        all_gather_vlen(coords, group=group),
        all_gather_vlen(times_ns, group=group),
        all_gather_vlen(row_ids, group=group),
    )


def write_output(
    cf,
    val_cfg,
    batch_size,
    mini_epoch,
    batch_idx,
    dn_data,
    batch,
    model_output,
    target_aux_out,
    stage_callback=None,
):
    """
    Interface for writing model output
    """

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for i, (loss_name, loss_term) in enumerate(val_cfg.losses.items())
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out = target_aux_out[outputs_physical[0]]

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    forecast_offset = timestep_idxs[0]
    targets_lens = []
    local_validation = bool(cf.get("spatial_local_validation", False))
    if stage_callback is not None:
        stage_callback("validation_output_reassembly_start")

    # TODO Maybe stopping at forecast_steps explained #1657
    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        targets_lens += [[]]
        for sname in cf.streams.keys():
            # handle spoof data: do not write since it might corrupt validation (spoofing invisible
            # there)
            if target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
                targets = target_aux_out.physical[t_idx][sname]["target"]
                # for-loop to make sure we have a consistent number of samples
                preds_s = [np.zeros((1, 0, t.shape[1])) for t in targets]
                targets_s = [np.zeros((0, t.shape[1])) for t in targets]
                t_coords_s = [np.zeros((0, 2)) for t in targets]
                t_times_s = [np.array([]).astype("datetime64[ns]") for t in targets]

            else:
                preds = model_output.get_physical_prediction(t_idx, sname)
                targets = target_aux_out.physical[t_idx][sname]["target"]

                preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []

                # handle forcing streams or if sample is empty
                if preds is None:
                    # preds are empty so create copy of target and add ensemble dimension
                    assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                    preds = [target.clone().unsqueeze(0) for target in targets]

                for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                    target_data = target_aux_out.physical[t_idx][sname]
                    t_coords = target_data["target_coords"][i_batch]
                    t_times = target_data["target_times"][i_batch]

                    idxs_inv = target_data["idxs_inv"][i_batch]
                    if local_validation:
                        pred, target, t_coords, t_times = _gather_and_reassemble_validation_rows(
                            cf,
                            pred,
                            target,
                            t_coords,
                            t_times,
                            target_data["row_ids"][i_batch],
                        )
                    elif idxs_inv is not None:
                        pred = pred[:, idxs_inv]
                        target = target[idxs_inv]
                        t_coords = t_coords[idxs_inv]
                        t_times = t_times[idxs_inv]

                    # denormalize data if requested and map to storage format
                    preds_s += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]
                    targets_s += [dn_data(sname, target.to(fp32)).detach().cpu().numpy()]

                    # extract original target coords and times from target data
                    t_coords_s += [t_coords.cpu().numpy()]
                    t_times_s += [t_times.astype("datetime64[ns]")]

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

    if stage_callback is not None:
        stage_callback("validation_output_reassembly_end")

    if len(preds_all) == 0 or np.array([p.shape[1] for pp in preds_all for p in pp]).sum() == 0:
        _logger.warning("Writing no data since predictions are empty.")
        return

    # collect source information
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            # TODO: support multiple input steps
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    # more prep work

    # output stream names to be written, use specified ones or all if nothing specified
    stream_names = list(cf.streams.keys())
    stream_infos = list(cf.streams.values())
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names

    output_streams = {name: stream_names.index(name) for name in output_stream_names}
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in stream_infos]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in stream_infos]

    geoinfo_channels = [[] for _ in stream_infos]  # TODO obtain channels

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * batch_size

    # write output

    start_date = val_cfg.start_date
    end_date = val_cfg.end_date

    twh = TimeWindowHandler(
        start_date,
        end_date,
        val_cfg.time_window_len,
        val_cfg.time_window_step,
    )
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

    data = io.OutputBatchData(
        sources,
        source_intervals,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        output_streams,
        target_channels,
        source_channels,
        geoinfo_channels,
        sample_start,
        forecast_offset,
    )
    with zarrio_writer(config.get_path_results(cf, mini_epoch)) as zio:
        for subset in data.items():
            zio.write_zarr(subset)
