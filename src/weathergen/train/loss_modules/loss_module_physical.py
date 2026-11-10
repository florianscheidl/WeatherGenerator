# pylint: disable=bad-builtin
# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig

import weathergen.train.loss_modules.loss_functions as loss_fns
from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.train.utils import TRAIN, VAL, Stage

_logger = logging.getLogger(__name__)

# Device types where pinned host memory makes host-to-device copies asynchronous. Elsewhere
# (CPU, MPS) pinning is pointless, and a non_blocking copy to MPS can return wrong data.
_PIN_MEMORY_DEVICE_TYPES = ("cuda", "xpu")


def _host_to_device_async(t: torch.Tensor, device) -> torch.Tensor:
    """
    Move a small host tensor to the device without stalling the host: a pageable copy waits for
    the stream, a non_blocking copy from pinned memory does not. Plain copy on other devices.
    """
    if torch.device(device).type in _PIN_MEMORY_DEVICE_TYPES:
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _div_by_count(x: torch.Tensor, count) -> torch.Tensor:
    """
    x / count if count > 0 else x, without reading count on the host when it is a device tensor
    (x / 1 == x, so clamping the count to 1 gives the same values).

    The result is bitwise equal to dividing by count as a Python number. On CUDA, ATen divides by
    a host scalar as a multiplication with its reciprocal (computed in the op math type), so
    the same is done here; on other devices it is a true division.
    """
    if not isinstance(count, torch.Tensor):
        return x / count if count > 0 else x
    c = count.clamp(min=1).to(torch.float32)
    xf = x.to(torch.float32)
    y = xf * torch.reciprocal(c) if x.device.type == "cuda" else xf / c
    return y.to(x.dtype)


def get_num_samples(config) -> np.typing.NDArray:
    """
    Get number of samples in source/target config
    """
    return np.array([s_cfg.get("num_samples", 1) for _, s_cfg in config.items()])


class LossPhysical(LossModuleBase):
    """
    Manages and computes the overall loss for a WeatherGenerator model during
    training and validation stages.

    This class handles the initialization and application of various loss functions,
    applies channel-specific weights, constructs masks for missing data, and
    aggregates losses across different data streams, channels, and forecast steps.
    It provides both the main loss for backpropagation and detailed loss metrics for logging.
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts,
    ):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.mode_cfg = mode_cfg
        self.stage = stage
        self.device = device
        self.name = "LossPhysical"

        # dynamically load loss functions based on configuration and stage
        self.loss_fcts = [
            [
                getattr(loss_fns, name),
                params.get("weight", 1.0),
                name,
            ]
            for name, params in loss_fcts.items()
        ]

        # channel weights per stream on the device, copied once (they come from the config)
        self._weights_channels = {}
        # spoof weights (0.0, 1.0) on the device, created once
        self._spoof_weights = {}
        # the zero-loss configuration check reads the loss on the host: first step only
        self._checked_zero_loss = False

    def _get_weights(self, stream_name, stream_info):
        """
        Get weights for current stream
        """

        # Determine stream and channel loss weights based on the current stage
        if self.stage == TRAIN:
            # set loss_weights to 1. when not specified
            stream_info_loss_weight = stream_info.get("loss_weight", 1.0)
            weights_channels = None
            if "target_channel_weights" in stream_info:
                if stream_name not in self._weights_channels:
                    self._weights_channels[stream_name] = _host_to_device_async(
                        torch.tensor(stream_info["target_channel_weights"]), self.device
                    )
                weights_channels = self._weights_channels[stream_name]
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels = None

        return stream_info_loss_weight, weights_channels

    def _get_spoof_weight(self, sw: float) -> torch.Tensor:
        """
        Spoof weight sw as a 0-dim device tensor, created on the device once per value (no
        gradient, never modified, so it is shared across calls).
        """
        if sw not in self._spoof_weights:
            self._spoof_weights[sw] = torch.full((), sw, device=self.device)
        return self._spoof_weights[sw]

    def _get_output_step_weights(self, len_forecast_steps):
        timestep_weight_config = self.mode_cfg.get("forecast", {}).get("timestep_weight", {})
        if len(timestep_weight_config) == 0:
            return [1.0 for _ in range(len_forecast_steps)]
        weights_timestep_fct = getattr(loss_fns, list(timestep_weight_config.keys())[0])
        decay_factor = list(timestep_weight_config.values())[0]["decay_factor"]
        return weights_timestep_fct(len_forecast_steps, decay_factor)

    def _get_location_weights(self, stream_info, target_coords, substep_idxs):
        location_weight_type = stream_info.get("location_weight", None)
        if location_weight_type is None:
            return [None for _ in substep_idxs]

        target_coords = target_coords.to(self.device, non_blocking=True)
        weights_locations_fct = getattr(loss_fns, location_weight_type)
        weights_locations = [
            weights_locations_fct(target_coords.index_select(0, idxs)) for idxs in substep_idxs
        ]

        return weights_locations

    def _get_substep_idxs(self, stream_info, output_step, target_times):
        """
        Find substeps and the indices of their target points (reused across loss functions).

        The indices are computed on the host from target_times (a host array) and selected with
        index_select: a boolean mask on the device would make the output size depend on device
        data, which reads the mask on the host (in forward and backward). The points and their
        order are the same as with the mask t == target_times.
        """

        tok_spacetime = stream_info.get("tokenize_spacetime", None)
        target_times_unique = np.unique(target_times) if tok_spacetime else [target_times]
        substep_idxs = []
        for t in target_times_unique:
            # find substep
            idxs_t = torch.from_numpy(np.flatnonzero(t == target_times))
            substep_idxs.append(_host_to_device_async(idxs_t, self.device))

        return substep_idxs

    @staticmethod
    def _loss_per_loss_function(
        loss_fct,
        target: torch.Tensor,
        pred: torch.Tensor,
        substep_idxs: list[torch.Tensor],
        weights_channels: torch.Tensor,
        weights_locations: list[torch.Tensor],
    ):
        """
        Compute loss for given loss function
        """

        loss_lfct = torch.zeros((), device=target.device, requires_grad=True)
        losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=torch.float32)

        # number of substeps with a positive loss, counted on the device
        ctr_substeps = 0
        for i_t, idxs_t in enumerate(substep_idxs):
            assert (
                len(idxs_t) == len(weights_locations[i_t])
                if weights_locations[i_t] is not None
                else True
            )

            loss, loss_chs = loss_fct(
                target.index_select(0, idxs_t),
                pred.index_select(1, idxs_t),
                weights_channels,
                weights_locations[i_t],
            )

            # accumulate loss
            loss_lfct = loss_lfct + loss
            losses_chs = losses_chs + loss_chs.detach() if len(loss_chs) > 0 else losses_chs
            ctr_substeps = ctr_substeps + (loss.detach() > 0.0)

        # normalize over forecast steps in window
        losses_chs = _div_by_count(losses_chs, ctr_substeps)

        # TODO: substep weight
        loss_lfct = _div_by_count(loss_lfct, ctr_substeps)

        return loss_lfct, losses_chs

    def compute_loss(self, preds: dict, targets: dict, metadata, **kwargs) -> LossValues:
        """
        Computes the total loss for a given batch of predictions and corresponding
        stream data.

        The computed loss is:

        Mean_{stream}( Mean_{output_steps}( Mean_{loss_fcts}( loss_fct( target, pred, weigths) )))

        This method orchestrates the calculation of the overall loss by iterating through
        different data streams, forecast steps, channels, and configured loss functions.
        It applies weighting, handles NaN values through masking, and accumulates
        detailed loss metrics for logging.

        Args:
            preds: A nested list of prediction tensors. The outer list represents forecast steps,
                   the inner list represents streams. Each tensor contains predictions for that
                   step and stream.
            streams_data: A nested list representing the input batch data. The outer list is for
                          batch items, the inner list for streams. Each element provides an object
                          (e.g., dataclass instance) containing target data and metadata.

        Returns:
            A ModelLoss dataclass instance containing:
            - loss: The loss for back-propagation.
            - losses_all: A dictionary mapping stream names to a tensor of per-channel and
                          per-loss-function losses, normalized by non-empty targets/forecast steps.
            - stddev_all: A dictionary mapping stream names to a tensor of mean standard deviations
                          of predictions for channels with statistical loss functions, normalized.
        """

        # The accumulators are created on the device (torch.tensor(0.0, device=...) would copy
        # from the host and wait for the stream) and the counters below are device tensors once
        # they depend on a loss value, so that no loss value is read on the host.

        # gradient loss
        loss = torch.zeros((), device=self.device, requires_grad=True)
        # counter for non-empty targets
        ctr_streams = 0

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all = defaultdict(dict)
        # per entry of losses_all, its term in the per stream and loss function average: the
        # value, or 0.0 where losses_all holds NaN for a zero or spoofed loss
        avg_terms = defaultdict(dict)

        source2target_idxs, output_info, target2source_idxs, target_info = metadata

        # TODO: iterate over batch dimension
        for stream_name, stream_info in self.cf.streams.items():
            # TODO: avoid this
            target_channels = (
                stream_info.val_target_channels
                if self.stage == "val"
                else stream_info.train_target_channels
            )

            losses_all[stream_name] = defaultdict(dict)
            avg_terms[stream_name] = defaultdict(dict)

            stream_loss_weight, weights_channels = self._get_weights(stream_name, stream_info)

            # TODO: make nicer
            output_step_loss_weights = self._get_output_step_weights(len(targets.output_idxs))
            if len(targets.physical) - len(targets.output_idxs) > 0:
                output_step_loss_weights.insert(0, None)

            # loss_stream: loss for given stream
            loss_stream = torch.zeros((), device=self.device, requires_grad=True)
            ctr_timesteps = 0
            for timestep_idx, (preds_cur, target_cur) in enumerate(
                zip(preds.physical, targets.physical, strict=True)
            ):
                preds_batch = preds_cur.get(stream_name, [])
                if not preds_batch:
                    # skip to next timestep if preds of current timestep are empty
                    continue

                targets_batch = target_cur[stream_name]["target"]
                targets_coords_batch = target_cur[stream_name]["target_coords"]
                targets_times_batch = target_cur[stream_name]["target_times"]
                targets_params = target_cur[stream_name]["target_metda_data"]
                targets_is_spoof = target_cur[stream_name]["is_spoof"]

                output_step_weight = output_step_loss_weights[timestep_idx]

                # loss_timestep: loss for given timestep
                loss_timestep = torch.zeros((), device=self.device, requires_grad=True)
                ctr_batch = 0
                for pred, pred_params in zip(preds_batch, output_info, strict=True):
                    # source has a unique target but index is not invariant with multiple
                    # target_aux calculators
                    target_idx_native = pred_params.global_params.get("correspondence", -1)
                    target_idx = [
                        i
                        for i, t in enumerate(targets_params)
                        if t[stream_name].global_params["idx"] == target_idx_native
                    ]
                    # source/model_input has no target for physical loss
                    if len(target_idx) == 0:
                        continue
                    # source -> target correspondence has to be unique
                    assert len(target_idx) == 1
                    target_idx = target_idx[0]

                    # current target data
                    target = targets_batch[target_idx]
                    target_times = targets_times_batch[target_idx]

                    # get indices of the target points of the sub-time steps
                    substep_idxs = self._get_substep_idxs(stream_info, timestep_idx, target_times)

                    # get weights for locations
                    weights_locations = self._get_location_weights(
                        stream_info, targets_coords_batch[target_idx], substep_idxs
                    )

                    # loss_st_corr: loss for give source-target correspondence
                    loss_st_corr = torch.zeros((), device=self.device, requires_grad=True)
                    ctr_loss_fcts = 0
                    for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                        # skip is loss is not computed for this sample
                        if loss_fct_name not in pred_params.global_params["loss"]:
                            continue

                        # spoofed inputs are masked in the output calculations
                        is_spoof = targets_is_spoof[target_idx]
                        sw = 0.0 if is_spoof else 1.0
                        spoof_weight = self._get_spoof_weight(sw)

                        # skip if either target or prediction has no data points
                        if not (target.shape[0] > 0 and pred.shape[0] > 0):
                            continue

                        # reshape prediction tensor to match target's dimensions: extract
                        # data/coords and remove token dimension if it exists.
                        # expected shape of pred is [ensemble_size, num_samples, num_channels].
                        pred = pred.reshape([pred.shape[0], *target.shape])
                        assert pred.shape[1] > 0

                        losses_all[stream_name][str(timestep_idx)][loss_fct_name] = defaultdict(
                            dict
                        )
                        avg_terms[stream_name][str(timestep_idx)][loss_fct_name] = {}
                        # loss_lfct: loss for given loss function aggregated over all channels
                        # loss_lfct_chs: loss for given loss function per channel
                        loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                            loss_fct,
                            target,
                            pred,
                            substep_idxs,
                            weights_channels,
                            weights_locations,
                        )

                        # per channel: NaN where the loss is zero or the target spoofed, the
                        # weighted loss otherwise (selected on the device, not with v != 0.0)
                        ch_vals = ch_terms = None
                        if not is_spoof:
                            ch_valid = loss_lfct_chs != 0.0
                            ch_weighted = spoof_weight * loss_lfct_chs
                            ch_vals = torch.where(ch_valid, ch_weighted, torch.nan)
                            ch_terms = torch.where(ch_valid, ch_weighted, 0.0)
                        lfct_vals = losses_all[stream_name][str(timestep_idx)][loss_fct_name]
                        lfct_terms = avg_terms[stream_name][str(timestep_idx)][loss_fct_name]
                        assert len(target_channels) == len(loss_lfct_chs)
                        for i_ch, ch_n in enumerate(target_channels):
                            lfct_vals[ch_n] = torch.nan if is_spoof else ch_vals[i_ch]
                            lfct_terms[ch_n] = 0.0 if is_spoof else ch_terms[i_ch]

                        # Add the weighted and normalized loss from this loss function to the total
                        # batch loss
                        loss_cur_w = spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
                        loss_st_corr = loss_st_corr + loss_cur_w
                        if not is_spoof:
                            ctr_loss_fcts = ctr_loss_fcts + (loss_cur_w.detach() > 0.0)

                    loss_timestep = loss_timestep + loss_st_corr
                    ctr_batch = ctr_batch + (ctr_loss_fcts > 0)

                loss_stream = loss_stream + loss_timestep
                ctr_timesteps = ctr_timesteps + (ctr_batch > 0)

            loss = loss + _div_by_count(stream_loss_weight * loss_stream, ctr_timesteps)

            ctr_streams = ctr_streams + (ctr_timesteps > 0)

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        # A zero loss is a configuration error: check it on the first step only, reading the
        # loss on the host every step would stall it.
        if not self._checked_zero_loss:
            self._checked_zero_loss = True
            if loss == 0.0:
                _logger.warning(
                    "Loss is 0.0, likely incorrect configuration. Check stream"
                    " support time and training configuration."
                )
        loss = _div_by_count(loss, ctr_streams)

        def _nested_dict():
            return defaultdict(dict)

        # Reorder losses_all to [stream_name][loss_fct_name][ch_n][output_step]
        reordered_losses = defaultdict(dict)
        reordered_terms = defaultdict(dict)
        for stream_name, output_step_dict in losses_all.items():
            reordered_losses[stream_name] = defaultdict(_nested_dict)
            reordered_terms[stream_name] = defaultdict(_nested_dict)
            for output_step, lfct_dict in output_step_dict.items():
                for loss_fct_name, ch_dict in lfct_dict.items():
                    for ch_n, v in ch_dict.items():
                        reordered_losses[stream_name][loss_fct_name][ch_n][output_step] = v
                        reordered_terms[stream_name][loss_fct_name][ch_n][output_step] = avg_terms[
                            stream_name
                        ][output_step][loss_fct_name][ch_n]

        # Calculate per stream, per lfct average across channels and output_steps
        for stream_name, lfct_dict in reordered_losses.items():
            for loss_fct_name, ch_dict in lfct_dict.items():
                terms_dict = reordered_terms[stream_name][loss_fct_name]
                reordered_losses[stream_name][loss_fct_name]["avg"] = 0
                count = 0
                for ch_n, output_step_dict in ch_dict.items():
                    if ch_n != "avg":
                        for output_step in output_step_dict:
                            # the NaN entries (zero or spoofed loss) count as 0.0
                            v = terms_dict[ch_n][output_step]
                            reordered_losses[stream_name][loss_fct_name]["avg"] += v
                            count += 1
                reordered_losses[stream_name][loss_fct_name]["avg"] /= count

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=reordered_losses, stddev_all=None)
