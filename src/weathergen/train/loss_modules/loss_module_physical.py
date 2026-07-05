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


def _host_to_device_async(t: torch.Tensor, device) -> torch.Tensor:
    """
    Move a small host tensor to the device without stalling the compute stream.

    A pageable host-to-device copy synchronizes the stream; pinning first makes the
    copy truly asynchronous. On non-CUDA devices this is a plain copy.
    """
    if torch.device(device).type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def get_num_samples(config) -> np.typing.NDArray:
    """
    Get number of samples in source/target config
    """
    return np.array([s_cfg.get("num_samples", 1) for _, s_cfg in config.items()])


class DynamicLossEMA:
    """
    Tracks and applies dynamic channel weights using an Exponential Moving Average (EMA)
    of inverse MSE, as described in Samudra 2.
    """

    def __init__(self, cfg: dict | None, streams_cfg: dict, device: str):
        self.enabled = cfg is not None
        if self.enabled:
            self.window = cfg.get("window", 100)
            self.L = cfg.get("L", 20.0)
            self.channel_weights_ema = {}
            for stream_name, stream_info in streams_cfg.items():
                num_channels = len(stream_info.train_target_channels)
                self.channel_weights_ema[stream_name] = torch.ones(num_channels, device=device)

    def get_weights(
        self, stream_name: str, weights_channels_static: torch.Tensor | None
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None

        ema = self.channel_weights_ema[stream_name]
        if ema.numel() > 0:
            l_min = ema.min().clamp(min=1e-6)
            # Clamp max weight to L * min weight as per Samudra 2 paper
            clamped_ema = ema.clamp(max=self.L * l_min)
            # Normalize so mean is 1.0 to preserve overall learning rate scale
            weights_channels = clamped_ema / clamped_ema.mean()
        else:
            weights_channels = ema.clone()

        if weights_channels_static is not None and weights_channels_static.numel() > 0:
            weights_channels = weights_channels * weights_channels_static

        return weights_channels

    def update(self, stream_name: str, loss_lfct_chs: torch.Tensor):
        if not self.enabled:
            return

        with torch.no_grad():
            mse_per_chan = loss_lfct_chs.detach().clamp(min=1e-6)
            inv_mse = 1.0 / mse_per_chan
            self.channel_weights_ema[stream_name] = (
                1.0 - 1.0 / self.window
            ) * self.channel_weights_ema[stream_name] + (1.0 / self.window) * inv_mse


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

        # Dynamic Loss state (extract it before parsing the actual loss functions)
        self.dynamic_loss_cfg = loss_fcts.get("dynamic_loss")
        self.forecast_offset = self.mode_cfg.forecast.offset

        # dynamically load loss functions based on configuration and stage
        self.loss_fcts = [
            [
                getattr(loss_fns, name),
                params.get("weight", 1.0),
                name,
            ]
            for name, params in loss_fcts.items()
            if name != "dynamic_loss"
        ]

        self.dynamic_loss_ema = DynamicLossEMA(
            self.dynamic_loss_cfg if self.stage == TRAIN else None,
            self.cf.streams,
            self.device,
        )

        # per-stream static channel weights are constant; cache the device tensor so it is
        # not rebuilt every step (torch.tensor(list).to(device) is a synchronizing
        # pageable host-to-device copy)
        self._weights_channels_static_cache = {}
        # counts compute_loss invocations; used to limit the loss==0 sanity check (a host
        # read of the device loss) to the first steps
        self._num_compute_loss_calls = 0

    def _get_weights(self, stream_name, stream_info):
        """
        Get weights for current stream
        """

        device = self.device

        # Determine stream and channel loss weights based on the current stage
        if self.stage == TRAIN:
            # set loss_weights to 1. when not specified
            stream_info_loss_weight = stream_info.get("loss_weight", 1.0)
            if stream_info.get("target_channel_weights"):
                weights_channels_static = self._weights_channels_static_cache.get(stream_name)
                if weights_channels_static is None:
                    weights_channels_static = _host_to_device_async(
                        torch.tensor(stream_info["target_channel_weights"]), device
                    )
                    self._weights_channels_static_cache[stream_name] = weights_channels_static
            else:
                weights_channels_static = None
        elif self.stage == VAL:
            # in validation mode, always unweighted loss
            stream_info_loss_weight = 1.0
            weights_channels_static = None

        if self.dynamic_loss_ema.enabled:
            weights_channels = self.dynamic_loss_ema.get_weights(
                stream_name, weights_channels_static
            )
        else:
            weights_channels = (
                weights_channels_static
                if weights_channels_static is None or weights_channels_static.numel() > 0
                else None
            )

        return stream_info_loss_weight, weights_channels

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
            weights_locations_fct(
                target_coords if idxs is None else target_coords.index_select(0, idxs)
            )
            for idxs in substep_idxs
        ]

        return weights_locations

    def _get_substep_masks(self, stream_info, output_step, target_times):
        """
        Find substeps and create corresponding index tensors (reused across loss functions)

        Returns a list with one entry per substep: None (= all points, the common case
        without tokenize_spacetime) or a device index tensor. The indices are computed on
        the host from the numpy times and moved pinned + non-blocking; device-side boolean
        masks would force a host-device sync per use (data-dependent output shape of
        boolean indexing).
        """

        tok_spacetime = stream_info.get("tokenize_spacetime", None)
        if not tok_spacetime:
            return [None]

        substep_idxs = []
        for t in np.unique(target_times):
            idxs_t = torch.from_numpy(np.flatnonzero(t == target_times))
            substep_idxs.append(_host_to_device_async(idxs_t, self.device))

        return substep_idxs

    @staticmethod
    def _loss_per_loss_function(
        loss_fct,
        target: torch.Tensor,
        pred: torch.Tensor,
        substep_masks: list[torch.Tensor],
        weights_channels: torch.Tensor,
        weights_locations: list[torch.Tensor],
    ):
        """
        Compute loss for given loss function
        """

        # accumulators live on the device; scalar torch.tensor(0.0, device=...) would be a
        # synchronizing pageable host-to-device copy, and `if loss > 0.0` a device read
        loss_lfct = torch.zeros((), device=target.device, requires_grad=True)
        losses_chs = torch.zeros(target.shape[-1], device=target.device, dtype=torch.float32)

        ctr_substeps = torch.zeros((), device=target.device)
        for i_t, idxs_t in enumerate(substep_masks):
            if weights_locations[i_t] is not None:
                num_points = target.shape[0] if idxs_t is None else idxs_t.shape[0]
                assert num_points == len(weights_locations[i_t])

            target_t = target if idxs_t is None else target.index_select(0, idxs_t)
            pred_t = pred if idxs_t is None else pred.index_select(1, idxs_t)
            loss, loss_chs = loss_fct(target_t, pred_t, weights_channels, weights_locations[i_t])

            # accumulate loss
            loss_lfct = loss_lfct + loss
            losses_chs = losses_chs + loss_chs.detach() if len(loss_chs) > 0 else losses_chs
            ctr_substeps = ctr_substeps + (loss > 0.0)

        # normalize over forecast steps in window
        denom = ctr_substeps.clamp(min=1.0)
        losses_chs = losses_chs / denom

        # TODO: substep weight
        loss_lfct = loss_lfct / denom

        return loss_lfct, losses_chs

    def compute_loss(self, preds: dict, targets: dict, metadata) -> LossValues:
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

        self._num_compute_loss_calls += 1

        # gradient loss; all scalar accumulators and counters are created and kept on the
        # device: torch.tensor(0.0, device=...) is a synchronizing pageable host-to-device
        # copy and host-side counters would require reading device values back per branch
        loss = torch.zeros((), device=self.device, requires_grad=True)
        # counter for non-empty targets
        ctr_streams = torch.zeros((), device=self.device)

        # initialize dictionaries for detailed loss tracking and standard deviation statistics
        # create tensor for each stream
        losses_all = defaultdict(dict)

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

            stream_loss_weight, weights_channels = self._get_weights(stream_name, stream_info)
            if self.dynamic_loss_ema.enabled and weights_channels is not None:
                losses_all[stream_name][str(self.forecast_offset)]["mse_ema_weight"] = {}
                # single batched device read instead of one w.item() sync per channel
                for ch_n, w in zip(target_channels, weights_channels.tolist(), strict=True):
                    losses_all[stream_name][str(self.forecast_offset)]["mse_ema_weight"][ch_n] = w

            # TODO: make nicer
            output_step_loss_weights = self._get_output_step_weights(len(targets.output_idxs))
            if len(targets.physical) - len(targets.output_idxs) > 0:
                output_step_loss_weights.insert(0, None)

            # loss_stream: loss for given stream
            loss_stream = torch.zeros((), device=self.device, requires_grad=True)
            ctr_timesteps = torch.zeros((), device=self.device)
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
                ctr_batch = torch.zeros((), device=self.device)
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

                    # get masks for sub-time steps
                    substep_masks = self._get_substep_masks(stream_info, timestep_idx, target_times)

                    # get weights for locations
                    weights_locations = self._get_location_weights(
                        stream_info, targets_coords_batch[target_idx], substep_masks
                    )

                    # loss_st_corr: loss for give source-target correspondence
                    loss_st_corr = torch.zeros((), device=self.device, requires_grad=True)
                    ctr_loss_fcts = torch.zeros((), device=self.device)
                    for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:
                        # skip is loss is not computed for this sample
                        if loss_fct_name not in pred_params.global_params["loss"]:
                            continue

                        # spoofed inputs are masked in the output calculations; a plain
                        # python float multiplies device tensors without any host copy
                        is_spoof = targets_is_spoof[target_idx]
                        spoof_weight = 0.0 if is_spoof else 1.0

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
                        # loss_lfct: loss for given loss function aggregated over all channels
                        # loss_lfct_chs: loss for given loss function per channel
                        loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                            loss_fct,
                            target,
                            pred,
                            substep_masks,
                            weights_channels,
                            weights_locations,
                        )

                        # mark channels without contribution as nan on the device; the
                        # per-channel `v != 0.0` branch would sync once per channel
                        loss_lfct_chs_masked = torch.where(
                            loss_lfct_chs != 0.0,
                            loss_lfct_chs,
                            torch.full_like(loss_lfct_chs, torch.nan),
                        )
                        for ch_n, v in zip(target_channels, loss_lfct_chs_masked, strict=True):
                            losses_all[stream_name][str(timestep_idx)][loss_fct_name][ch_n] = (
                                torch.nan if is_spoof else v
                            )

                        # Update EMA for dynamic loss if enabled
                        if (
                            self.dynamic_loss_ema.enabled
                            and timestep_idx == self.forecast_offset
                            and loss_fct_name == "mse"
                            and not is_spoof
                        ):
                            self.dynamic_loss_ema.update(stream_name, loss_lfct_chs)

                        # Add the weighted and normalized loss from this loss function to the total
                        # batch loss
                        loss_cur_w = spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
                        loss_st_corr = loss_st_corr + loss_cur_w
                        # counters stay on device: `if loss_cur_w > 0.0` would be a sync
                        if not is_spoof:
                            ctr_loss_fcts = ctr_loss_fcts + (loss_cur_w > 0.0)

                    loss_timestep = loss_timestep + loss_st_corr
                    ctr_batch = ctr_batch + (ctr_loss_fcts > 0.0)

                loss_stream = loss_stream + loss_timestep
                ctr_timesteps = ctr_timesteps + (ctr_batch > 0)

            denom = ctr_timesteps.clamp(min=1.0)
            loss = loss + (stream_loss_weight * loss_stream) / denom

            ctr_streams = ctr_streams + (ctr_timesteps > 0)

        # normalize by all targets and forecast steps that were non-empty
        # (with each having an expected loss of 1 for an uninitalized neural net)
        # misconfiguration sanity check: reading the device loss is a sync, so only check
        # during the first steps where a misconfiguration would already show
        if self._num_compute_loss_calls <= 3 and bool(loss == 0.0):
            _logger.warning(
                "Loss is 0.0, likely incorrect configuration. Check stream"
                " support time and training configuration."
            )
        loss = loss / ctr_streams.clamp(min=1.0)

        def _nested_dict():
            return defaultdict(dict)

        # Reorder losses_all to [stream_name][loss_fct_name][ch_n][output_step]
        reordered_losses = defaultdict(dict)
        for stream_name, output_step_dict in losses_all.items():
            reordered_losses[stream_name] = defaultdict(_nested_dict)
            for output_step, lfct_dict in output_step_dict.items():
                for loss_fct_name, ch_dict in lfct_dict.items():
                    for ch_n, v in ch_dict.items():
                        reordered_losses[stream_name][loss_fct_name][ch_n][output_step] = v

        # Calculate per stream, per lfct average across channels and output_steps;
        # values are a mix of python floats (nan markers) and 0-dim device tensors (which
        # may hold nan for zero-contribution channels) — nans count as 0 contribution.
        # All tensor arithmetic stays on device; no host reads.
        for stream_name, lfct_dict in reordered_losses.items():
            for loss_fct_name, ch_dict in lfct_dict.items():
                total = 0.0
                count = 0
                for ch_n, output_step_dict in ch_dict.items():
                    if ch_n != "avg":
                        for _, v in output_step_dict.items():
                            if torch.is_tensor(v):
                                total = total + v.nan_to_num(0.0)
                            elif not (type(v) is float and np.isnan(v)):
                                total = total + v
                            count += 1
                reordered_losses[stream_name][loss_fct_name]["avg"] = total / count

        # Return all computed loss components encapsulated in a ModelLoss dataclass
        return LossValues(loss=loss, losses_all=reordered_losses, stddev_all=None)
