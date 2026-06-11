# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import json
import logging
import platform
from datetime import datetime
from typing import Literal

import torch
from omegaconf import OmegaConf
from torch.profiler import record_function

import weathergen
from weathergen.common import config
from weathergen.common.config import Config, merge_configs

# Run stages
Stage = Literal["train", "val", "test"]
TRAIN: Stage = "train"
VAL: Stage = "val"
TEST: Stage = "test"

# keys to filter using enabled: True/False
cfg_keys_to_filter = ["losses", "model_input", "target_input"]


# TODO: remove this definition, it should directly using common.
get_run_id = config.get_run_id

# Constants
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

logger: logging.Logger = logging.getLogger(__name__)


def str_to_tensor(modelid):
    return torch.tensor([ord(c) for c in modelid], dtype=torch.int32)


def tensor_to_str(tensor):
    return "".join([chr(x) for x in tensor])


def json_to_dict(fname):
    with open(fname) as f:
        json_str = f.readlines()
    return json.loads("".join([s.replace("\n", "") for s in json_str]))


def flatten_dict(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary, keeping lists of scalar values intact.

    :param d: The dictionary to flatten.
    :param parent_key: The base key for recursion (used internally).
    :param sep: The separator to join keys.
    :return: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        # Construct the new key
        new_key = parent_key + sep + k if parent_key else k

        # 1. Handle Dictionaries (Recursion)
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())

        # 2. Handle Lists
        elif isinstance(v, list):
            # Check if the list contains non-scalar/non-empty values (i.e., nested dicts/lists)
            # A value is considered a scalar if it's NOT a dict or a list.
            is_scalar_list = all(not isinstance(item, (dict | list)) for item in v)

            if is_scalar_list:
                # Requirement: Keep lists of scalar values as is
                items.append((new_key, v))
            else:
                # If the list contains nested dicts/lists, we must iterate and flatten them
                for i, item in enumerate(v):
                    index_key = new_key + sep + str(i)
                    if isinstance(item, dict):
                        # Recursively flatten the dictionary inside the list
                        items.extend(flatten_dict(item, index_key, sep=sep).items())
                    elif isinstance(item, list):
                        # Treat list within a list as a scalar list *at that level*
                        # and append it (to avoid overly complex list indexing)
                        items.append((index_key, item))
                    else:
                        # Append the scalar item
                        items.append((index_key, item))

        # 3. Handle Scalar Values
        else:
            # Append all other scalar values (str, int, float, bool, None, etc.)
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d, separator="."):
    """
    Unflattens a dictionary where nested keys were joined by a separator.

    :param d: The flattened dictionary.
    :param separator: The delimiter used to join nested keys.
    :return: The unflattened dictionary.
    """
    unflattened = {}
    for key, value in d.items():
        # Split the key into its components
        parts = key.split(separator)

        # Start at the root of the unflattened dictionary
        current_level = unflattened

        # Iterate over all parts of the key except the last one
        for part in parts[:-1]:
            # If the part is not a key in the current level, create a new dictionary
            if part not in current_level:
                current_level[part] = {}

            # Move down to the next level
            current_level = current_level[part]

        # Set the value for the final, innermost key
        current_level[parts[-1]] = value

    return unflattened


def extract_batch_metadata(batch):
    return (
        batch.source2target_matching_idxs,
        [list(sample.meta_info.values())[0] for sample in batch.source_samples.get_samples()],
        batch.target2source_matching_idxs,
        [list(sample.meta_info.values())[0] for sample in batch.target_samples.get_samples()],
    )


def get_batch_size_from_config(config: Config) -> int:
    """
    Determine batch size from training/validation/test config by parsing num_samples
    """

    num_samples = 0
    for _, source_cfg in config.model_input.items():
        if source_cfg.get("enabled", True):
            num_samples += source_cfg.get("num_samples", 1)
    assert num_samples > 0, "Number of samples in source configs needs to greater than 0."

    return num_samples


def get_target_idxs_from_cfg(cfg, loss_name) -> list[int] | None:
    """
    Extract target idxs from training/validation/test config
    """

    tc = [v.get("target_source_correspondence") for _, v in cfg.losses[loss_name].loss_fcts.items()]
    tc = [list(t.keys()) for t in tc if t is not None]
    target_idxs = list(set([int(i) for t in tc for i in t])) if len(tc) > 0 else None

    return target_idxs


def get_active_stage_config(
    base_config: dict | OmegaConf, merge_config: dict | OmegaConf, keys_to_filter: list[str]
) -> dict | OmegaConf:
    """
    Combine a stage config with its predecessor and filter by enabled: False to obtain the
    final config that is used
    """

    result_cfg = merge_configs(base_config, merge_config)
    result_cfg = filter_config_by_enabled(result_cfg, keys_to_filter)

    return result_cfg


def filter_config_by_enabled(cfg: dict | OmegaConf, keys: list[str]):
    """
    Filtered disabled entries from config
    """

    cfg_out = copy.deepcopy(cfg)

    for key in keys:
        filtered = {}
        for k, v in cfg_out.get(key, {}).items():
            if v.get("enabled", True):
                filtered[k] = v
        cfg_out[key] = filtered

    return cfg_out

def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

def stop_record_memory_history() -> None:
    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(cfg: dict | OmegaConf) -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    base_path = config.get_path_profiler(cfg)

    timestamp = datetime.now().strftime(TIME_FORMAT_STR)

    file_prefix = base_path / f"{timestamp}_rank_{weathergen.utils.distributed.get_rank()}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def trace_handler(cfg: dict | OmegaConf, prof: torch.profiler.profile) -> None:
    # Prefix for file names.
    base_path = config.get_path_profiler(cfg)

    timestamp = datetime.now().strftime(TIME_FORMAT_STR)

    file_prefix = base_path / f"{timestamp}_rank_{weathergen.utils.distributed.get_rank()}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    on_aarch64 = platform.machine() == "aarch64"
    if not on_aarch64:
        prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    else:
        logger.info("[profiler] Memory distribution timeline skipped on aarch64")


class _ProfiledModule(torch.nn.Module):
    """
    Wraps a module with a torch.profiler.record_function context without replacing
    its forward method. This is critical for FSDP2 compatibility: FSDP2 installs a
    DTensor-propagation shim on nn.Module.forward; monkey-patching module.forward
    (as the previous implementation did) replaces that shim with a plain function,
    causing "got mixed torch.Tensor and DTensor" errors in sharded Linear layers.

    With this wrapper, FSDP2 dispatches DTensors through _ProfiledModule.forward,
    and the wrapper passes them through to the inner module unchanged.
    """

    _is_profiled_module = True  # marker to skip during recursion

    def __init__(self, module: torch.nn.Module, profile_name: str):
        super().__init__()
        self._inner = module
        self._profile_name = profile_name

    def forward(self, *args, **kwargs):
        with record_function(self._profile_name):
            return self._inner(*args, **kwargs)


def wrap_module_forward_with_profiling(model, prefix=""):
    """
    Recursively wrap all custom nn.Module forward methods with profiling context.
    Uses _ProfiledModule wrappers instead of monkey-patching .forward so that
    FSDP2's DTensor dispatch shim (installed via fully_shard) is preserved.
    """
    for name, module in model.named_children():
        module_name = f"{prefix}.{name}" if prefix else name

        # Skip standard PyTorch modules (they're already traced).
        # Also skip modules that are already _ProfiledModule wrappers to avoid
        # double-wrapping in recursive calls.
        if (
            type(module).__module__.startswith("torch.nn.modules")
            or getattr(module, "_is_profiled_module", False)
        ):
            # Still recurse into children
            wrap_module_forward_with_profiling(module, module_name)
            continue

        # Replace the module in its parent with a ProfiledModule wrapper.
        # setattr on the parent container properly updates PyTorch's module registry.
        setattr(model, name, _ProfiledModule(module, f"nn.Module: {module_name}"))

        # Recurse into the (now-wrapped) module's children.
        # The _is_profiled_module marker above prevents re-wrapping the inner module.
        wrap_module_forward_with_profiling(module, module_name)
