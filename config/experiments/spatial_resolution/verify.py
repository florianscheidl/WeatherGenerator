"""Check the experiment matrix without datasets, private config, or a GPU.

Run from the repository root with:
    uv run --no-sync python config/experiments/spatial_resolution/verify.py
"""

import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from weathergen.common import config

ROOT = Path("config/experiments/spatial_resolution")
logger = logging.getLogger(__name__)
INPUTS = ("o96_o256", "n320_o256", "n320_h512")
ARMS = ("data_parallel", "spatial_full_read", "spatial_local")
SOURCE_NAMES = {
    "ERA5_in",
    "METEOSAT_SEVIRI_IR",
    "GOES_ABI_IR",
    "GOES_ABI_VIS",
    "HIMAWARI_AHI_IR",
    "HIMAWARI_AHI_VIS",
}


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Treat lists as atomic settings so channel order remains significant."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def load(variant: str, level: int, arm: str) -> dict:
    with patch.object(config, "_load_private_conf", return_value=OmegaConf.create({})):
        cf = config.load_merge_configs(
            None,
            None,
            None,
            ROOT / "base.yml",
            ROOT / f"input_{variant}.yml",
            ROOT / f"healpix_{level}.yml",
            ROOT / f"{arm}.yml",
        )
    # run_train reloads this directory after config merging on both code bases.
    cf.streams = config.load_streams(Path(cf.streams_directory))
    return OmegaConf.to_container(config._strip_interpolation(cf), resolve=False)


def verify_launcher_overlays(variant: str, level: int, arm: str) -> None:
    """Ensure separately MLflow-logged extra configs have no repeated parameter keys."""
    paths = (
        ROOT / f"input_{variant}.yml",
        ROOT / f"healpix_{level}.yml",
        ROOT / f"{arm}.yml",
    )
    keys_seen: set[str] = set()
    for path in paths:
        keys = set(OmegaConf.load(path).keys())
        assert keys_seen.isdisjoint(keys), (path, keys_seen & keys)
        keys_seen.update(keys)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)
    baseline = load("o96_o256", 5, "data_parallel")
    frozen = flatten(baseline)
    allowed = {
        "streams_directory",
        "healpix_level",
        "encoder_spatial_parallel_size",
        "spatial_local_physical_loss",
        "data_loading.reader_spatial_filtering",
        *(f"streams.{name}.filenames" for name in SOURCE_NAMES),
    }
    assert frozen["data_loading.num_workers"] == 8
    assert frozen["data_loading.memory_pinning"] is False
    assert frozen["training_config.forecast.num_steps"] == 2
    assert frozen["training_config.model_input.source_masking.num_samples"] == 1
    assert frozen["training_config.model_input.source_masking.num_steps_input"] == 1
    assert frozen["training_config.learning_rate_scheduling.parallel_scaling_policy"] == "const"

    signatures = {}
    for variant in INPUTS:
        for level in (5, 6):
            for arm in ARMS:
                verify_launcher_overlays(variant, level, arm)
                cf = load(variant, level, arm)
                current = flatten(cf)
                changed = {
                    key
                    for key in current.keys() | frozen.keys()
                    if current.get(key) != frozen.get(key)
                }
                assert changed <= allowed, (variant, level, arm, changed - allowed)
                assert cf["healpix_level"] == level
                assert cf["encoder_spatial_parallel_size"] == (1 if arm == "data_parallel" else 4)
                assert cf["spatial_local_physical_loss"] == (arm == "spatial_local")
                assert cf["data_loading"]["reader_spatial_filtering"] == (arm == "spatial_local")
                assert cf["streams"]["ERA5"] == baseline["streams"]["ERA5"]
                analysis_file = cf["streams"]["ERA5_in"]["filenames"][0]
                assert ("-n320-" in analysis_file) == (variant != "o96_o256")
                for name in SOURCE_NAMES - {"ERA5_in"}:
                    stream = cf["streams"][name]
                    assert stream["forcing"] is True
                    assert stream["embed"]["net"] == "transformer"
                    assert stream["embed"]["dim_embed"] == 512
                    assert stream["token_size"] == 1024
                    assert ("-h512-" in stream["filenames"][0]) == (variant == "n320_h512")
                key = f"{variant}/h{level}/{arm}"
                signatures[key] = hashlib.sha256(
                    json.dumps(cf, sort_keys=True).encode()
                ).hexdigest()

    logger.info("Verified 18 merged configurations (six resolution cells x three execution arms).")
    logger.info("Only source filenames, HEALPix level, and declared execution switches differ.")
    logger.info("Launcher overlays have disjoint MLflow parameter keys.")
    logger.info("Resolved-config SHA256 signatures (compare between branches):")
    logger.info(json.dumps(signatures, indent=2, sort_keys=True))

    for level in (5, 6):
        verify_launcher_overlays("o96_o256", level, "data_parallel_single_worker")
        diagnostic = load("o96_o256", level, "data_parallel_single_worker")
        diagnostic_flat = flatten(diagnostic)
        changed = {
            key
            for key in diagnostic_flat.keys() | frozen.keys()
            if diagnostic_flat.get(key) != frozen.get(key)
        }
        assert changed <= {"healpix_level", "data_loading.num_workers"}, changed
        assert diagnostic["healpix_level"] == level
        assert diagnostic["data_loading"]["num_workers"] == 1

    logger.info("Verified matched HEALPix 5/6 single-worker host-memory diagnostics.")


if __name__ == "__main__":
    main()
