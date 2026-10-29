# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
The entry point for training and inference weathergen-atmo
"""

# first, so that the phase timer covers the imports below
from weathergen.utils import phase_timer  # isort: skip

import logging
import os
import pdb
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import weathergen.common.config as config
import weathergen.utils.cli as cli
from weathergen.common.config import Config
from weathergen.common.logger import init_loggers
from weathergen.train.profiling_trainer import ProfilingTrainer
from weathergen.train.trainer import Trainer
from weathergen.utils.nsys_windows import NsysWindows
from weathergen.utils.performance import MemoryTracker, ThroughputTracker
from weathergen.utils.profiling import ProfilingConfig

logger = logging.getLogger(__name__)


def get_trainer(cf: Config) -> Trainer:
    """Select the trainer: the ProfilingTrainer if the run is measured, a plain one otherwise."""
    profiling_cfg = ProfilingConfig.from_config(cf)
    measured = (
        profiling_cfg.enabled
        or NsysWindows.from_config(cf) is not None
        or ThroughputTracker.is_enabled(cf)
        or MemoryTracker.is_enabled(cf)
        or phase_timer.is_enabled(cf)
    )
    if measured:
        logger.info("Profiling or performance logging enabled: running with ProfilingTrainer.")
        return ProfilingTrainer(cf.train_logging, profiling_cfg)

    return Trainer(cf.train_logging)


def _configure_phase_timer(cf: Config) -> None:
    """Enable the phase timer on this rank, or make it a no-op, once the rank is known."""
    phase_timer.timer.configure(
        phase_timer.is_enabled(cf), config.get_path_run(cf), cf.get("rank", 0)
    )
    phase_timer.timer.switch("setup/init")


def _handle_failure(cf: Config) -> None:
    """
    Debug a failed single-process run; end a failed multi-rank run at once.

    Without this, the other ranks of a multi-rank run wait in their next collective until the
    NCCL timeout, and the job runs on. So the failing rank writes what it has, cancels its
    Slurm job step (which ends every rank of the step, this one included), and otherwise exits
    non-zero. `os._exit`, because a normal interpreter exit can block on NCCL's threads.
    """
    _, _, tb = sys.exc_info()
    traceback.print_exc()
    if cf.get("world_size", 1) == 1:
        pdb.post_mortem(tb)
        return

    logger.error(f"Rank {cf.get('rank', '?')} failed; ending the run.")
    phase_timer.timer.write_report()  # exit handlers do not run after os._exit
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    job_id, step_id = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    if job_id and step_id and shutil.which("scancel"):
        subprocess.run(["scancel", f"{job_id}.{step_id}"], check=False)
    os._exit(1)


def train() -> None:
    """Entry point for calling the training code from the command line."""
    main([cli.Stage.train] + sys.argv[1:])


def train_continue() -> None:
    """Entry point for calling train_continue from the command line."""
    main([cli.Stage.train_continue] + sys.argv[1:])


def inference():
    """Entry point for calling the inference code from the command line."""
    main([cli.Stage.inference] + sys.argv[1:])


def main(argl: list[str]):
    phase_timer.timer.switch("setup/config")
    try:
        argl = _fix_argl(argl)
    except ValueError as e:
        logger.error(str(e))

    parser = cli.get_main_parser()
    args = parser.parse_args(argl)
    match args.stage:
        case cli.Stage.train:
            run_train(args)
        case cli.Stage.train_continue:
            run_continue(args)
        case cli.Stage.inference:
            run_inference(args)
        case _:
            logger.error("No stage was found.")


def _fix_argl(argl):  # TODO remove this fix after grace period
    """Ensure `stage` positional argument is in arglist."""
    if argl[0] not in cli.Stage:
        try:
            stage = os.environ.get("WEATHERGEN_STAGE")
        except KeyError as e:
            msg = (
                "`stage` postional argument and environment variable 'WEATHERGEN_STAGE' missing.",
                "Provide either one or the other.",
            )
            raise ValueError(msg) from e

        argl = [stage] + argl

    return argl


def run_inference(args):
    """
    Inference function for WeatherGenerator model.

    Note: Additional configuration for inference (`test_config`) is set in the function.
    """

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)
    cf.data_loading.rng_seed = 42
    cf.stage = args.stage

    phase_timer.timer.switch("setup/dist_init")
    devices = Trainer.init_torch(device=cf.get("device", "auto"))
    cf = Trainer.init_ddp(cf)
    cf = Trainer.init_seeds(cf)
    _configure_phase_timer(cf)

    init_loggers(cf.general.run_id)

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = get_trainer(cf)
    try:
        trainer.inference(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        _handle_failure(cf)


def run_continue(args):
    """
    Function to continue training for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config,
        args.from_run_id,
        args.mini_epoch,
        args.base_config,
        *args.config,
        {},
        cli_overwrite,
    )
    cf = config.set_run_id(cf, args.run_id, args.reuse_run_id)
    cf.stage = args.stage

    mp_method = cf.general.get("multiprocessing_method", "fork")
    phase_timer.timer.switch("setup/dist_init")
    devices = Trainer.init_torch(device=cf.get("device", "auto"), multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)
    cf = Trainer.init_seeds(cf)
    _configure_phase_timer(cf)

    init_loggers(cf.general.run_id)

    # track history of run to ensure traceability of results
    cf.general.run_history += [(args.from_run_id, cf.general.istep)]

    trainer = get_trainer(cf)

    try:
        trainer.run(cf, devices, args.from_run_id, args.mini_epoch)
    except Exception:
        _handle_failure(cf)


def run_train(args):
    """
    Training function for WeatherGenerator model.

    Note: All model configurations are set in the function body.
    """

    cli_overwrite = config.from_cli_arglist(args.options)
    cf = config.load_merge_configs(
        args.private_config, None, None, args.base_config, *args.config, cli_overwrite
    )
    cf = config.set_run_id(cf, args.run_id, False)
    cf.stage = args.stage
    mp_method = cf.general.get("multiprocessing_method", "fork")
    phase_timer.timer.switch("setup/dist_init")
    devices = Trainer.init_torch(device=cf.get("device", "auto"), multiprocessing_method=mp_method)
    cf = Trainer.init_ddp(cf)
    cf = Trainer.init_seeds(cf)
    _configure_phase_timer(cf)

    # this line should probably come after the processes have been sorted out else we get lots
    # of duplication due to multiple process in the multiGPU case
    init_loggers(cf.general.run_id)

    logger.info(f"DDP initialization: rank={cf.rank}, world_size={cf.world_size}")

    cf.streams = config.load_streams(Path(cf.streams_directory))

    trainer = get_trainer(cf)

    try:
        trainer.run(cf, devices)
    except Exception:
        _handle_failure(cf)


if __name__ == "__main__":
    main(sys.argv[1:])
