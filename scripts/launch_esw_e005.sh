#!/usr/bin/env bash
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ESW-E005: the sync-barrier stack (SYNC-I5) against its control, timed with the phase timer
# (3 runs per arm, interleaved), plus one sync count of the treatment as a pinning check.
# One node each (4 GPUs, FSDP over 4 ranks), so at most 4 nodes are in use at once. Streams:
# the low-resolution set (era5_georing_avhrr_synop_lowres) instead of the base config's
# high-resolution one, whose data loader workers ran out of shared memory.
#
# Run with bash from the root of a clean WeatherGenerator checkout that has
# ../WeatherGenerator-private and ../weathergen-configs next to it:
#
#   bash scripts/launch_esw_e005.sh            # submit
#   DRY_RUN=1 bash scripts/launch_esw_e005.sh  # check the paths, print the commands
#   RUNS=tc1 bash scripts/launch_esw_e005.sh   # only these runs (default: all, in order:
#                                              # tc1 tt1 tc2 tt2 tc3 tt3 st)
#
# Run IDs are <RUN_TAG>tc<i> (control), <RUN_TAG>tt<i> (treatment) and <RUN_TAG>st (pinning
# check), RUN_TAG=e5 by default. The launcher refuses a run ID whose slurm directory exists, so
# a retry after a partial launch needs a new tag (e.g. RUN_TAG=e5b), or the old directories
# removed once their jobs are known not to exist. The script prints each run ID it submitted.
#
# Each launch checks out the commit it measures (the launcher copies the tracked files at
# submission); the checkout returns to where it was at the end. Neither commit contains this
# script, so everything runs from main(), which bash has read completely before the first
# checkout.

set -euo pipefail

main() {
  local launcher=../WeatherGenerator-private/hpc/launch-slurm.py
  local base=../weathergen-configs/performance/base_operan_georing_avhrr_forecasting_highres.yml
  local hp5=../weathergen-configs/performance/healpix_5.yml
  # the same relative form as the base config's streams_directory
  local streams=../weathergen-configs/streams/era5_georing_avhrr_synop_lowres/
  local control=0f1f50fefa70602d3d20d1e0e7dee44d13ca4cf3
  local treatment=18e85186253f39da673df21f464c53f83ac753d9

  # preflight: fail before submitting anything
  local missing=0 path commit
  for path in "$launcher" "$base" "$hp5" "$streams"; do
    if [ ! -e "$path" ]; then
      echo "missing: $path (relative to $PWD)" >&2
      missing=1
    fi
  done
  for commit in "$control" "$treatment"; do
    if ! git cat-file -e "$commit^{commit}" 2>/dev/null; then
      echo "commit $commit not found; git fetch" \
        "https://github.com/florianscheidl/WeatherGenerator.git" \
        "develop-ssl-diffusion-performance-local" >&2
      missing=1
    fi
  done
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "the checkout has uncommitted changes; the launcher would copy them" >&2
    missing=1
  fi
  [ "$missing" -eq 0 ] || exit 1

  echo "WeatherGenerator-private: $(git -C ../WeatherGenerator-private rev-parse HEAD)"
  echo "weathergen-configs:       $(git -C ../weathergen-configs rev-parse HEAD)"

  local tag=${RUN_TAG:-e5}
  orig=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
  # back to where the checkout was, also when a launch fails
  trap 'git checkout -q "$orig"' EXIT

  local selected=" ${RUNS:-tc1 tt1 tc2 tt2 tc3 tt3 st} "
  local data=(data_loading.rng_seed=1 training_config.start_date=2017-06-01T00:00
    "streams_directory=$streams")
  local timing=("${data[@]}" training_config.num_mini_epochs=1
    training_config.samples_per_mini_epoch=1024 validation_config.samples_per_mini_epoch=32)

  # launch <commit> <run name> <time> <extra config> <option>...
  launch() {
    local commit=$1 run_id=$tag$2 time=$3 extra_config=$4
    [[ "$selected" == *" $2 "* ]] || return 0
    shift 4
    git checkout -q --detach "$commit"
    [ -e "$extra_config" ] || { echo "missing at $commit: $extra_config" >&2; exit 1; }
    local cmd=("$launcher" --nodes=1 --time="$time"
      --wgen-dir "$PWD" --run-id "$run_id" --base-config "$base"
      --config "$hp5" "$extra_config" --options "$@")
    if [ -n "${DRY_RUN:-}" ]; then
      echo "[$(git rev-parse --short HEAD)] ${cmd[*]}"
    else
      "${cmd[@]}"
      echo ">>> submitted $run_id at $(git rev-parse --short HEAD)"
    fi
  }

  # timing: 1024 samples / (4 ranks x batch 1) = 256 steps per rank. Each rank's share is
  # rounded down to a multiple of its 8 workers, so 32 validation samples (the smallest count
  # that is not zero) give 8 validation batches per rank. Three repeats per arm, interleaved.
  local i
  for i in 1 2 3; do
    launch "$control" "tc$i" 45:00 ./config/config_phase_timing.yml "${timing[@]}"
    launch "$treatment" "tt$i" 45:00 ./config/config_phase_timing.yml "${timing[@]}"
  done

  # pinning check, treatment only (not timed)
  launch "$treatment" st 30:00 ./config/config_sync_count.yml "${data[@]}"
}

main "$@"
