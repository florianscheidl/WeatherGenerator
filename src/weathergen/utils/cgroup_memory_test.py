# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest

from weathergen.utils.cgroup_memory import (
    CgroupMemoryTimeline,
    read_cgroup_memory_snapshot,
    read_process_memory_snapshot,
    resolve_cgroup_v2_path,
)


def _write_cgroup_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory.current").write_text("1200\n")
    (path / "memory.peak").write_text("2400\n")
    (path / "memory.stat").write_text("anon 700\nfile 400\nshmem 100\n")
    (path / "memory.events").write_text("low 0\nhigh 1\nmax 2\noom 3\noom_kill 4\n")


def _write_smaps_rollup(proc_root: Path, pid: int, offset_kib: int = 0) -> None:
    process_path = proc_root / str(pid)
    process_path.mkdir(parents=True)
    (process_path / "smaps_rollup").write_text(
        "1000-2000 ---p 00000000 00:00 0 [rollup]\n"
        f"Rss: {10 + offset_kib} kB\n"
        f"Pss: {9 + offset_kib} kB\n"
        f"Pss_Anon: {6 + offset_kib} kB\n"
        "Pss_File: 2 kB\n"
        "Pss_Shmem: 1 kB\n"
        f"Private_Dirty: {5 + offset_kib} kB\n"
    )


class _FakeProcess:
    def __init__(
        self,
        pid: int,
        create_time: float,
        children: list["_FakeProcess"] | None = None,
    ) -> None:
        self.pid = pid
        self._create_time = create_time
        self._children = children or []

    def create_time(self) -> float:
        return self._create_time

    def children(self, recursive: bool) -> list["_FakeProcess"]:
        assert recursive
        return self._children


def test_resolve_slurm_job_cgroup(tmp_path: Path) -> None:
    cgroup_mount = tmp_path / "cgroup"
    job_path = cgroup_mount / "slurm" / "job_42"
    task_path = job_path / "step_0" / "task_0"
    _write_cgroup_files(job_path)
    _write_cgroup_files(task_path)
    proc_cgroup = tmp_path / "proc-cgroup"
    proc_cgroup.write_text("0::/slurm/job_42/step_0/task_0\n")

    assert resolve_cgroup_v2_path(proc_cgroup, cgroup_mount, slurm_job_id="42") == job_path


def test_read_cgroup_memory_snapshot(tmp_path: Path) -> None:
    _write_cgroup_files(tmp_path)

    assert read_cgroup_memory_snapshot(tmp_path) == {
        "diagnostic.cgroup_memory.current_bytes": 1200.0,
        "diagnostic.cgroup_memory.peak_bytes": 2400.0,
        "diagnostic.cgroup_memory.anon_bytes": 700.0,
        "diagnostic.cgroup_memory.file_bytes": 400.0,
        "diagnostic.cgroup_memory.shmem_bytes": 100.0,
        "diagnostic.cgroup_memory.events.high": 1.0,
        "diagnostic.cgroup_memory.events.max": 2.0,
        "diagnostic.cgroup_memory.events.oom": 3.0,
        "diagnostic.cgroup_memory.events.oom_kill": 4.0,
    }


def test_read_process_memory_snapshot(tmp_path: Path) -> None:
    _write_smaps_rollup(tmp_path, 10)

    assert read_process_memory_snapshot(10, tmp_path) == {
        "diagnostic.process_memory.pss_bytes": 9 * 1024.0,
        "diagnostic.process_memory.pss_anon_bytes": 6 * 1024.0,
        "diagnostic.process_memory.pss_file_bytes": 2 * 1024.0,
        "diagnostic.process_memory.pss_shmem_bytes": 1024.0,
        "diagnostic.process_memory.rss_bytes": 10 * 1024.0,
        "diagnostic.process_memory.private_dirty_bytes": 5 * 1024.0,
    }


def test_timeline_records_samples_and_markers(tmp_path: Path) -> None:
    _write_cgroup_files(tmp_path)
    records: list[dict[str, float]] = []
    timeline = CgroupMemoryTimeline(records.append, sampling_interval_ms=10, cgroup_path=tmp_path)

    timeline.start()
    timeline.record_stage("batch_dequeued", monotonic_ns=123, batch_index=2)
    timeline.stop()

    assert any(record.get("diagnostic.timeline.sample") == 1.0 for record in records)
    marker = next(
        record
        for record in records
        if record.get("diagnostic.timeline.stage.batch_dequeued") == 1.0
    )
    assert marker["diagnostic.timeline.monotonic_ns"] == 123.0
    assert marker["diagnostic.timeline.rank"] == 0.0
    assert marker["diagnostic.timeline.batch_index"] == 2.0


def test_timeline_samples_trainer_and_worker_per_rank(tmp_path: Path) -> None:
    trainer = _FakeProcess(10, 1.0, [_FakeProcess(11, 2.0)])
    _write_smaps_rollup(tmp_path, 10)
    _write_smaps_rollup(tmp_path, 11, offset_kib=10)
    records: list[dict[str, float]] = []
    timeline = CgroupMemoryTimeline(
        records.append,
        sampling_interval_ms=100,
        sample_cgroup=False,
        process_sampling_interval_ms=10,
        rank=2,
        proc_root=tmp_path,
        trainer_process=trainer,
    )

    timeline.start()
    timeline.stop()

    process_records = [
        record for record in records if record.get("diagnostic.process_memory.sample") == 1.0
    ]
    assert len(process_records) == 2
    trainer_record = next(
        record
        for record in process_records
        if record.get("diagnostic.process_memory.role.trainer") == 1.0
    )
    worker_record = next(
        record
        for record in process_records
        if record.get("diagnostic.process_memory.role.worker") == 1.0
    )
    assert trainer_record["diagnostic.timeline.rank"] == 2.0
    assert trainer_record["diagnostic.process_memory.worker_slot"] == -1.0
    assert worker_record["diagnostic.process_memory.pid"] == 11.0
    assert worker_record["diagnostic.process_memory.worker_slot"] == 0.0
    assert worker_record["diagnostic.process_memory.worker_generation"] == 0.0


def test_worker_slot_generation_increments_after_respawn(tmp_path: Path) -> None:
    _write_smaps_rollup(tmp_path, 10)
    timeline = CgroupMemoryTimeline(
        lambda _record: None,
        sample_cgroup=False,
        process_sampling_interval_ms=1000,
        proc_root=tmp_path,
        trainer_process=_FakeProcess(10, 1.0),
    )

    assert timeline._update_worker_metadata({(11, 2.0)})[(11, 2.0)] == (0, 0)
    timeline._update_worker_metadata(set())
    assert timeline._update_worker_metadata({(12, 3.0)})[(12, 3.0)] == (0, 1)


def test_watchdog_configuration_requires_timeout_and_output_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configured together"):
        CgroupMemoryTimeline(
            lambda _record: None,
            sample_cgroup=False,
            watchdog_timeout_seconds=1,
        )
    with pytest.raises(ValueError, match="greater than zero"):
        CgroupMemoryTimeline(
            lambda _record: None,
            sample_cgroup=False,
            watchdog_timeout_seconds=0,
            watchdog_output_path=tmp_path / "watchdog.log",
        )


def test_watchdog_dumps_once_for_each_stalled_stage(tmp_path: Path) -> None:
    records: list[dict[str, float]] = []
    output_path = tmp_path / "watchdog.log"
    timeline = CgroupMemoryTimeline(
        records.append,
        sample_cgroup=False,
        watchdog_timeout_seconds=1,
        watchdog_output_path=output_path,
        rank=2,
    )

    timeline._last_progress = (1_000_000_000, "ema_start")
    timeline._maybe_dump_watchdog(2_100_000_000)
    timeline._maybe_dump_watchdog(3_100_000_000)

    output = output_path.read_text()
    assert output.count("trainer watchdog") == 1
    assert "rank=2 last_stage=ema_start stalled_seconds=1.1" in output
    assert len(records) == 1
    assert records[0]["diagnostic.timeline.watchdog_dump"] == 1.0
    assert records[0]["diagnostic.timeline.watchdog_last_stage.ema_start"] == 1.0

    timeline._last_progress = (4_000_000_000, "ema_end")
    timeline._maybe_dump_watchdog(5_100_000_000)

    output = output_path.read_text()
    assert output.count("trainer watchdog") == 2
    assert "rank=2 last_stage=ema_end stalled_seconds=1.1" in output
