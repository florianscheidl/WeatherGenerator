import json
from types import SimpleNamespace

from weathergen.utils import performance
from weathergen.utils.performance import (
    InferencePhaseProfiler,
    LoaderPhaseProfiler,
    NvtxAnnotator,
)


def test_inference_phase_profiler_writes_monotonic_timing(tmp_path, monkeypatch):
    timestamps = iter([1_000_000_000, 1_500_000_000, 2_250_000_000])
    monkeypatch.setattr(performance.time, "perf_counter_ns", lambda: next(timestamps))
    output_path = tmp_path / "inference_phase_timing_rank0002.json"
    profiler = InferencePhaseProfiler(
        output_path=output_path,
        rank=2,
        world_size=4,
        run_id="example",
        nvtx_annotate=False,
        metadata={"effective_num_workers": 6},
    )

    with profiler.phase("data_loader_next", batch_idx=3):
        pass
    profiler.write()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["rank"] == 2
    assert payload["world_size"] == 4
    assert payload["schema_version"] == 2
    assert payload["started_ns"] == 1_000_000_000
    assert payload["metadata"]["effective_num_workers"] == 6
    assert payload["phases"] == [
        {
            "name": "data_loader_next",
            "batch_idx": 3,
            "start_ns": 1_500_000_000,
            "end_ns": 2_250_000_000,
            "duration_ns": 750_000_000,
            "start_s": 0.5,
            "duration_s": 0.75,
        }
    ]


def test_inference_phase_profiler_balances_nvtx_on_error(monkeypatch):
    calls = []
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_push", calls.append)
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_pop", lambda: calls.append("pop"))
    profiler = InferencePhaseProfiler(
        output_path=None,
        rank=0,
        world_size=1,
        run_id="example",
        nvtx_annotate=True,
    )

    try:
        with profiler.phase("forward", batch_idx=1):
            raise RuntimeError("expected")
    except RuntimeError:
        pass

    assert calls == ["inference.forward.batch_1", "pop"]


def test_nvtx_annotator_nests_prefixes(monkeypatch):
    calls = []
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_push", calls.append)
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_pop", lambda: calls.append("pop"))
    annotator = NvtxAnnotator(True, "loader.")

    with annotator.range("get_batch"):
        with annotator.child("stream.ERA5.").range("read_source_windows"):
            pass

    assert calls == ["loader.get_batch", "loader.stream.ERA5.read_source_windows", "pop", "pop"]


def test_nvtx_annotator_is_silent_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_push", calls.append)
    monkeypatch.setattr(performance.torch.cuda.nvtx, "range_pop", lambda: calls.append("pop"))
    annotator = NvtxAnnotator(False, "loader.")

    with annotator.range("get_batch"), annotator.child("stream.ERA5.").range("read_source_windows"):
        pass

    assert calls == []


def test_loader_phase_profiler_writes_nested_worker_jsonl(tmp_path, monkeypatch):
    timestamps = iter([1_000, 2_000, 5_000, 8_000])
    monkeypatch.setattr(performance.time, "perf_counter_ns", lambda: next(timestamps))
    monkeypatch.setattr(performance.os, "getpid", lambda: 4321)
    monkeypatch.setattr(
        performance.torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(id=3),
    )
    profiler = LoaderPhaseProfiler(
        timing_enabled=True,
        output_dir=tmp_path,
        rank=2,
        run_id="example",
        stage="val",
        prefix="loader.",
    )
    reader_metadata = {"selected_channels": 0}

    with profiler.range("w3.batch_0", {"forecast_steps": 8}):
        with profiler.child("stream.ERA5.").range("read.target.reader_get", reader_metadata):
            reader_metadata["rows"] = 42

    output_path = tmp_path / "loader_phase_timing_rank0002_worker03.jsonl"
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record["name"] for record in records] == [
        "loader.stream.ERA5.read.target.reader_get",
        "loader.w3.batch_0",
    ]
    assert records[0]["worker"] == 3
    assert records[0]["pid"] == 4321
    assert records[0]["stage"] == "val"
    assert records[0]["metadata"] == {"selected_channels": 0, "rows": 42}
    assert records[0]["context"] == [
        {"name": "loader.w3.batch_0", "metadata": {"forecast_steps": 8}}
    ]
    assert records[0]["start_ns"] == 2_000
    assert records[0]["end_ns"] == 5_000
    assert records[0]["duration_ns"] == 3_000


def test_loader_phase_profiler_records_range_that_raises(tmp_path, monkeypatch):
    timestamps = iter([10, 25])
    monkeypatch.setattr(performance.time, "perf_counter_ns", lambda: next(timestamps))
    monkeypatch.setattr(performance.torch.utils.data, "get_worker_info", lambda: None)
    profiler = LoaderPhaseProfiler(timing_enabled=True, output_dir=tmp_path, rank=0)

    try:
        with profiler.range("loader.get_batch"):
            raise RuntimeError("expected")
    except RuntimeError:
        pass

    output_path = tmp_path / "loader_phase_timing_rank0000_workermain.jsonl"
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["name"] == "loader.get_batch"
    assert record["worker"] is None
    assert record["duration_ns"] == 15


def test_loader_phase_profiler_disabled_does_not_create_output(tmp_path):
    profiler = LoaderPhaseProfiler(timing_enabled=False, output_dir=tmp_path)

    with profiler.range("loader.get_batch"):
        pass

    assert list(tmp_path.iterdir()) == []
