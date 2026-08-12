import json

from weathergen.utils import performance
from weathergen.utils.performance import InferencePhaseProfiler


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
    assert payload["metadata"]["effective_num_workers"] == 6
    assert payload["phases"] == [
        {
            "name": "data_loader_next",
            "batch_idx": 3,
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
