from weathergen.utils import performance
from weathergen.utils.performance import NvtxAnnotator


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
