# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import contextlib
import importlib.util
import sys
import types
from unittest.mock import patch

import pytest

if importlib.util.find_spec("flash_attn") is not None:
    from weathergen.model.model import Model, ModelOutput
else:
    # the attention blocks import flash-attn (not in the cpu extra); the rollout tested here
    # never calls it, so a stub stands in while the model is imported
    stub = types.ModuleType("flash_attn")
    stub.flash_attn_func = stub.flash_attn_varlen_func = None
    sys.modules["flash_attn"] = stub
    try:
        from weathergen.model.model import Model, ModelOutput
    finally:
        del sys.modules["flash_attn"]


class _Tokens:
    def __init__(self, step):
        self.step = step

    def record_stream(self, stream):
        pass


class _Stream:
    """Records what is queued on it."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def record_event(self):
        event = f"e{sum(e.startswith('record') for e in self.log)}"
        self.log.append(f"record {event}")
        return event

    def wait_event(self, event):
        self.log.append(f"{self.name} waits {event}")

    def wait_stream(self, stream):
        self.log.append(f"{self.name} waits {stream.name}")


def _rollout(chunks):
    """Overlapped rollout of the chunks; returns what was queued, in order."""
    log = []
    compute, decode = _Stream("compute", log), _Stream("decode", log)
    current = [compute]

    def forecast_engine(tokens, step, meta_info, coords):
        log.append(f"engine {step}")
        return _Tokens(step)

    def decode_step(model_params, step, tokens, source_samples, output, intermediates):
        assert tokens.step == step
        log.append(f"decode {step} on {current[-1].name}")
        return output

    @contextlib.contextmanager
    def on_stream(stream):
        current.append(stream)
        yield
        current.pop()

    model = types.SimpleNamespace(forecast_engine=forecast_engine, _decode_step=decode_step)
    params = types.SimpleNamespace(rope_coords=None)
    samples = types.SimpleNamespace(samples=[types.SimpleNamespace(meta_info={})])

    output = types.SimpleNamespace(next_tokens=None)
    with (
        patch("torch.cuda.current_stream", lambda: current[-1]),
        patch("torch.cuda.stream", on_stream),
    ):
        for i, chunk in enumerate(chunks):
            next_step = chunks[i + 1][0] if i + 1 < len(chunks) else None
            prev, output = output, ModelOutput.__new__(ModelOutput)
            output.next_tokens = None
            Model._rollout_overlapped(
                model,
                params,
                _Tokens(0),
                prev.next_tokens,
                chunk,
                next_step,
                samples,
                output,
                None,
                decode,
            )
            log.append("return")
    return log


def test_engine_step_is_queued_before_previous_step_is_decoded():
    assert _rollout([[1, 2, 3]]) == [
        "engine 1",
        "record e0",
        "engine 2",
        "record e1",
        "decode waits e0",
        "decode 1 on decode",
        "engine 3",
        "record e2",
        "decode waits e1",
        "decode 2 on decode",
        "decode waits e2",
        "decode 3 on decode",
        "compute waits decode",
        "return",
    ]


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_overlap_continues_across_chunks(chunk_size):
    steps = [1, 2, 3, 4]
    log = _rollout([steps[i : i + chunk_size] for i in range(0, len(steps), chunk_size)])

    # every step is computed once, and before the previous step is decoded
    engines = [e for e in log if e.startswith("engine")]
    assert engines == [f"engine {s}" for s in steps]
    for step in steps[1:]:
        assert log.index(f"engine {step}") < log.index(f"decode {step - 1} on decode")

    # a step computed ahead in the previous chunk is decoded after an event recorded in its
    # own chunk, so after what the compute stream queued in between (e.g. the copies of
    # the chunk's target coordinates)
    for step in steps[chunk_size::chunk_size]:
        decode_idx = log.index(f"decode {step} on decode")
        event = log[decode_idx - 1].removeprefix("decode waits ")
        assert log.index(f"record {event}") > log.index(f"engine {step}")
        assert log.index(f"record {event}") > max(
            i for i, e in enumerate(log[:decode_idx]) if e == "return"
        )
