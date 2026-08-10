# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.stream_data import StreamData


def test_model_batch_reports_temporal_index_and_unique_tensor_storage_bytes() -> None:
    batch = ModelBatch(["stream"], 1, 1, 0, 1, temporal_index=17)
    shared = torch.arange(8, dtype=torch.float32)
    separate = torch.ones(3, dtype=torch.int32)

    batch.source_samples.tokens_lens = shared
    batch.target_samples.tokens_lens = shared[2:]
    batch.source_samples.samples[0].meta_info["stream"] = SampleMetaData(params={}, mask=separate)

    assert batch.temporal_index == 17
    assert batch.unique_tensor_storage_bytes() == (
        shared.untyped_storage().nbytes() + separate.untyped_storage().nbytes()
    )


def test_model_batch_reports_unique_storage_bytes_by_component() -> None:
    batch = ModelBatch(["stream"], 1, 1, 0, 1, temporal_index=17)
    shared = torch.arange(8, dtype=torch.float32)
    mask = torch.ones(3, dtype=torch.int32)
    source_tokens = torch.ones(5, dtype=torch.float64)
    stream_data = object.__new__(StreamData)
    stream_data.source_tokens_cells = [source_tokens, source_tokens[1:]]

    batch.source_samples.tokens_lens = shared
    batch.target_samples.tokens_lens = shared[2:]
    batch.source_samples.samples[0].meta_info["stream"] = SampleMetaData(params={}, mask=mask)
    batch.source_samples.samples[0].streams_data["stream"] = stream_data

    component_bytes = batch.unique_tensor_storage_bytes_by_component()

    assert component_bytes == {
        "shared": shared.untyped_storage().nbytes(),
        "source.stream.metadata.mask": mask.untyped_storage().nbytes(),
        "source.stream.source_tokens_cells": source_tokens.untyped_storage().nbytes(),
    }
    assert sum(component_bytes.values()) == batch.unique_tensor_storage_bytes()


def test_source_tensor_content_fingerprints_compare_logical_values() -> None:
    def make_batch(source_values: torch.Tensor, target_value: float) -> ModelBatch:
        batch = ModelBatch(["stream"], 1, 1, 0, 1, temporal_index=17)
        source_data = object.__new__(StreamData)
        source_data.source_tokens_cells = [source_values]
        target_data = object.__new__(StreamData)
        target_data.target_tokens = [torch.tensor([target_value])]
        batch.source_samples.samples[0].streams_data["stream"] = source_data
        batch.target_samples.samples[0].streams_data["stream"] = target_data
        return batch

    reference = make_batch(torch.arange(6, dtype=torch.float32).reshape(2, 3), 1.0)
    same_values = make_batch(torch.arange(6, dtype=torch.float32).reshape(2, 3), 2.0)
    changed_source = make_batch(torch.arange(6, dtype=torch.float32).reshape(3, 2), 1.0)

    reference_fingerprints = reference.source_tensor_content_fingerprints()

    assert reference_fingerprints == same_values.source_tensor_content_fingerprints()
    assert reference_fingerprints != changed_source.source_tensor_content_fingerprints()
    assert set(reference_fingerprints) == {"_aggregate", "stream.source_tokens_cells"}
    assert all(0 <= word < 2**32 for words in reference_fingerprints.values() for word in words)
