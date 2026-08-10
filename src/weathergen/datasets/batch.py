"""
Data structures for student-teacher multi-view training.

Provides clean separation between:
  - Model data (StreamData objects containing tensors)
  - View metadata (spatial masks, strategies, relationships)
"""

import copy
import hashlib
import struct
from dataclasses import dataclass

import numpy as np
import torch

from weathergen.common.config import Config
from weathergen.datasets.stream_data import StreamData


@dataclass
class SampleMetaData:
    # sample parameters (masking)
    params: Config | dict

    mask: torch.Tensor | None = None

    global_params: dict | None = None


class Sample:
    # keys: stream name, values: SampleMetaData
    meta_info: dict[str | SampleMetaData]

    # data for all streams
    # keys: stream_name, values: StreamData
    streams_data: dict[str, StreamData | None]

    def pin_memory(self):
        """Pin all tensors in this Sample to CPU pinned memory"""

        # Pin StreamData objects in streams_data dict
        if hasattr(self, "streams_data") and isinstance(self.streams_data, dict):
            for _stream_name, stream_data in self.streams_data.items():
                if stream_data is not None and hasattr(stream_data, "pin_memory"):
                    stream_data.pin_memory()

        # Pin tensors in meta_info
        if hasattr(self, "meta_info") and isinstance(self.meta_info, dict):
            for _key, meta_data in self.meta_info.items():
                if isinstance(meta_data, SampleMetaData):
                    # Pin mask tensor
                    if meta_data.mask is not None and isinstance(meta_data.mask, torch.Tensor):
                        meta_data.mask = meta_data.mask.pin_memory()

        return self

    def __init__(self, stream_names: list[str]) -> None:
        self.meta_info = {}

        self.streams_data = {}
        for stream_name in stream_names:
            self.streams_data[stream_name] = None

    def to_device(self, device) -> None:
        for key in self.meta_info.keys():
            self.meta_info[key].mask = (
                self.meta_info[key].mask.to(device, non_blocking=True)
                if self.meta_info[key].mask is not None
                else None
            )

        for key, val in self.streams_data.items():
            if val is not None:
                self.streams_data[key] = val.to_device(device)

    def is_empty(self) -> bool:
        """
        Check if sample is empty
        """
        empty = [s.empty() if s is not None else True for _, s in self.streams_data.items()]
        return np.array(empty).all()

    def is_nan(self) -> bool:
        """
        Check if sample is all NaN
        """
        is_nan = [s.nan() if s is not None else False for _, s in self.streams_data.items()]
        return np.array(is_nan).all()

    def sources_empty(self) -> bool:
        """
        Check if sources for sample are empty
        """
        empty = [s.source_empty() if s is not None else True for _, s in self.streams_data.items()]
        return np.array(empty).all()

    def sources_nan(self) -> bool:
        """
        Check if sources for sample are all NaN
        """
        is_nan = [s.source_nan() if s is not None else False for _, s in self.streams_data.items()]
        return np.array(is_nan).all()

    def targets_empty(self) -> bool:
        """
        Check if targets for sample are empty
        """
        empty = [s.target_empty() if s is not None else True for _, s in self.streams_data.items()]
        return np.array(empty).all()

    def targets_nan(self) -> bool:
        """
        Check if targets for sample are all NaN
        """
        is_nan = [s.target_nan() if s is not None else False for _, s in self.streams_data.items()]
        return np.array(is_nan).all()

    def add_stream_data(self, stream_name: str, stream_data: StreamData) -> None:
        """
        Add data for stream @stream_name to sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        self.streams_data[stream_name] = stream_data

    def add_meta_info(self, stream_name: str, meta_info: SampleMetaData) -> None:
        """
        Add metadata for stream @stream_name to sample
        """
        self.meta_info[stream_name] = meta_info

    def get_stream_data(self, stream_name: str) -> StreamData:
        """
        Get data for stream @stream_name from sample
        """
        assert self.streams_data.get(stream_name, -1) != -1, "stream name does not exist"
        return self.streams_data[stream_name]

    def get_num_source_steps(self) -> int:
        """
        Get number of source steps from smallest of all available streams
        """
        lens = [
            stream.get_num_source_steps()
            for _, stream in self.streams_data.items()
            if stream is not None
        ]
        return min(lens) if len(lens) > 0 else 0

    def get_num_target_steps(self) -> int:
        """
        Get number of target steps from smallest of all available streams
        """
        lens = [
            stream.get_num_target_steps()
            for _, stream in self.streams_data.items()
            if stream is not None
        ]
        return min(lens) if len(lens) > 0 else 0


class BatchSamples:
    """
    Container for source or target samples
    """

    samples: list[Sample]
    tokens_lens: torch.Tensor | None
    output_steps: int
    output_idxs: list[int]
    device: str | None

    def __init__(
        self, stream_names: list[str], num_samples: int, output_steps, output_idxs
    ) -> None:
        self.samples = [Sample(stream_names) for _ in range(num_samples)]
        self.tokens_lens = None
        self.output_steps = output_steps
        self.output_idxs = output_idxs
        self.device = None

    def __len__(self) -> int:
        return len(self.samples)

    def to_device(self, device):
        for sample in self.samples:
            sample.to_device(device)

        self.tokens_lens = (
            self.tokens_lens.to(device, non_blocking=True) if self.tokens_lens is not None else None
        )

        self.device = device

        return self

    def get_samples(self) -> list[Sample]:
        return self.samples

    def get_subset(self, subset: list | None = None):
        if subset is None:
            return self
        else:
            assert len(list(set(subset))) == len(subset), "subset contains duplicates"
            # create copy and then select subset for samples and tokens_lens
            bs = copy.deepcopy(self)
            bs.samples = [bs.samples[i] for i in subset]
            torch_idxs = torch.tensor(subset, dtype=torch.long, device=bs.tokens_lens.device)
            bs.tokens_lens = torch.index_select(bs.tokens_lens, 1, torch_idxs)
            return bs

    def get_num_source_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        return self.samples[0].get_num_source_steps()

    def get_num_target_steps(self) -> int:
        """
        Get number of target steps from smallest of all available streams
        """
        return self.samples[0].get_num_target_steps()

    def get_output_idxs(self) -> int:
        """
        Get forecast indices
        """
        return self.output_idxs

    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    def sources_empty(self) -> bool:
        """
        Check if sources for all samples are empty
        """
        return np.array([s.sources_empty() if s is not None else True for s in self.samples]).all()

    def targets_empty(self) -> bool:
        """
        Check if targets for all samples are empty
        """
        return np.array([s.targets_empty() if s is not None else True for s in self.samples]).all()

    def sources_nan(self) -> bool:
        """
        Check if sources for all samples are all NaN
        """
        return np.array([s.sources_nan() if s is not None else False for s in self.samples]).all()

    def targets_nan(self) -> bool:
        """
        Check if targets for all samples are all NaN
        """
        return np.array([s.targets_nan() if s is not None else False for s in self.samples]).all()

    def pin_memory(self):
        """Pin all tensors in this batch to CPU pinned memory"""

        # pin all samples
        for sample in self.samples:
            sample.pin_memory()

        # pin source_tokens_lens
        if isinstance(self.tokens_lens, torch.Tensor):
            self.tokens_lens = self.tokens_lens.pin_memory()

        return self


class ModelBatch:
    """
    Container for all data and metadata for one training batch.
    """

    # source samples (for model)
    source_samples: BatchSamples

    # target samples (for TargetAuxCalculator)
    target_samples: BatchSamples

    # index of corresponding target (for source samples) or source (for target samples)
    # these are in 1-to-1 corresponding for classical training modes (e.g. MTM, forecasting) but
    # can be more complex for strategies like student-teacher training
    source2target_matching_idxs: np.typing.NDArray[np.int32]
    target2source_matching_idxs: np.typing.NDArray[np.int32]

    # indices of valid outputs
    output_idxs: list[int]

    # device of the tensors in the batch
    device: str | torch.device

    # temporal index selected by the sampler
    temporal_index: int

    def __init__(
        self,
        stream_names: list[str],
        num_source_samples: int,
        num_target_samples: int,
        output_offset,
        output_steps,
        temporal_index: int,
    ) -> None:
        """ """

        # define forecast indices
        self.output_offset = output_offset
        self.output_steps = output_steps
        self.output_idxs = list(range(output_offset, output_steps))
        self.temporal_index = int(temporal_index)

        self.source_samples = BatchSamples(
            stream_names, num_source_samples, output_steps, self.output_idxs
        )
        self.target_samples = BatchSamples(
            stream_names, num_target_samples, output_steps, self.output_idxs
        )

        self.source2target_matching_idxs = np.full(num_source_samples, -1, dtype=np.int32)
        self.target2source_matching_idxs = [[] for _ in range(num_target_samples)]

    def unique_tensor_storage_bytes(self) -> int:
        """Return bytes owned by unique tensor storages referenced by this batch."""
        return sum(self.unique_tensor_storage_bytes_by_component().values())

    def unique_tensor_storage_bytes_by_component(self) -> dict[str, int]:
        """Return unique tensor-storage bytes grouped by batch role, stream, and field.

        A storage referenced by more than one component is counted once in ``shared``.
        Repeated references within the same component remain attributed to that component.
        """
        storage_bytes: dict[tuple[str, int | None, int, int], int] = {}
        storage_components: dict[tuple[str, int | None, int, int], set[str]] = {}

        def add_tensors(value, component: str) -> None:
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                num_bytes = storage.nbytes()
                key = (
                    value.device.type,
                    value.device.index,
                    storage.data_ptr(),
                    num_bytes,
                )
                storage_bytes[key] = num_bytes
                storage_components.setdefault(key, set()).add(component)
            elif isinstance(value, dict):
                for item in value.values():
                    add_tensors(item, component)
            elif isinstance(value, list | tuple):
                for item in value:
                    add_tensors(item, component)

        for role, batch_samples in (
            ("source", self.source_samples),
            ("target", self.target_samples),
        ):
            add_tensors(batch_samples.tokens_lens, f"{role}._batch.tokens_lens")
            for sample in batch_samples.samples:
                for stream_name, metadata in sample.meta_info.items():
                    add_tensors(metadata.mask, f"{role}.{stream_name}.metadata.mask")
                for stream_name, stream_data in sample.streams_data.items():
                    if stream_data is not None:
                        for field_name, value in vars(stream_data).items():
                            add_tensors(value, f"{role}.{stream_name}.{field_name}")

        component_bytes: dict[str, int] = {}
        for key, components in storage_components.items():
            component = next(iter(components)) if len(components) == 1 else "shared"
            component_bytes[component] = component_bytes.get(component, 0) + storage_bytes[key]
        return dict(sorted(component_bytes.items()))

    def source_tensor_content_fingerprints(self) -> dict[str, tuple[int, int, int, int]]:
        """Return deterministic BLAKE2b-128 fingerprints of CPU source tensors.

        The four 32-bit words are exactly representable by the float-only diagnostic
        logger. Tensor names, dtype, shape, order, and logical values contribute to
        the digest; storage addresses do not.
        """
        component_hashers = {}

        def add_value(hasher, value) -> bool:
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise ValueError("Source content fingerprints require CPU tensors")
                if value.layout != torch.strided:
                    raise ValueError("Source content fingerprints require strided tensors")
                tensor = value.detach().contiguous()
                hasher.update(b"tensor\0")
                hasher.update(str(tensor.dtype).encode("ascii"))
                hasher.update(b"\0")
                hasher.update(struct.pack("<Q", tensor.ndim))
                for dimension in tensor.shape:
                    hasher.update(struct.pack("<q", dimension))
                byte_values = tensor.reshape(-1).view(torch.uint8).numpy()
                hasher.update(memoryview(byte_values))
                return True
            if isinstance(value, dict):
                found_tensor = False
                for key in sorted(value, key=str):
                    hasher.update(b"dict-key\0")
                    hasher.update(str(key).encode("utf-8"))
                    hasher.update(b"\0")
                    found_tensor = add_value(hasher, value[key]) or found_tensor
                return found_tensor
            if isinstance(value, list | tuple):
                found_tensor = False
                for index, item in enumerate(value):
                    hasher.update(b"item\0")
                    hasher.update(struct.pack("<Q", index))
                    found_tensor = add_value(hasher, item) or found_tensor
                return found_tensor
            return False

        def add_component(component: str, sample_index: int, value) -> None:
            occurrence = hashlib.blake2b(digest_size=16, person=b"WGModelBatch-v1")
            occurrence.update(b"sample\0")
            occurrence.update(struct.pack("<Q", sample_index))
            if add_value(occurrence, value):
                component_hashers.setdefault(
                    component,
                    hashlib.blake2b(digest_size=16, person=b"WGModelBatch-v1"),
                ).update(occurrence.digest())

        add_component("_batch.tokens_lens", 0, self.source_samples.tokens_lens)
        for sample_index, sample in enumerate(self.source_samples.samples):
            for stream_name in sorted(sample.meta_info):
                add_component(
                    f"{stream_name}.metadata.mask",
                    sample_index,
                    sample.meta_info[stream_name].mask,
                )
            for stream_name in sorted(sample.streams_data):
                stream_data = sample.streams_data[stream_name]
                if stream_data is None:
                    continue
                for field_name, value in sorted(vars(stream_data).items()):
                    add_component(f"{stream_name}.{field_name}", sample_index, value)

        component_digests = {
            component: hasher.digest() for component, hasher in sorted(component_hashers.items())
        }
        aggregate = hashlib.blake2b(digest_size=16, person=b"WGModelBatch-v1")
        for component, digest in component_digests.items():
            aggregate.update(component.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest)
        component_digests["_aggregate"] = aggregate.digest()

        return {
            component: struct.unpack("<4I", digest)
            for component, digest in component_digests.items()
        }

    def pin_memory(self):
        """Pin all tensors in this batch to CPU pinned memory"""

        # pin source samples
        self.source_samples.pin_memory()

        # pin target samples
        self.target_samples.pin_memory()

        return self

    def to_device(self, device):  # -> ModelBatch
        """
        Move batch to device
        """

        self.source_samples.to_device(device)
        self.target_samples.to_device(device)

        self.device = device

        return self

    def add_source_stream(
        self,
        source_sample_idx: int,
        target_sample_idx: int,
        stream_name: str,
        stream_data: StreamData,
        source_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @source_sample_idx
        """
        self.source_samples.samples[source_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info
        self.source_samples.samples[source_sample_idx].add_meta_info(stream_name, source_meta_info)

        assert target_sample_idx < len(self.target_samples), "invalid value for target_sample_idx"
        self.source2target_matching_idxs[source_sample_idx] = target_sample_idx

    def add_target_stream(
        self,
        target_sample_idx: int,
        source_sample_idx: int | list[int],
        stream_name: str,
        stream_data: StreamData,
        target_meta_info: SampleMetaData,
    ) -> None:
        """
        Add data for one stream to sample @target_sample_idx
        """
        self.target_samples.samples[target_sample_idx].add_stream_data(stream_name, stream_data)

        # add the meta_info -- for target we have different
        self.target_samples.samples[target_sample_idx].add_meta_info(stream_name, target_meta_info)

        if isinstance(source_sample_idx, int):
            assert source_sample_idx < len(self.source_samples), (
                "invalid value for source_sample_idx"
            )
        else:
            assert all(idx < len(self.source_samples) for idx in source_sample_idx), (
                "invalid value for source_sample_idx"
            )
        self.target2source_matching_idxs[target_sample_idx] = source_sample_idx

    def is_empty(self):
        """
        Check if batch is empty
        """
        return self.source_samples.sources_empty() or self.target_samples.targets_empty()

    def is_nan(self):
        """
        Check if batch is all NaN
        """
        return self.source_samples.sources_nan() or self.target_samples.targets_nan()

    def sources_empty(self):
        """
        Check if batch sources are empty
        """
        return self.source_samples.sources_empty()

    def sources_nan(self):
        """
        Check if batch sources are all NaN
        """
        return self.source_samples.sources_nan()

    def targets_empty(self):
        """
        Check if batch targets are empty
        """
        return self.target_samples.targets_empty()

    def targets_nan(self):
        """
        Check if batch targets are all NaN
        """
        return self.target_samples.targets_nan()

    def len_sources(self) -> int:
        """
        Number of source samples
        """
        return len(self.source_samples)

    def len_targets(self) -> int:
        """
        Number of target samples
        """
        return len(self.target_samples)

    def get_source_sample(self, idx: int) -> Sample:
        """
        Get a source sample
        """
        return self.source_samples.samples[idx]

    def get_source_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get source samples
        """
        return self.source_samples.get_subset(subset)

    def get_target_sample(self, idx: int) -> Sample:
        """
        Get a target sample
        """
        return self.target_samples.samples[idx]

    def get_target_samples(self, subset: list | None = None) -> BatchSamples:
        """
        Get target samples
        """
        return self.target_samples.get_subset(subset)

    def get_source_idx_for_target(self, target_idx: int) -> int:
        """
        Get index of source sample for a given target sample index
        """
        return int(self.target2source_matching_idxs[target_idx])

    def get_target_idx_for_source(self, source_idx: int) -> int:
        """
        Get index of target sample for a given source sample index
        """
        return int(self.source2target_matching_idxs[source_idx])

    def get_output_idxs(self) -> int:
        """
        Get valid output steps
        """
        return self.output_idxs

    def get_output_len(self) -> int:
        """
        Get length of output
        """
        return self.output_steps

    def get_device(self) -> str | torch.device:
        """
        Get device of tensors in the batch
        """
        return self.device

    def get_num_source_steps(self) -> int:
        """
        Get number of input/source steps from smallest of all available streams
        """
        return self.source_samples.get_num_source_steps()

    def get_num_target_steps(self) -> int:
        """
        Get number of target steps from smallest of all available streams
        """
        return self.target_samples.get_num_target_steps()
