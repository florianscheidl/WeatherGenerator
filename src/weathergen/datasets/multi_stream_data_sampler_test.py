from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from weathergen.datasets.data_reader_base import ReaderData
from weathergen.datasets.multi_stream_data_sampler import (
    _empty_role_data,
    _stream_read_roles,
    _tokenize_role_windows,
    collect_datasources,
)


def _reader(source_channels: int, target_channels: int, geoinfo_size: int = 4):
    return SimpleNamespace(
        source_idx=list(range(source_channels)),
        target_idx=list(range(target_channels)),
        get_geoinfo_size=lambda: geoinfo_size,
    )


def test_stream_read_roles_cover_all_role_combinations():
    assert _stream_read_roles([_reader(2, 3)]) == (True, True)
    assert _stream_read_roles([_reader(2, 0)]) == (True, False)
    assert _stream_read_roles([_reader(0, 3)]) == (False, True)
    assert _stream_read_roles([_reader(0, 0)]) == (False, False)


def test_stream_read_roles_are_active_when_any_reader_has_channels():
    readers = [_reader(0, 2), _reader(3, 0)]

    assert _stream_read_roles(readers) == (True, True)


def test_empty_role_data_is_true_empty_not_spoofed():
    data = _empty_role_data([_reader(0, 0, geoinfo_size=7)])

    assert data.is_empty()
    assert data.is_spoof is False
    assert data.coords.shape == (0, 2)
    assert data.geoinfos.shape == (0, 7)
    assert data.data.shape == (0, 0)
    assert data.datetimes.dtype == np.dtype("datetime64[ns]")


def test_inactive_role_skips_tokenizer():
    tokenizer = Mock()
    data = [_empty_role_data([_reader(0, 0)]) for _ in range(8)]

    tokens = _tokenize_role_windows(tokenizer, {}, data, False, role_active=False)

    assert tokens == [(None, None)] * 8
    tokenizer.get_tokens_windows.assert_not_called()


def test_active_role_delegates_to_tokenizer():
    tokenizer = Mock()
    tokenizer.get_tokens_windows.return_value = [([1], [1])]
    data = [_empty_role_data([_reader(1, 1)])]
    stream_info = {"token_size": 8}

    tokens = _tokenize_role_windows(tokenizer, stream_info, data, True, role_active=True)

    assert tokens == [([1], [1])]
    tokenizer.get_tokens_windows.assert_called_once_with(stream_info, data, True)


def test_collect_datasources_skips_reader_with_no_channels_for_side():
    active_data = ReaderData(
        coords=np.zeros((2, 2), dtype=np.float32),
        geoinfos=np.zeros((2, 1), dtype=np.float32),
        data=np.ones((2, 1), dtype=np.float32),
        datetimes=np.zeros(2, dtype="datetime64[ns]"),
    )
    inactive = Mock(
        source_idx=[],
        target_idx=[0],
        stream_info={"shuffle_source": False},
    )
    active = Mock(
        source_idx=[0],
        target_idx=[],
        stream_info={"shuffle_source": False},
    )
    active.get_source.return_value = active_data
    active.normalize_source_channels.side_effect = lambda data: data
    active.normalize_geoinfos.side_effect = lambda data: data

    result = collect_datasources([inactive, active], 3, "source", np.random.default_rng(1))

    inactive.get_source.assert_not_called()
    active.get_source.assert_called_once_with(3)
    assert result.data.shape == (2, 1)
