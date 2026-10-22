import pathlib

import pytest

from weathergen.datasets.data_reader_base import is_remote_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://example.com/data.zarr", True),
        ("s3://bucket/data.zarr", True),
        ("/data/data.zarr", False),
        (pathlib.Path("/data/data.zarr"), False),
    ],
)
def test_is_remote_path(path, expected):
    assert is_remote_path(path) == expected
