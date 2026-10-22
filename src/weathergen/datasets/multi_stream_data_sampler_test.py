from weathergen.datasets.multi_stream_data_sampler import _join_data_path, _resolve_data_path

URL = "https://example.com/datasets/v1"


def test_join_data_path_url():
    assert _join_data_path(URL, "data.zarr") == f"{URL}/data.zarr"
    assert _join_data_path(URL + "/", "data.zarr") == f"{URL}/data.zarr"


def test_join_data_path_local(tmp_path):
    assert _join_data_path(tmp_path, "data.zarr") == tmp_path / "data.zarr"
    assert _join_data_path(str(tmp_path), "data.zarr") == tmp_path / "data.zarr"


def test_resolve_data_path_url_filename(tmp_path):
    fname = "https://example.com/data.zarr"
    assert _resolve_data_path(fname, [tmp_path]) == fname


def test_resolve_data_path_existing_filename(tmp_path):
    fname = tmp_path / "data.zarr"
    fname.mkdir()
    assert _resolve_data_path(fname, [URL]) == fname


def test_resolve_data_path_first_local_path_with_dataset(tmp_path):
    paths = [tmp_path / name for name in ("a", "b", "c")]
    for path in paths[1:]:
        (path / "data.zarr").mkdir(parents=True)

    assert _resolve_data_path("data.zarr", paths) == paths[1] / "data.zarr"


def test_resolve_data_path_url_after_local_paths(tmp_path):
    assert _resolve_data_path("data.zarr", [tmp_path, URL]) == f"{URL}/data.zarr"


def test_resolve_data_path_url_before_local_path(tmp_path):
    (tmp_path / "data.zarr").mkdir()
    # the existence of remote datasets is not checked, so the URL comes first
    assert _resolve_data_path("data.zarr", [URL, tmp_path]) == f"{URL}/data.zarr"


def test_resolve_data_path_not_found(tmp_path):
    assert _resolve_data_path("data.zarr", [tmp_path]) is None
    assert _resolve_data_path("data.zarr", []) is None
