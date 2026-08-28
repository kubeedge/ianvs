from core.common import utils


def test_is_local_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.touch()
    assert utils.is_local_file(str(test_file)) is True
    assert utils.is_local_file(str(tmp_path / "nonexistent.txt")) is False
    assert utils.is_local_file(str(tmp_path)) is False


def test_is_local_dir(tmp_path):
    assert utils.is_local_dir(str(tmp_path)) is True
    assert utils.is_local_dir(str(tmp_path / "nonexistent")) is False
    test_file = tmp_path / "test.txt"
    test_file.touch()
    assert utils.is_local_dir(str(test_file)) is False


def test_get_file_format():
    assert utils.get_file_format("data/metadata.json") == "jsonforllm"
    assert utils.get_file_format("config.yaml") == "yaml"
    assert utils.get_file_format("script.py") == "py"
    assert utils.get_file_format("noextension") == ""


def test_parse_kwargs():
    def dummy_func(a, b, c=1):
        pass

    kwargs = {"a": 10, "b": 20, "c": 30, "d": 40}
    parsed = utils.parse_kwargs(dummy_func, **kwargs)
    assert parsed == {"a": 10, "b": 20, "c": 30}

    # Test with non-callable
    assert utils.parse_kwargs("not a function", **kwargs) == kwargs

    # Test with func that accepts **kwargs
    def kwarg_func(a, **kwargs):
        pass

    assert utils.parse_kwargs(kwarg_func, **kwargs) == kwargs
