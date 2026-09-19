import sys
import pytest

from core.common import utils

def test_py2dict_success(tmp_path):
    test_file = tmp_path / "good_module.py"
    test_file.write_text("TEST_VAR = 'value'")

    initial_sys_path = list(sys.path)

    result = utils.py2dict(str(test_file))

    assert result.get("TEST_VAR") == "value"
    assert sys.path == initial_sys_path

def test_py2dict_failure(tmp_path):
    test_file = tmp_path / "bad_module.py"
    test_file.write_text("1 / 0")

    initial_sys_path = list(sys.path)

    with pytest.raises(ZeroDivisionError):
        utils.py2dict(str(test_file))

    assert sys.path == initial_sys_path

def test_load_module_success(tmp_path):
    test_file = tmp_path / "good_load.py"
    test_file.write_text("LOADED = True")

    initial_sys_path = list(sys.path)

    utils.load_module(str(test_file))

    assert sys.path == initial_sys_path

def test_load_module_failure(tmp_path):
    test_file = tmp_path / "bad_load.py"
    test_file.write_text("import missing_package_non_existent")

    initial_sys_path = list(sys.path)

    with pytest.raises(RuntimeError):
        utils.load_module(str(test_file))

    assert sys.path == initial_sys_path
