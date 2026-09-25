import pytest

from core.testcasecontroller.algorithm.algorithm import Algorithm


def make_config(name="test", paradigm_type="singletasklearning"):
    return {
        "algorithm": {
            "name": name,
            "paradigm_type": paradigm_type,
        }
    }


def test_algorithm_rejects_non_string_name():
    with pytest.raises(ValueError, match="algorithm name"):
        Algorithm("test", make_config(name=123))


def test_algorithm_rejects_non_string_paradigm_type():
    with pytest.raises(ValueError, match="algorithm paradigm"):
        Algorithm("test", make_config(paradigm_type=123))
