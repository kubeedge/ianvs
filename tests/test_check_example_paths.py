import os
import sys
from pathlib import Path
import yaml
REPO_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT_DIR / "scripts"))
from check_example_paths import (
    resolve_path,
    check_benchmarkingjob,
    load_allowlist
)
def test_resolve_path():
    repo_root = Path("/fake/repo")
    assert resolve_path("./test/path.yaml", repo_root) == repo_root / "test/path.yaml"
    assert resolve_path("test/path.yaml", repo_root) == repo_root / "test/path.yaml"
    if os.name == 'nt':
        assert resolve_path("C:\\absolute\\path", repo_root) == Path("C:\\absolute\\path")
    else:
        assert resolve_path("/absolute/path", repo_root) == Path("/absolute/path")
def create_yaml_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        yaml.dump(data, file_handle)
def test_valid_config_chain(tmp_path):
    repo_root = tmp_path
    job_file = repo_root / "examples" / "benchmarkingjob.yaml"
    job_data = {
        "benchmarkingjob": {
            "testenv": "./examples/testenv.yaml",
            "test_object": {
                "algorithms": [
                    {"url": "./examples/algorithm.yaml"}
                ]
            }
        }
    }
    create_yaml_file(job_file, job_data)
    testenv_file = repo_root / "examples" / "testenv.yaml"
    testenv_data = {
        "testenv": {
            "metrics": [{"url": "./examples/metric.py"}],
            "model_eval": {"model_metric": {"url": "./examples/model_metric.py"}}
        }
    }
    create_yaml_file(testenv_file, testenv_data)
    algorithm_file = repo_root / "examples" / "algorithm.yaml"
    algorithm_data = {
        "algorithm": {
            "modules": [{"url": "./examples/module.py"}]
        }
    }
    create_yaml_file(algorithm_file, algorithm_data)
    (repo_root / "examples" / "metric.py").touch()
    (repo_root / "examples" / "model_metric.py").touch()
    (repo_root / "examples" / "module.py").touch()
    broken_paths = []
    check_benchmarkingjob(job_file, repo_root, broken_paths)
    assert len(broken_paths) == 0
def test_broken_testenv_path(tmp_path):
    repo_root = tmp_path
    job_file = repo_root / "examples" / "benchmarkingjob.yaml"
    job_data = {
        "benchmarkingjob": {
            "testenv": "./examples/missing_testenv.yaml",
        }
    }
    create_yaml_file(job_file, job_data)
    broken_paths = []
    check_benchmarkingjob(job_file, repo_root, broken_paths)
    assert len(broken_paths) == 1
    assert broken_paths[0]["field"] == "testenv"
    assert broken_paths[0]["path"] == "./examples/missing_testenv.yaml"
def test_broken_module_url(tmp_path):
    repo_root = tmp_path
    job_file = repo_root / "examples" / "benchmarkingjob.yaml"
    job_data = {
        "benchmarkingjob": {
            "test_object": {
                "algorithms": [
                    {"url": "./examples/algorithm.yaml"}
                ]
            }
        }
    }
    create_yaml_file(job_file, job_data)
    algorithm_file = repo_root / "examples" / "algorithm.yaml"
    algorithm_data = {
        "algorithm": {
            "modules": [{"url": "./examples/missing_module.py"}]
        }
    }
    create_yaml_file(algorithm_file, algorithm_data)
    broken_paths = []
    check_benchmarkingjob(job_file, repo_root, broken_paths)
    assert len(broken_paths) == 1
    assert broken_paths[0]["field"] == "modules[0].url"
    assert broken_paths[0]["path"] == "./examples/missing_module.py"
def test_absolute_path_detected(tmp_path):
    repo_root = tmp_path
    job_file = repo_root / "examples" / "benchmarkingjob.yaml"
    if os.name == 'nt':
        abs_path = "C:\\missing_testenv.yaml"
    else:
        abs_path = "/missing_testenv.yaml"
    job_data = {
        "benchmarkingjob": {
            "testenv": abs_path,
        }
    }
    create_yaml_file(job_file, job_data)
    broken_paths = []
    check_benchmarkingjob(job_file, repo_root, broken_paths)
    assert len(broken_paths) == 1
    assert broken_paths[0]["field"] == "testenv"
    assert broken_paths[0]["path"] == abs_path
def test_missing_optional_model_eval(tmp_path):
    repo_root = tmp_path
    job_file = repo_root / "examples" / "benchmarkingjob.yaml"
    job_data = {
        "benchmarkingjob": {
            "testenv": "./examples/testenv_no_modeleval.yaml",
        }
    }
    create_yaml_file(job_file, job_data)
    testenv_file = repo_root / "examples" / "testenv_no_modeleval.yaml"
    testenv_data = {
        "testenv": {
            "metrics": []
        }
    }
    create_yaml_file(testenv_file, testenv_data)
    broken_paths = []
    check_benchmarkingjob(job_file, repo_root, broken_paths)
    assert len(broken_paths) == 0
def test_load_allowlist(tmp_path):
    allowlist_file = tmp_path / "known_broken_paths.txt"
    with open(allowlist_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("# comment\n")
        file_handle.write("examples/job.yaml:testenv:./broken.yaml\n")
        file_handle.write("\n")
        file_handle.write("examples/job2.yaml:testenv:./broken2.yaml\n")
    entries = load_allowlist(allowlist_file)
    assert len(entries) == 2
    assert "examples/job.yaml:testenv:./broken.yaml" in entries
    assert "examples/job2.yaml:testenv:./broken2.yaml" in entries
