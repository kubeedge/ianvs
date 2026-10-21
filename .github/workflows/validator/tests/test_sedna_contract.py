# Copyright 2026 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for module ②, the Sedna Contract Check (services/sedna_contract.py).

The symbol checks inspect the Sedna installed for the Python running the
tests, so they need the vendored sedna-0.6.0.1 wheel installed.
"""

import importlib.metadata
import sys

import pytest

from services import sedna_contract as sc

try:
    INSTALLED = importlib.metadata.version("sedna")
except importlib.metadata.PackageNotFoundError:
    INSTALLED = None

needs_sedna = pytest.mark.skipif(INSTALLED is None, reason="sedna is not installed")


def _repo(tmp_path, wheel_version=INSTALLED or "0.6.0.1", core_source=""):
    if wheel_version:
        wheels = tmp_path / sc.VENDORED_WHEEL_DIR
        wheels.mkdir(parents=True)
        (wheels / f"sedna-{wheel_version}-py3-none-any.whl").write_bytes(b"")
    core = tmp_path / "core"
    core.mkdir()
    (core / "module.py").write_text(core_source, encoding="utf-8")
    return tmp_path


def _by_name(results):
    return {result.name: result for result in results}


def test_core_symbols_are_extracted_from_imports(tmp_path):
    repo = _repo(tmp_path, core_source=(
        "import os\n"
        "import sedna.common.log\n"
        "from sedna.core.lifelong_learning import LifelongLearning\n"
        "from sedna.datasources import (BaseDataSource, TxtDataParse)\n"
        "from .local import thing\n"
    ))

    assert sc.core_sedna_symbols(repo) == [
        ("sedna.common.log", ""),
        ("sedna.core.lifelong_learning", "LifelongLearning"),
        ("sedna.datasources", "BaseDataSource"),
        ("sedna.datasources", "TxtDataParse"),
    ]


def test_vendored_versions_read_from_wheel_names(tmp_path):
    repo = _repo(tmp_path, wheel_version="0.6.0.1")

    assert sc.vendored_versions(repo) == ["0.6.0.1"]


def test_missing_wheel_fails(tmp_path):
    results = _by_name(sc.check_sedna_contract(_repo(tmp_path, wheel_version=None)))

    assert results["Vendored Sedna wheel present"].status == "FAIL"


@needs_sedna
def test_matching_install_and_resolvable_symbols_pass(tmp_path):
    repo = _repo(tmp_path, core_source="from sedna.core.lifelong_learning import LifelongLearning\n")

    results = _by_name(sc.check_sedna_contract(
        repo, python=sys.executable,
        extra_symbols=[("sedna.core.lifelong_learning", "LifelongLearning.update")]))

    assert results["Vendored Sedna wheel present"].status == "PASS"
    assert results["Installed Sedna matches vendored wheel"].status == "PASS"
    assert results["Sedna symbols used by core resolve"].status == "PASS"


@needs_sedna
def test_version_mismatch_fails(tmp_path):
    repo = _repo(tmp_path, wheel_version="9.9.9")

    results = _by_name(sc.check_sedna_contract(repo, python=sys.executable))

    assert results["Installed Sedna matches vendored wheel"].status == "FAIL"
    assert "9.9.9" in results["Installed Sedna matches vendored wheel"].message


@needs_sedna
def test_unresolvable_symbols_fail_and_are_listed(tmp_path):
    repo = _repo(tmp_path, core_source=(
        "from sedna.core.lifelong_learning import LifelongLearning\n"
        "from sedna.core.lifelong_learning import NoSuchClass\n"
        "from sedna.no_such_module import Thing\n"
    ))

    results = _by_name(sc.check_sedna_contract(
        repo, python=sys.executable,
        extra_symbols=[("sedna.core.lifelong_learning", "LifelongLearning.my_inference")]))

    check = results["Sedna symbols used by core resolve"]
    assert check.status == "FAIL"
    assert len(check.details) == 3
    assert any("NoSuchClass" in detail for detail in check.details)
    assert any("my_inference" in detail for detail in check.details)


def test_probe_that_cannot_run_is_an_error(tmp_path):
    repo = _repo(tmp_path)

    results = _by_name(sc.check_sedna_contract(repo, python=str(tmp_path / "no-such-python")))

    assert results["Installed Sedna matches vendored wheel"].status == "ERROR"
    assert "Sedna symbols used by core resolve" not in results
