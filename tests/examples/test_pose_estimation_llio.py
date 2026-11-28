# Copyright 2022 The KubeEdge Authors.
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

import importlib.util
from pathlib import Path
import sys
from unittest.mock import MagicMock
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LLIO_BASEMODEL_PATH = (
    REPO_ROOT
    / "examples"
    / "industrialEI"
    / "pose-estimation-llio"
    / "testalgorithms"
    / "llio_fusion"
    / "basemodel.py"
)


def _load_llio_basemodel():
    # Mock optional heavy dependencies if not installed in CI environment
    for mod_name in ("pykitti", "llio_estimator"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    spec = importlib.util.spec_from_file_location("llio_basemodel", str(LLIO_BASEMODEL_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BaseModel


@pytest.fixture(scope="module")
def llio_basemodel_class():
    return _load_llio_basemodel()


@pytest.mark.parametrize(
    "sequence_path,expected_date,expected_drive",
    [
        ("2011_09_26/2011_09_26_drive_0001_sync", "2011_09_26", "0001"),
        (r"2011_09_26\2011_09_26_drive_0001_sync", "2011_09_26", "0001"),
        ("/dataset/kitti/2011_09_26/2011_09_26_drive_0005_sync", "2011_09_26", "0005"),
        (r"C:\data\kitti\2011_09_26\2011_09_26_drive_0009_sync", "2011_09_26", "0009"),
        (r"mixed/path\2011_09_30/2011_09_30_drive_0018_extract", "2011_09_30", "0018"),
        ("2011_10_03/2011_10_03_drive_0027", "2011_10_03", "0027"),
    ],
)
def test_extract_sequence_parts_cross_platform(llio_basemodel_class, sequence_path, expected_date, expected_drive):
    dataname, datadrive = llio_basemodel_class._extract_sequence_parts(sequence_path)
    assert dataname == expected_date
    assert datadrive == expected_drive


@pytest.mark.parametrize(
    "invalid_path",
    [
        "invalid_sequence_path",
        "2011_09_26/without_drive_token",
        "drive_0001_sync/no_date_here",
        "",
        "/just/some/random/folders/",
    ],
)
def test_extract_sequence_parts_invalid_paths(llio_basemodel_class, invalid_path):
    dataname, datadrive = llio_basemodel_class._extract_sequence_parts(invalid_path)
    assert not dataname or not datadrive
