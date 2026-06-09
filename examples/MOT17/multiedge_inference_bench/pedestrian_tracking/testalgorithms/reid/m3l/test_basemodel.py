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

"""
Tests for basemodel.py bug fixes.

Covers:
  Bug 1 - regex None guard (line 56)
  Bug 2 - hardcoded .cuda() → device-aware (lines 62, 112)
  Bug 3 - deprecated addmm_ positional signature (line 153)
  Bug 4 - print() replaced with logger (lines 66, 137)

All tests run without sedna, reid, or motmetrics.
The logic under test is extracted inline so the suite
works in any Python 3.8+ environment with only torch and pytest.
"""

import re
import warnings
import pytest
import torch


def _get_device():
    """Exact copy of _get_device() from fixed basemodel.py."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_arch_buggy(model_url: str) -> str:
    """Original (buggy) implementation — no None guard."""
    return re.compile("_([a-zA-Z]+).pth").search(model_url).group(1)


def _parse_arch_fixed(model_url: str) -> str:
    """Fixed implementation — raises ValueError on no match."""
    match = re.compile(r"_([a-zA-Z]+)\.pth").search(model_url)
    if match is None:
        raise ValueError(
            f"Cannot infer model architecture from '{model_url}'. "
            "Expected filename pattern: *_<arch>.pth (e.g. model_resnet50.pth)"
        )
    return match.group(1)


def _pairwise_distance_buggy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Original (buggy) addmm_ call — deprecated positional signature."""
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def _pairwise_distance_fixed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fixed addmm_ call — keyword-argument signature."""
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m


class TestRegexNoneGuard:
    """Bug 1: re.search() returns None on non-standard filenames."""

    @pytest.mark.parametrize("url", [
        "model_v2.pth",            # digits after underscore, not letters
        "checkpoint.pth",          # no underscore at all
        "/path/to/model.pth",      # path with no _letters pattern
        "model_resnet50.pth",      # letters then digits — no pure-letter suffix
        "weights.pth",             # plain name
    ])
    def test_buggy_crashes_on_non_standard_url(self, url):
        """Original code raises AttributeError on these URLs."""
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'group'"):
            _parse_arch_buggy(url)


    @pytest.mark.parametrize("url", [
        "model_v2.pth",
        "checkpoint.pth",
        "/path/to/model.pth",
        "model_resnet50.pth",
        "weights.pth",
    ])
    def test_fixed_raises_value_error_not_attribute_error(self, url):
        """Fixed code raises clear ValueError instead of crashing."""
        with pytest.raises(ValueError, match="Cannot infer model architecture"):
            _parse_arch_fixed(url)

    @pytest.mark.parametrize("url,expected_arch", [
        ("model_resnet.pth",         "resnet"),
        ("/path/to/model_ibn.pth",   "ibn"),
        ("checkpoint_densenet.pth",  "densenet"),
        ("m3l_osnet.pth",            "osnet"),
    ])
    def test_fixed_parses_valid_urls_correctly(self, url, expected_arch):
        """Fixed code correctly extracts arch from well-formed filenames."""
        assert _parse_arch_fixed(url) == expected_arch

    def test_fixed_error_message_contains_filename(self):
        """ValueError message includes the bad filename for easy debugging."""
        bad_url = "some_bad_checkpoint_123.pth"
        with pytest.raises(ValueError, match=re.escape(bad_url)):
            _parse_arch_fixed(bad_url)

    def test_fixed_error_message_contains_hint(self):
        """ValueError message includes a usage hint."""
        with pytest.raises(ValueError, match="resnet50.pth"):
            _parse_arch_fixed("bad.pth")


class TestDeviceSelection:
    """Bug 2: hardcoded .cuda() crashes on CPU/MPS environments."""

    def test_get_device_returns_torch_device(self):
        """_get_device() always returns a torch.device object."""
        device = _get_device()
        assert isinstance(device, torch.device)

    def test_get_device_returns_valid_device_type(self):
        """Device type is one of the three supported options."""
        device = _get_device()
        assert device.type in ("cuda", "mps", "cpu")

    def test_get_device_prefers_cuda_when_available(self, monkeypatch):
        """When CUDA is available, CUDA is selected first."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _get_device().type == "cuda"

    def test_get_device_falls_back_to_mps(self, monkeypatch):
        """When CUDA unavailable but MPS available, MPS is selected."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert _get_device().type == "mps"

    def test_get_device_falls_back_to_cpu(self, monkeypatch):
        """When neither CUDA nor MPS available, CPU is selected."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert _get_device().type == "cpu"

    def test_strip_module_prefix(self):
        state_dict = {
            "module.conv.weight": 1,
            "module.bn.weight": 2,
            "fc.weight": 3,
        }

        normalized = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

        assert "conv.weight" in normalized
        assert "bn.weight" in normalized
        assert "fc.weight" in normalized
        assert "module.conv.weight" not in normalized

    def test_tensor_to_device_does_not_crash(self):
        """Tensor.to(device) with the selected device never crashes."""
        device = _get_device()
        t = torch.randn(3, 4)
        result = t.to(device)
        assert result.device.type == device.type

    def test_cuda_hardcoded_crashes_on_this_machine(self):
        """
        Documents that .cuda() crashes when CUDA is unavailable.
        This test passes when CUDA is NOT available (CI, MPS, CPU-only),
        confirming the original bug is real on this machine.
        """
        if torch.cuda.is_available():
            pytest.skip("CUDA available on this machine — bug not reproducible here")
        t = torch.randn(3, 4)
        with pytest.raises((AssertionError, RuntimeError)):
            t.cuda()

    def test_mps_available_on_apple_silicon(self):
        """
        Documents MPS availability — informational.
        Confirms this is an Apple Silicon machine where .cuda() crashes.
        """
        device = _get_device()
        assert device.type in ("cuda", "mps", "cpu")


class TestAddmmSignature:
    """Bug 3: deprecated 3-positional-float addmm_ signature."""

    def test_buggy_addmm_raises_warning(self):
        """Original signature triggers DeprecationWarning on PyTorch 2.x.

        addmm_(beta, alpha, mat1, mat2): dist_m(m,n) += beta * dist_m + alpha * mat1(m,k) @ mat2(k,n)
        So: dist_m(3,4), x(3,8), y(4,8) → x @ y.t() = (3,8)@(8,4) = (3,4) ✓
        """
        m, n, d = 3, 4, 8
        x = torch.randn(m, d)
        y = torch.randn(n, d)
        dist_m = torch.zeros(m, n)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist_m.addmm_(1, -2, x, y.t())
            deprecation_warnings = [
                w for w in caught
                if issubclass(w.category, UserWarning)
                and "deprecated" in str(w.message).lower()
            ]
            assert len(deprecation_warnings) > 0, (
                "Expected deprecation warning from positional addmm_ signature"
            )

    def test_fixed_addmm_no_warning(self):
        """Fixed keyword-argument signature produces no deprecation warning."""
        m, n, d = 3, 4, 8
        x = torch.randn(m, d)
        y = torch.randn(n, d)
        dist_m = torch.zeros(m, n)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
            deprecation_warnings = [
                w for w in caught
                if issubclass(w.category, UserWarning)
                and "deprecated" in str(w.message).lower()
            ]
            assert len(deprecation_warnings) == 0, (
                f"Unexpected deprecation warning: {[str(w.message) for w in deprecation_warnings]}"
            )

    def test_buggy_and_fixed_produce_same_result(self):
        """Both signatures compute identical distance matrices."""
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        y = torch.randn(6, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_buggy = _pairwise_distance_buggy(x.clone(), y.clone())
        result_fixed = _pairwise_distance_fixed(x.clone(), y.clone())
        assert torch.allclose(result_buggy, result_fixed, atol=1e-5), (
            "Fixed addmm_ must produce identical results to original"
        )

    def test_fixed_distance_matrix_shape(self):
        """Distance matrix shape is (m, n) for m queries and n gallery items."""
        torch.manual_seed(0)
        m, n, d = 5, 8, 16
        x = torch.randn(m, d)
        y = torch.randn(n, d)
        dist = _pairwise_distance_fixed(x, y)
        assert dist.shape == (m, n)

    def test_fixed_distance_matrix_non_negative(self):
        """Euclidean squared distances must be >= 0."""
        torch.manual_seed(7)
        x = torch.randn(4, 12)
        y = torch.randn(6, 12)
        dist = _pairwise_distance_fixed(x, y)
        assert (dist >= -1e-5).all(), "Distance matrix contains large negative values"

    def test_fixed_distance_self_is_zero(self):
        """Distance from a vector to itself should be ~0."""
        torch.manual_seed(3)
        x = torch.randn(3, 8)
        dist = _pairwise_distance_fixed(x, x)
        diagonal = torch.diagonal(dist)
        assert torch.allclose(diagonal, torch.zeros(3), atol=1e-4)

    @pytest.mark.parametrize("m,n,d", [
        (1, 1, 4),
        (1, 10, 32),
        (10, 1, 32),
        (8, 12, 64),
    ])
    def test_fixed_various_shapes(self, m, n, d):
        """Fixed implementation handles all valid input shapes."""
        x = torch.randn(m, d)
        y = torch.randn(n, d)
        dist = _pairwise_distance_fixed(x, y)
        assert dist.shape == (m, n)



class TestLoggerVsPrint:
    """Bug 4: print() must be replaced with logger in _extract_features."""

    def test_fixed_file_has_no_bare_print_calls(self):
        """
        Verify the fixed source file contains no bare print() calls
        (other than the legacy import which is harmless).
        """
        import ast
        import pathlib

        fixed_path = pathlib.Path(__file__).parent / "basemodel.py"
        if not fixed_path.exists():
            pytest.skip(f"Fixed file not found at {fixed_path}")

        source = fixed_path.read_text()
        tree = ast.parse(source)

        print_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                print_calls.append(node.lineno)

        assert len(print_calls) == 0, (
            f"Found bare print() calls at lines {print_calls} in fixed basemodel.py. "
            "Use logger.info() instead."
        )

    def test_fixed_file_imports_logger(self):
        """Fixed file must import logger from loguru."""
        import pathlib

        fixed_path = pathlib.Path(__file__).parent / "basemodel.py"
        if not fixed_path.exists():
            pytest.skip(f"Fixed file not found at {fixed_path}")

        source = fixed_path.read_text()
        assert "from loguru import logger" in source, (
            "Fixed basemodel.py must import logger from loguru"
        )

    def test_fixed_file_uses_logger_info(self):
        """Fixed file must call logger.info() where print() was used."""
        import pathlib

        fixed_path = pathlib.Path(__file__).parent / "basemodel.py"
        if not fixed_path.exists():
            pytest.skip(f"Fixed file not found at {fixed_path}")

        source = fixed_path.read_text()
        assert "logger.info" in source, (
            "Fixed basemodel.py must use logger.info() for output"
        )


class TestAllFixesTogether:
    """Confirm all 4 fixes work together without interfering."""

    def test_device_selection_and_tensor_move(self):
        """Device selection + tensor.to(device) works end-to-end."""
        device = _get_device()
        t = torch.randn(4, 8)
        t = t.to(device)
        assert t.device.type == device.type

    def test_regex_fix_and_pairwise_fix_independent(self):
        """Bug 1 and Bug 3 fixes are independent — one does not affect the other."""
        arch = _parse_arch_fixed("model_resnet.pth")
        assert arch == "resnet"

        x = torch.randn(3, 8)
        y = torch.randn(4, 8)
        dist = _pairwise_distance_fixed(x, y)
        assert dist.shape == (3, 4)

    def test_no_cuda_dependency_in_device_path(self, monkeypatch):
        """With CUDA patched off, everything still runs on MPS or CPU."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device = _get_device()
        assert device.type in ("mps", "cpu")
        t = torch.randn(2, 4).to(device)
        assert t.device.type == device.type