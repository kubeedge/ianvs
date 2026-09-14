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

"""Tests for compute device capability reporting.

The cases below stand in a fake torch module so that every capability tier can
be exercised on any machine, including CI runners with no GPU. Running the real
thing would only ever cover whichever tier the runner happens to be.
"""

import sys
import types

import pytest

from core.common.device import (
    check_min_capability,
    device_profile,
    parse_capability,
    supports_bf16,
)


def fake_torch(available=True, capability=(8, 6), name="Fake GPU", memory_gb=16):
    """Build a torch stand-in exposing only what device_profile() reads."""
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: 0,
        get_device_capability=lambda index: capability,
        get_device_properties=lambda index: types.SimpleNamespace(
            name=name,
            total_memory=int(memory_gb * 1024 ** 3),
        ),
    )
    torch.cuda = cuda
    return torch


@pytest.fixture
def install_torch(monkeypatch):
    """Register a torch stand-in for the duration of one test."""

    def install(module):
        monkeypatch.setitem(sys.modules, "torch", module)

    return install


def test_reports_cpu_when_torch_is_not_installed(monkeypatch):
    """core does not depend on torch, so its absence is not an error."""
    monkeypatch.setitem(sys.modules, "torch", None)

    profile = device_profile()

    assert profile["device"] == "cpu"
    assert profile["capability"] is None
    assert profile["bf16_native"] is False


def test_reports_cpu_when_no_cuda_device_is_available(install_torch):
    install_torch(fake_torch(available=False))

    profile = device_profile()

    assert profile["device"] == "cpu"
    assert profile["bf16_native"] is False


def test_pre_ampere_device_does_not_report_native_bf16(install_torch):
    """A GTX 1080 is compute capability 6.1: bfloat16 there is emulated."""
    install_torch(fake_torch(capability=(6, 1), name="NVIDIA GeForce GTX 1080", memory_gb=8))

    profile = device_profile()

    assert profile["device"] == "cuda"
    assert profile["capability"] == (6, 1)
    assert profile["bf16_native"] is False
    assert profile["total_memory_gb"] == 8.0


def test_turing_device_does_not_report_native_bf16(install_torch):
    """Turing (7.5) clears vLLM's floor but still has no native bfloat16."""
    install_torch(fake_torch(capability=(7, 5)))

    assert device_profile()["bf16_native"] is False


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (9, 0)])
def test_ampere_and_newer_report_native_bf16(install_torch, capability):
    install_torch(fake_torch(capability=capability))

    assert device_profile()["bf16_native"] is True


def test_supports_bf16_tracks_the_profile(install_torch):
    install_torch(fake_torch(capability=(6, 1)))
    assert supports_bf16() is False

    install_torch(fake_torch(capability=(8, 6)))
    assert supports_bf16() is True


def test_emulation_is_reported_on_every_affected_model_load(install_torch, caplog):
    """Silence is the defect this module exists to remove, so the fallback is logged.

    Every occurrence, not just the first. A benchmarking job runs its test cases in
    one process, so suppressing repeats would announce the first affected
    measurement and pass silently over the rest. Reported in review of #957.
    """
    install_torch(fake_torch(capability=(6, 1), name="NVIDIA GeForce GTX 1080"))

    with caplog.at_level("WARNING"):
        assert supports_bf16() is False
        assert supports_bf16() is False
        assert supports_bf16() is False

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 3
    assert all("6.1" in w.getMessage() for w in warnings)
    assert all("GTX 1080" in w.getMessage() for w in warnings)


def test_no_warning_when_bf16_is_native(install_torch, caplog):
    install_torch(fake_torch(capability=(8, 6)))

    with caplog.at_level("WARNING"):
        assert supports_bf16() is True

    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


class TestParseCapability:
    """Normalising what a yaml author is likely to write."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("8.0", (8, 0)),
            ("7.5", (7, 5)),
            ("8.6", (8, 6)),
            (8.0, (8, 0)),
            (8, (8, 0)),
            ("  8.0  ", (8, 0)),
        ],
    )
    def test_accepts_the_usual_spellings(self, value, expected):
        assert parse_capability(value) == expected

    @pytest.mark.parametrize("value", ["eight", "8.0.1", "", "8.x", "-8.0"])
    def test_rejects_anything_else_with_an_actionable_message(self, value):
        with pytest.raises(ValueError, match="min_compute_capability"):
            parse_capability(value)


class TestCheckMinCapability:
    """A declared floor is enforced only where it can be."""

    def test_no_declared_floor_is_not_an_error(self, install_torch):
        install_torch(fake_torch(capability=(6, 1)))

        check_min_capability(None)

    def test_device_below_the_floor_is_refused(self, install_torch):
        install_torch(fake_torch(capability=(6, 1), name="NVIDIA GeForce GTX 1080"))

        with pytest.raises(RuntimeError) as excinfo:
            check_min_capability((8, 0))

        message = str(excinfo.value)
        assert "8.0" in message
        assert "6.1" in message
        assert "GTX 1080" in message

    @pytest.mark.parametrize("capability", [(8, 0), (8, 6), (9, 0)])
    def test_device_meeting_the_floor_passes(self, install_torch, capability):
        install_torch(fake_torch(capability=capability))

        check_min_capability((8, 0))

    def test_cpu_run_is_reported_but_allowed(self, install_torch, caplog):
        """`use_gpu: false` asks for cpu; a declared floor must not veto that."""
        install_torch(fake_torch(available=False))

        with caplog.at_level("WARNING"):
            check_min_capability((8, 0))

        assert any("min_compute_capability" in r.getMessage() for r in caplog.records)
