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

"""
Tests for the simulation sandbox.

Every test that asserts a *fix* names the breakage identifier from the proposal
(B1..B15), so a reviewer can trace each assertion back to the defect it guards.

Run with::

    pytest tests/simulation -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.common.constant import IsolationLevel, SandboxMode  # noqa: E402
from core.testcasecontroller.simulation.config import (  # noqa: E402
    ResourceQuota,
    SandboxConfig,
    parse_memory,
)
from core.testcasecontroller.simulation.profiler import (  # noqa: E402
    ProfileResult,
    detect_oom_kill,
)
from core.testcasecontroller.simulation.simulation import (  # noqa: E402
    SEDNA_MAX_CLOUD_WORKER_NODES,
    SEDNA_MAX_EDGE_NODES,
    Simulation,
)

VALID = {
    "cloud_number": 1,
    "edge_number": 2,
    "cluster_name": "sedna-mini",
    "kubeedge_version": "v1.8.0",
    "sedna_version": "v0.4.3",
}


# ---------------------------------------------------------------- compatibility

class TestBackwardCompatibility:
    """The 2022 schema must keep working; this feature is a restoration."""

    def test_legacy_config_still_parses(self):
        sim = Simulation(dict(VALID))
        assert sim.cloud_number == 1
        assert sim.edge_number == 2
        assert sim.cluster_name == "sedna-mini"

    def test_class_name_and_attributes_unchanged(self):
        """benchmarkingjob.py dispatches on str.lower(Simulation.__name__)."""
        assert Simulation.__name__ == "Simulation"
        sim = Simulation(dict(VALID))
        for attribute in ("cloud_number", "edge_number", "cluster_name",
                          "kubeedge_version", "sedna_version"):
            assert hasattr(sim, attribute)

    def test_absent_sandbox_block_is_disabled(self):
        """No sandbox block means the default execution path, untouched."""
        assert SandboxConfig.disabled().enabled is False
        assert SandboxConfig().enabled is False


# --------------------------------------------------------------- simulation cfg

class TestSimulationValidation:
    """Each test corresponds to one verified defect in the 2022 implementation."""

    def test_b2_bool_rejected_as_node_count(self):
        """B2: isinstance(True, int) is True, so booleans slipped through."""
        with pytest.raises(ValueError, match="must be int type"):
            Simulation({**VALID, "edge_number": True})

    def test_b3_unknown_key_rejected(self):
        """B3: 'edge_nodes: 5' silently produced edge_number == 0."""
        config = {k: v for k, v in VALID.items() if k != "edge_number"}
        config["edge_nodes"] = 5
        with pytest.raises(ValueError, match="unknown field"):
            Simulation(config)

    def test_b4_empty_cluster_name_rejected(self):
        """B4: an empty name produced a malformed 'CLUSTER_NAME=' command."""
        with pytest.raises(ValueError, match="non-empty string"):
            Simulation({**VALID, "cluster_name": ""})

    @pytest.mark.parametrize("edge", [SEDNA_MAX_EDGE_NODES + 1, 10, 100])
    def test_b12_edge_count_ceiling_enforced(self, edge):
        """
        B12: the Sedna all-in-one backend caps edge nodes at 3.

        The original class accepted any integer, so the headline capability --
        large-scale node simulation -- failed inside a piped shell script with
        an opaque error, after every Ianvs-side check had passed.
        """
        with pytest.raises(ValueError, match="out of range"):
            Simulation({**VALID, "edge_number": edge})

    @pytest.mark.parametrize("cloud", [SEDNA_MAX_CLOUD_WORKER_NODES + 1, 5])
    def test_b12_cloud_count_ceiling_enforced(self, cloud):
        with pytest.raises(ValueError, match="out of range"):
            Simulation({**VALID, "cloud_number": cloud})

    def test_zero_edge_nodes_rejected(self):
        """A cluster with no edge node cannot simulate edge-cloud synergy."""
        with pytest.raises(ValueError, match="out of range"):
            Simulation({**VALID, "edge_number": 0})

    @pytest.mark.parametrize("name", ["My_Cluster", "-leading", "UPPER", "a b"])
    def test_invalid_rfc1123_names_rejected(self, name):
        """kind derives container names from this; invalid labels fail late."""
        with pytest.raises(ValueError, match="RFC 1123"):
            Simulation({**VALID, "cluster_name": name})

    def test_literal_latest_normalised_to_empty(self):
        """
        The all-in-one script resolves the newest release only when the variable
        is empty; the literal string 'latest' 404s on the release asset.
        """
        sim = Simulation({**VALID, "kubeedge_version": "latest"})
        assert sim.kubeedge_version == ""

    def test_ceilings_match_documented_backend_limits(self):
        assert SEDNA_MAX_CLOUD_WORKER_NODES == 2
        assert SEDNA_MAX_EDGE_NODES == 3


# ------------------------------------------------------------------ quota parse

class TestMemoryParsing:

    @pytest.mark.parametrize("text,expected", [
        ("2Gi", 2 * 2 ** 30),
        ("512Mi", 512 * 2 ** 20),
        ("1GB", 10 ** 9),
        ("1024", 1024),
        (2 ** 30, 2 ** 30),
    ])
    def test_valid_quantities(self, text, expected):
        assert parse_memory(text) == expected

    @pytest.mark.parametrize("text", ["2Zi", "abc", "", None])
    def test_invalid_or_empty(self, text):
        if text in ("", None):
            assert parse_memory(text) is None
        else:
            with pytest.raises(ValueError):
                parse_memory(text)

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="boolean"):
            parse_memory(True)


class TestResourceQuota:

    def test_unbounded_by_default(self):
        assert ResourceQuota().is_unbounded()

    def test_cpus_cannot_exceed_host(self):
        """A quota larger than the host constrains nothing; fail loudly."""
        with pytest.raises(ValueError, match="exceeds the host CPU count"):
            ResourceQuota(cpus=(os.cpu_count() or 1) + 64)

    @pytest.mark.parametrize("cpus", [0, -1])
    def test_non_positive_cpus_rejected(self, cpus):
        with pytest.raises(ValueError, match="must be positive"):
            ResourceQuota(cpus=cpus)

    def test_serialisable(self):
        quota = ResourceQuota(memory="1Gi", cpus=1, timeout=60)
        assert quota.as_dict() == {
            "memory_bytes": 2 ** 30, "cpus": 1.0, "timeout": 60,
        }


# ---------------------------------------------------------------- sandbox cfg

class TestSandboxConfig:

    def test_full_config(self):
        config = SandboxConfig({
            "enabled": True, "mode": "process", "isolation": "venv",
            "resources": {"memory": "2Gi", "cpus": 1, "timeout": 600},
            "metrics": ["peak_memory", "wall_time"],
        })
        assert config.enabled
        assert config.mode == SandboxMode.PROCESS.value
        assert config.isolation == IsolationLevel.VENV.value
        assert config.quota.memory_bytes == 2 * 2 ** 30

    @pytest.mark.parametrize("mode", ["kubernetes", "docker", "vm"])
    def test_unknown_mode_rejected(self, mode):
        with pytest.raises(ValueError, match="not supported"):
            SandboxConfig({"enabled": True, "mode": mode})

    def test_unknown_top_level_key_rejected(self):
        """A typo must not be silently ignored, as it was in the 2022 parser."""
        with pytest.raises(ValueError, match="unknown field"):
            SandboxConfig({"enabled": True, "enabeld": True})

    def test_unknown_resource_key_rejected(self):
        with pytest.raises(ValueError, match="unknown field"):
            SandboxConfig({"enabled": True, "resources": {"ram": "2Gi"}})

    def test_all_modes_constructible(self):
        for mode in SandboxMode:
            assert SandboxConfig(
                {"enabled": True, "mode": mode.value}
            ).mode == mode.value


# ------------------------------------------------------------------- profiler

class TestOomDetection:

    @pytest.mark.parametrize("code", [-9, 137])
    def test_sigkill_codes_flagged(self, code):
        assert detect_oom_kill(code) is True

    @pytest.mark.parametrize("text", [
        "Out of memory: Killed process 1234",
        "MemoryError",
        "container was OOMKilled",
    ])
    def test_stderr_markers_flagged(self, text):
        assert detect_oom_kill(1, text) is True

    def test_clean_exit_not_flagged(self):
        assert detect_oom_kill(0) is False

    def test_ordinary_failure_not_flagged(self):
        assert detect_oom_kill(1, "ValueError: bad shape") is False


class TestProfileResult:

    def test_headroom_none_without_quota(self):
        assert ProfileResult().memory_headroom_pct() is None

    def test_headroom_positive_when_under_budget(self):
        result = ProfileResult()
        result.quota_memory_bytes = 2 ** 30          # 1 GiB
        result.peak_memory_bytes = 2 ** 29           # 512 MiB
        assert result.memory_headroom_pct() == 50.0

    def test_headroom_negative_when_over_budget(self):
        """Exceeding the declared edge budget must be visible, not clamped."""
        result = ProfileResult()
        result.quota_memory_bytes = 2 ** 29
        result.peak_memory_bytes = 2 ** 30
        assert result.memory_headroom_pct() == -100.0

    def test_as_dict_is_json_safe(self):
        import json

        json.dumps(ProfileResult().as_dict())


# ------------------------------------------------------------------ host check

class TestHostCheck:

    def test_process_tier_is_permissive(self):
        """
        B15: Docker and kind are cluster-tier requirements only.

        Demanding them for a pure-Python run blocked macOS and CI users for no
        reason; the process tier must come up on any host that can run Ianvs.
        """
        from core.testcasecontroller.simulation.hostcheck import check_process_tier

        assert bool(check_process_tier()) is True

    def test_probe_never_raises_on_missing_binary(self):
        """B1: a missing tool must report, not raise CalledProcessError."""
        from core.testcasecontroller.simulation.hostcheck import command_version

        assert command_version("definitely-not-a-real-binary-xyz") is None

    def test_cpu_count_positive(self):
        from core.testcasecontroller.simulation.hostcheck import get_cpu_count

        assert get_cpu_count() >= 1


# ------------------------------------------------------ integration (isolation)

@pytest.mark.skipif(os.name == "nt", reason="POSIX process semantics required")
class TestProcessSandboxIntegration:
    """End-to-end: a crashing test case must not take the framework with it."""

    @staticmethod
    def _make_sandbox(tmp_path, memory="256Mi"):
        from core.testcasecontroller.simulation.sandbox.process import ProcessSandbox

        config = SandboxConfig({
            "enabled": True, "mode": "process", "isolation": "none",
            "resources": {"memory": memory, "timeout": 60},
        })
        return ProcessSandbox(config, ianvs_root=str(tmp_path))

    def test_teardown_removes_workdirs(self, tmp_path):
        sandbox = self._make_sandbox(tmp_path)
        sandbox.prepare()
        # Simulate a run having created a workdir.
        import tempfile

        workdir = tempfile.mkdtemp(prefix="ianvs-sbx-test-")
        sandbox._workdirs.append(workdir)  # pylint: disable=protected-access
        assert os.path.isdir(workdir)
        sandbox.teardown()
        assert not os.path.isdir(workdir), (
            "teardown must remove transient workdirs; the 2022 implementation "
            "never called its own cleanup function at all"
        )

    def test_keep_workdir_retains_directory(self, tmp_path):
        from core.testcasecontroller.simulation.sandbox.process import ProcessSandbox
        import tempfile

        config = SandboxConfig({
            "enabled": True, "mode": "process", "isolation": "none",
            "keep_workdir": True,
        })
        sandbox = ProcessSandbox(config, ianvs_root=str(tmp_path))
        workdir = tempfile.mkdtemp(prefix="ianvs-sbx-keep-")
        sandbox._workdirs.append(workdir)  # pylint: disable=protected-access
        sandbox.teardown()
        assert os.path.isdir(workdir)
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    def test_environment_is_allowlisted(self, tmp_path):
        """A leaked PYTHONPATH from a previous test case must not survive."""
        sandbox = self._make_sandbox(tmp_path)
        os.environ["IANVS_TEST_LEAK"] = "should-not-propagate"
        try:
            env = sandbox._build_env(str(tmp_path))  # pylint: disable=protected-access
            assert "IANVS_TEST_LEAK" not in env
            assert env["PYTHONPATH"] == str(tmp_path)
            assert env["IANVS_SANDBOX"] == "1"
        finally:
            del os.environ["IANVS_TEST_LEAK"]


# ------------------------------------------------------------- env admin logic

class TestEnvironmentAdministrator:

    def test_cluster_mode_requires_simulation_block(self):
        """
        Asking for the cluster tier without a topology is a configuration
        error, not something to paper over with defaults.
        """
        from core.testcasecontroller.simulation.env_admin import (
            SimulationEnvironmentAdministrator,
        )

        admin = SimulationEnvironmentAdministrator(
            SandboxConfig({"enabled": True, "mode": "cluster"}), simulation=None
        )
        with pytest.raises(ValueError, match="no 'simulation' block"):
            admin.resolve_mode()

    def test_auto_falls_back_to_process_without_simulation(self):
        from core.testcasecontroller.simulation.env_admin import (
            SimulationEnvironmentAdministrator,
        )

        admin = SimulationEnvironmentAdministrator(
            SandboxConfig({"enabled": True, "mode": "auto"}), simulation=None
        )
        assert admin.resolve_mode() == SandboxMode.PROCESS.value

    def test_destroy_is_safe_before_build(self):
        """Teardown runs from a finally block; it must never raise."""
        from core.testcasecontroller.simulation.env_admin import (
            SimulationEnvironmentAdministrator,
        )

        admin = SimulationEnvironmentAdministrator(SandboxConfig.disabled())
        admin.destroy()
        admin.destroy()
