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

"""Cluster-tier sandbox backend: kind + KubeEdge edgecore + Sedna.

Fixes, relative to the 2022 implementation:

B6  ``build`` and ``destroy`` used different Sedna branches (``/master/``
    vs ``/main/``). Both are verified live to return HTTP 200 -- GitHub
    redirects between them -- so this was graded a low-severity internal
    inconsistency rather than a hard failure, and both now use ``/main/``.
B7  the cluster was never torn down. Teardown is owned by
    ``SimulationEnvironmentAdministrator.destroy()`` and every caller runs
    it from a ``finally`` block.
B8  ``kind`` was pinned to a hardcoded 2022 release with no architecture
    awareness. Version resolution is delegated to the upstream installer,
    which is the actual place that needs to track new releases.
B11 ARM64 hosts are unsupported by the upstream installer's ``arch()``
    mapping. This is a known, documented limitation of the Sedna script
    this module wraps, not something silently mishandled here.

Not exercised against a live cluster in this environment -- no Docker or
kind were available where this was built. See the proposal's "Honest
limitations" section; this is disclosed week 8-9 work, not a finished,
validated path.
"""

import base64
import os
import pickle
import shutil
import subprocess
import tempfile

from core.testcasecontroller.simulation import hostcheck
from core.testcasecontroller.simulation.job_admin import JobAdministrator
from core.testcasecontroller.simulation.profiler import ProfileResult
from core.testcasecontroller.simulation.sandbox.base import Sandbox

_SEDNA_INSTALL_URL = (
    "https://raw.githubusercontent.com/kubeedge/sedna/main"
    "/scripts/installation/all-in-one.sh"
)


class ClusterBackend:
    """Provisions and tears down the kind + KubeEdge + Sedna cluster."""

    def __init__(self, simulation):
        self.simulation = simulation
        self.kubeconfig = os.path.expanduser(
            f"~/.kube/config-{simulation.cluster_name}")
        self._built = False

    def build(self):
        """Provision the cluster via the Sedna all-in-one installer."""
        checks = hostcheck.check_cluster_tier()
        missing = [name for name in ("docker", "kind", "kubectl", "linux")
                   if not checks[name]]
        if missing:
            raise RuntimeError(
                f"sandbox.mode is 'cluster' but this host is missing: "
                f"{missing}. Use sandbox.mode: process instead, or install "
                "the missing tools.")

        env = dict(os.environ)
        env.update({
            "NUM_CLOUD_WORKER_NODES": str(self.simulation.cloud_number),
            "NUM_EDGE_NODES": str(self.simulation.edge_number),
            "KUBEEDGE_VERSION": self.simulation.kubeedge_version,
            "SEDNA_VERSION": self.simulation.sedna_version,
            "CLUSTER_NAME": self.simulation.cluster_name,
        })
        cmd = f"curl -fsSL {_SEDNA_INSTALL_URL} | bash -"
        subprocess.run(cmd, shell=True, check=True, env=env)
        self._built = True

    def destroy(self):
        """
        Tear the cluster down.

        Safe to call even if ``build()`` never ran or partially failed --
        the 2022 code's equivalent function existed but was never called
        from anywhere (B7); this one always is, from a ``finally`` block.
        """
        if not self._built:
            return
        cmd = (
            f"curl -fsSL {_SEDNA_INSTALL_URL} | "
            f"CLUSTER_NAME={self.simulation.cluster_name} bash /dev/stdin clean"
        )
        subprocess.run(cmd, shell=True, check=False)
        self._built = False


class ClusterSandbox(Sandbox):
    """Runs each test case as a Kubernetes Job on the provisioned cluster."""

    def __init__(self, config, ianvs_root, backend):
        super().__init__(config, ianvs_root)
        self.backend = backend
        self.job_admin = JobAdministrator(kubeconfig=backend.kubeconfig)

    def run_testcase(self, testcase, workspace):
        build_dir = tempfile.mkdtemp(prefix="ianvs-cluster-ctx-")
        payload_dir = os.path.join(build_dir, "ianvs_payload")
        os.makedirs(payload_dir, exist_ok=True)

        with open(os.path.join(payload_dir, "testcase.pkl"), "wb") as handle:
            pickle.dump({"testcase": testcase, "workspace": workspace}, handle)

        requirements_file = getattr(testcase.algorithm, "requirements_file", None)
        requirements_dst = os.path.join(payload_dir, "requirements.txt")
        if requirements_file and os.path.isfile(requirements_file):
            shutil.copy(requirements_file, requirements_dst)
        else:
            with open(requirements_dst, "w", encoding="utf-8"):
                pass

        profile = ProfileResult()
        profile.quota_memory_bytes = self.config.quota.memory_bytes
        try:
            data = self.job_admin.run_testcase(
                build_dir, self.config.quota, build_dir,
                timeout=self.config.quota.timeout or 3600,
            )
            encoded = (data or {}).get("result")
            outcome = pickle.loads(base64.b64decode(encoded)) if encoded else {}
            profile.succeeded = outcome.get("succeeded", False)
            profile.error = outcome.get("error")
            return outcome.get("result"), profile
        except Exception as err:  # pylint: disable=broad-except
            profile.succeeded = False
            profile.error = str(err)
            return None, profile
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)
