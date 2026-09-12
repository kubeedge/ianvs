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

"""Thin subprocess wrappers around kubectl, used by the cluster-tier Job
Administrator (job_admin.py). Not exercised against a live cluster in this
environment -- see the proposal's "Honest limitations" section.
"""

import json
import subprocess
import time


class KubectlError(RuntimeError):
    """A kubectl invocation failed."""


def run_kubectl(args, kubeconfig=None, check=True, capture=True):
    """Run kubectl with the given args, returning the completed process."""
    cmd = ["kubectl"]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]
    cmd += args
    result = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if check and result.returncode != 0:
        raise KubectlError(
            f"kubectl {' '.join(args)} failed ({result.returncode}): {result.stderr}")
    return result


def apply_manifest(manifest_path, kubeconfig=None):
    """``kubectl apply -f <manifest_path>``."""
    return run_kubectl(["apply", "-f", manifest_path], kubeconfig=kubeconfig)


def delete_manifest(manifest_path, kubeconfig=None):
    """``kubectl delete -f <manifest_path>``, tolerating an already-gone resource."""
    return run_kubectl(
        ["delete", "-f", manifest_path, "--ignore-not-found=true"],
        kubeconfig=kubeconfig, check=False,
    )


def get_configmap(name, namespace="default", kubeconfig=None):
    """Return a ConfigMap's ``data`` as a dict, or ``None`` if it doesn't exist."""
    result = run_kubectl(
        ["get", "configmap", name, "-n", namespace, "-o", "json"],
        kubeconfig=kubeconfig, check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout).get("data", {})


def wait_for_configmap(name, namespace="default", kubeconfig=None,
                        timeout=3600, poll_interval=5):
    """
    Poll for a result ConfigMap until it appears or the timeout elapses.

    This is the "list-watch the results" responsibility from the original
    design (PR #35, responsibility 4); polling rather than a real Kubernetes
    watch keeps the dependency surface to ``kubectl`` alone.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = get_configmap(name, namespace=namespace, kubeconfig=kubeconfig)
        if data is not None:
            return data
        time.sleep(poll_interval)
    raise TimeoutError(
        f"ConfigMap {namespace}/{name} did not appear within {timeout}s.")
