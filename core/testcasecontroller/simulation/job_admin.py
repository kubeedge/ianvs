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

"""Simulation Job Administrator: deploys one test case as a Job on the
simulated cluster and retrieves its result.

PR #35 (OSPP 2022) specified four responsibilities for this component and
none were ever implemented -- see section 2.2 of
``docs/proposals/simulation/sandbox-engine/ianvs-simulation-sandbox.md``.
This module is the first implementation of all four:

1. Build the image of the algorithm to be tested
2. Generate the YAML file of the simulation job
3. Deploy and delete the simulation job
4. List-watch the results of the simulation job

Not exercised against a live cluster in this environment (no Docker/kind
were available where this was built) -- see the proposal's "Honest
limitations" section. The primary, tested execution path is the process
tier (``sandbox/process.py``).
"""

import os
import subprocess
import uuid

from core.testcasecontroller.simulation import kubeutil

_JOB_TEMPLATE = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
  labels:
    app: ianvs-simulation-sandbox
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: {image}
          env:
            - name: IANVS_RESULT_CONFIGMAP
              value: "{configmap_name}"
            - name: IANVS_RESULT_NAMESPACE
              value: "{namespace}"
          resources:
            limits:
              memory: "{memory}"
              cpu: "{cpus}"
"""


class JobAdministrator:
    """Owns one test case's lifecycle on the simulated cluster."""

    def __init__(self, kubeconfig, namespace="default", image_registry=None):
        self.kubeconfig = kubeconfig
        self.namespace = namespace
        self.image_registry = image_registry

    def build_image(self, build_context, tag):
        """Build the container image that will run one test case."""
        subprocess.run(["docker", "build", "-t", tag, build_context], check=True)
        return tag

    def generate_job_manifest(self, job_name, image, configmap_name, quota, workdir):
        """Write the Job manifest for one test case."""
        manifest = _JOB_TEMPLATE.format(
            job_name=job_name, namespace=self.namespace, image=image,
            configmap_name=configmap_name,
            memory=f"{(quota.memory_bytes or 2 * 2 ** 30) // 2 ** 20}Mi",
            cpus=quota.cpus or 1,
        )
        manifest_path = os.path.join(workdir, f"{job_name}.yaml")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(manifest)
        return manifest_path

    def deploy(self, manifest_path):
        """``kubectl apply`` the Job."""
        kubeutil.apply_manifest(manifest_path, kubeconfig=self.kubeconfig)

    def delete(self, manifest_path):
        """Remove the Job and its pods."""
        kubeutil.delete_manifest(manifest_path, kubeconfig=self.kubeconfig)

    def await_result(self, configmap_name, timeout):
        """List-watch the result ConfigMap the worker publishes on exit."""
        return kubeutil.wait_for_configmap(
            configmap_name, namespace=self.namespace,
            kubeconfig=self.kubeconfig, timeout=timeout,
        )

    def run_testcase(self, build_context, quota, workdir, timeout):
        """Run one test case end-to-end: build, deploy, await, clean up."""
        job_id = uuid.uuid4().hex[:10]
        job_name = f"ianvs-tc-{job_id}"
        configmap_name = f"{job_name}-result"
        tag = f"{self.image_registry or 'localhost'}/ianvs-sandbox:{job_id}"

        self.build_image(build_context, tag)
        manifest_path = self.generate_job_manifest(
            job_name, tag, configmap_name, quota, workdir)
        self.deploy(manifest_path)
        try:
            return self.await_result(configmap_name, timeout=timeout)
        finally:
            self.delete(manifest_path)
