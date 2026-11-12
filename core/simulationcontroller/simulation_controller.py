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

"""Simulation Controller: run test cases as jobs in the simulation cluster"""

import json
import os

from core.common import utils
from core.common.log import LOGGER
from core.simulationcontroller.runner import RESULT_FILE
from core.simulationcontroller.simulation_job_administrator import (
    INPUT_STAGE_ROOT, NODE_STAGE_ROOT, JobFailure, SimulationJobAdministrator)
from core.simulationcontroller.system_metrics_profiler import (
    SystemMetricsProfiler, parse_sample, select_metrics)

# the log of a simulation job's pod, saved in the test case's output directory
LOG_FILE = "sandbox.log"


# pylint: disable=too-few-public-methods,too-many-instance-attributes
class _JobObserver:
    """
    called at every poll of a simulation job's pod: applies the network emulation once
    the pod runs, then samples the pod's resources for the System Metrics Profiler.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, admin, container, job_stage, output_dir, profiler):
        self.admin = admin
        self.container = container
        self.job_stage = job_stage
        self.output_dir = output_dir
        self.profiler = profiler
        self.cgroup = None
        self.pid = None
        self.network_ready = False

    def on_poll(self, pod):
        """
        apply the network emulation as soon as the pod exists on its node, then sample
        it. The pod is found on the node, not through its status in the API: an edge
        node reports that status late, up to a minute after the pod started.
        """
        if self.cgroup is None:
            try:
                self.cgroup = self.admin.pod_cgroup(self.container, pod["metadata"]["uid"])
                self.pid = self.admin.pod_pid(self.container, self.cgroup)
            except RuntimeError as err:
                LOGGER.debug("pod not on the node yet: %s", err)
                self.cgroup = None
                return
        if not self.network_ready:
            pod_ip = self.admin.pod_ip(self.container, self.pid)
            if not pod_ip:
                return
            try:
                self.admin.apply_network(self.container, self.pid, pod_ip,
                                         self.profiler.bandwidth_mbps,
                                         self.profiler.latency_ms)
                self.admin.signal_network_ready(self.container, self.job_stage,
                                                self.output_dir)
            except RuntimeError as err:
                raise JobFailure("EXEC_FAILURE",
                                 f"network emulation could not be applied: {err}") from err
            self.network_ready = True
        try:
            sample = parse_sample(self.admin.sample_pod(
                self.container, self.cgroup, self.pid, self.job_stage + self.output_dir))
        except RuntimeError as err:
            # the pod may be gone between two polls; the samples so far are kept
            LOGGER.debug("resource sample skipped: %s", err)
            return
        if self.profiler.started is None and "t" in sample:
            # the test case starts once its network is ready
            self.profiler.started = sample["t"]
        self.profiler.add(sample)


def referenced_paths(config):
    """the existing local paths among the string values of a (nested) config."""
    paths = set()
    if isinstance(config, dict):
        for value in config.values():
            paths |= referenced_paths(value)
    elif isinstance(config, list):
        for value in config:
            paths |= referenced_paths(value)
    elif isinstance(config, str) and config.strip() and os.path.exists(config):
        paths.add(os.path.abspath(config))
    return paths


def mount_paths(paths, workdir):
    """
    the host paths to stage onto the node: the directory of each referenced file (an
    algorithm module may import its siblings, a dataset index lists files next to it),
    the file itself when that directory is the working directory or the root, and only
    the outermost of nested paths.
    """
    dirs = set()
    for path in paths:
        path = os.path.abspath(path)
        parent = path if os.path.isdir(path) else os.path.dirname(path)
        dirs.add(path if parent in (os.path.abspath(workdir), os.sep) else parent)
    outermost = []
    for path in sorted(dirs):
        if not any(path.startswith(kept.rstrip(os.sep) + os.sep) for kept in outermost):
            outermost.append(path)
    return outermost


def _covered(path, staged):
    """whether a path, or a directory that contains it, is among the staged paths."""
    return any(path == kept or path.startswith(kept.rstrip(os.sep) + os.sep)
               for kept in staged)


class SimulationController:
    """
    SimulationController: runs each test case as a Kubernetes Job on a node of the
    simulation cluster instead of in the Ianvs process, so the test case's dependencies,
    files and resource use are contained, and an out-of-memory kill is reported as the
    test case's result instead of stopping the benchmarking job.

    Parameters
    ----------
    simulation : Simulation
        config of the simulation cluster.
    profile : SandboxProfile
        how the test cases run in the cluster.
    testenv_url : string
        the test environment config file of the benchmarking job.
    """

    def __init__(self, simulation, profile, testenv_url):
        self.admin = SimulationJobAdministrator(simulation.cluster_name)
        self.profile = profile
        self.testenv_url = testenv_url
        self.image = None
        self.node = None
        self.container = None
        # the input paths copied onto the node; the test cases of a run share them
        self.staged = []

    def prepare(self):
        """build the image and load it into the node the test cases run on."""
        self.image = self.admin.build_image(self.profile)
        self.node = self.admin.select_node()
        self.container = self.admin.node_container(self.node)
        self.admin.load_image(self.image, self.container)
        self.admin.disable_swap(self.container)
        LOGGER.info("test cases run as simulation jobs on node %s", self.node)

    def run_sandboxed(self, testcase, workspace):
        """
        run a test case as a simulation job.

        Returns
        -------
        test result: dict
            e.g.: {"f1_score": 0.89}

        Raises
        ------
        JobFailure
            with status OOM_FAILURE, EXEC_FAILURE or TIMEOUT
        """
        output_dir = os.path.abspath(
            os.path.join(workspace, testcase.algorithm.name, str(testcase.id)))
        os.makedirs(output_dir, exist_ok=True)
        testcase.output_dir = output_dir
        job = self._job(testcase, workspace, output_dir)
        profiler = SystemMetricsProfiler(job["resources"], self.profile.network())
        observer = _JobObserver(self.admin, self.container, job["stage"], output_dir, profiler)
        terminated = {}
        try:
            self._stage_inputs(job["inputs"])
            terminated = self._run_job(job["name"], job["stage"], job["manifest"], output_dir,
                                       observer.on_poll)
            metrics = self._result(terminated, output_dir, job["resources"])
        except JobFailure as err:
            err.metrics = self._system_metrics(profiler, terminated, output_dir,
                                               oom=err.status == "OOM_FAILURE")
            self._record_failure(testcase.id, output_dir, err)
            raise
        return {**metrics, "status": "SUCCESS", **self._system_metrics(
            profiler, terminated, output_dir, oom=False,
            wall_time=self._parse_result(terminated, output_dir).get("wall_time_s"))}

    def _job(self, testcase, workspace, output_dir):
        """the simulation job of a test case: name, staging directory, inputs, resource
        limits and Kubernetes manifest."""
        workdir = os.getcwd()
        inputs = self._input_paths(testcase.algorithm, workdir, output_dir)
        name = f"ianvs-{testcase.id}"
        stage = f"{NODE_STAGE_ROOT}/{name}"
        mounts = [(INPUT_STAGE_ROOT + path, path) for path in inputs]
        mounts.append((stage + output_dir, output_dir))
        resources = self.profile.resources(testcase.algorithm.paradigm_type)
        bandwidth, latency = self.profile.network()
        LOGGER.info("testcase(id=%s) runs as simulation job %s (memory %s MiB, cpu %s, "
                    "network %s)", testcase.id, name, *resources,
                    f"{bandwidth:g} mbit/s +{latency:g} ms")
        return {"name": name, "stage": stage, "inputs": inputs, "resources": resources,
                "manifest": self.admin.job_manifest(
                    name, self.node, self.image, self._command(testcase, workspace, output_dir),
                    mounts, resources, workdir)}

    def _command(self, testcase, workspace, output_dir):
        """the command of the job's container: the runner, for this test case."""
        algorithm = testcase.algorithm
        # the package imports the runner first, which runpy warns about
        return ["python", "-W", "ignore::RuntimeWarning:runpy",
                "-m", "core.simulationcontroller.runner",
                "--testenv", os.path.abspath(self.testenv_url),
                "--algorithm-name", algorithm.name,
                "--algorithm-url", os.path.abspath(algorithm.config_url),
                "--modules-index", str(algorithm.modules_index),
                "--testcase-id", str(testcase.id),
                "--workspace", os.path.abspath(workspace),
                "--output-dir", output_dir,
                "--wait-network"]

    # pylint: disable=too-many-arguments
    def _system_metrics(self, profiler, terminated, output_dir, oom, wall_time=None):
        """the system metrics the sandbox profile asks for."""
        return select_metrics(profiler.metrics(terminated, oom, output_dir, wall_time),
                              self.profile.system_metrics)

    def _input_paths(self, algorithm, workdir, output_dir):
        """the host paths a test case reads: its configs, the paths they reference
        and the data paths of the sandbox profile, folded by mount_paths."""
        paths = {os.path.abspath(self.testenv_url), os.path.abspath(algorithm.config_url)}
        paths |= referenced_paths(utils.yaml2dict(self.testenv_url))
        paths |= referenced_paths(utils.yaml2dict(algorithm.config_url))
        for path in self.profile.data_paths:
            if not os.path.exists(path):
                raise RuntimeError(f"sandbox_profile data path {path} not found")
            paths.add(os.path.abspath(path))
        paths.discard(output_dir)
        return mount_paths(paths, workdir)

    def _stage_inputs(self, inputs):
        """copy the inputs onto the node, once per run: the test cases share them."""
        new = [path for path in inputs if not _covered(path, self.staged)]
        if new:
            LOGGER.info("copying inputs onto node %s: %s", self.node, ", ".join(new))
            self.admin.stage_paths(self.container, INPUT_STAGE_ROOT, new)
            self.staged.extend(new)

    # pylint: disable=too-many-arguments
    def _run_job(self, job_name, job_stage, manifest, output_dir, on_poll=None):
        """run the job; its output directory and log are copied back, even on failure."""
        try:
            self.admin.stage_paths(self.container, job_stage, [output_dir])
            try:
                terminated = self.admin.run_job(manifest, self.profile.timeout, on_poll)
            except JobFailure:
                try:
                    # whatever the test case wrote before it failed
                    self.admin.collect(self.container, job_stage, output_dir)
                except RuntimeError as err:
                    LOGGER.warning("outputs of simulation job %s: %s", job_name, err)
                raise
            self.admin.collect(self.container, job_stage, output_dir)
        finally:
            self.admin.save_logs(self.container, job_name,
                                 os.path.join(output_dir, LOG_FILE))
            self.admin.delete_job(self.container, job_name)
        return terminated

    @staticmethod
    def _record_failure(testcase_id, output_dir, err):
        """write a failed test case's status, error and system metrics next to its outputs."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, RESULT_FILE), "w", encoding="utf-8") as file:
            json.dump({"testcase_id": str(testcase_id), "status": err.status,
                       "error": str(err), "system_metrics": err.metrics}, file)

    @staticmethod
    def _parse_result(terminated, output_dir):
        """the runner's result: the termination message, or the result file it points to."""
        try:
            result = json.loads(terminated.get("message", ""))
        except ValueError:
            return {}
        if result.get("result_file"):
            try:
                with open(os.path.join(output_dir, RESULT_FILE), encoding="utf-8") as file:
                    result = json.load(file)
            except (OSError, ValueError):
                return {}
        return result

    @staticmethod
    def _result(terminated, output_dir, resources):
        reason = terminated.get("reason", "")
        message = terminated.get("message", "")
        try:
            status = json.loads(message).get("status")
        except (ValueError, AttributeError):
            status = None
        # edgecore reports an OOM kill as a plain SIGKILL (exit code 137, reason Error)
        if reason == "OOMKilled" or (terminated.get("exitCode") == 137 and status is None):
            raise JobFailure("OOM_FAILURE",
                             f"the test case exceeded its memory limit of {resources[0]} MiB; "
                             "raise device_simulation max_memory_mb")

        result = SimulationController._parse_result(terminated, output_dir)
        if terminated.get("exitCode") != 0 or result.get("status") != "SUCCESS":
            error = result.get("error") or message.strip()[-1000:] or reason
            raise JobFailure("EXEC_FAILURE", error)
        return result["metrics"]
