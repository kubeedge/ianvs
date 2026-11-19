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
Simulation Job Administrator: build the image of the algorithms under test, deploy a
test case as a Kubernetes Job onto a node of the simulation cluster, watch it and
collect its result.

Every node of the cluster (kind nodes and KubeEdge edge nodes alike) is a docker
container on the host, so files and images reach a node with `docker cp` and
`docker exec`, and kubectl runs inside the control-plane container: the host needs
no kubectl and no registry.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time

from core.common.log import LOGGER
from core.simulationcontroller.runner import NETWORK_READY_FILE

DEFAULT_CLUSTER_NAME = "sedna-mini"
EDGE_ROLE_LABEL = "node-role.kubernetes.io/edge"
EDGE_CONTAINER_LABEL = "sedna.io=sedna-mini-edge"
EDGE_CLUSTER_LABEL = "sedna.io/cluster"
IMAGE_REPOSITORY = "ianvs-sandbox"
# where files are staged on a node, at their host paths: the inputs of the test cases
# once per run under <INPUT_STAGE_ROOT>, the output directory of each job under
# <NODE_STAGE_ROOT>/<job>. They live in the node container, which is removed with the
# simulation cluster.
NODE_STAGE_ROOT = "/var/lib/ianvs-sandbox"
INPUT_STAGE_ROOT = f"{NODE_STAGE_ROOT}/inputs"
# where the kubelet (edgecore on an edge node) keeps the log of a job's container
POD_LOG_GLOB = "/var/log/pods/default_{job}-*_*/testcase/*.log"
POLL_INTERVAL_S = 2
EDGE_READY_TIMEOUT_S = 120
# the Ianvs repo files the image needs, relative to the repo root
_IMAGE_SOURCES = ("core", "requirements.txt", os.path.join("resources", "third_party"))

DOCKERFILE = """\
FROM {base_image}
WORKDIR /opt/ianvs
COPY . /opt/ianvs
RUN pip install --no-cache-dir resources/third_party/*.whl \\
    && pip install --no-cache-dir -r requirements.txt{extra_requirements}
ENV PYTHONPATH=/opt/ianvs
"""


class JobFailure(RuntimeError):
    """a simulation job did not succeed; status is OOM_FAILURE, EXEC_FAILURE or TIMEOUT."""

    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status
        # the system metrics measured before the failure, for the leaderboard row
        self.metrics = {}


def _run(cmd, **kwargs):
    """run a command (list, no shell); raise RuntimeError with its stderr on failure."""
    kwargs.setdefault("capture_output", True)
    try:
        ret = subprocess.run(cmd, check=False, **kwargs)
    except FileNotFoundError as err:
        raise RuntimeError(f"{cmd[0]} is not installed") from err
    if ret.returncode != 0:
        stderr = ret.stderr.decode(errors="replace") if isinstance(ret.stderr, bytes) \
            else (ret.stderr or "")
        raise RuntimeError(f"`{' '.join(cmd[:4])} ...` failed: {stderr.strip()[-500:]}")
    return ret


def ianvs_root():
    """the root of the Ianvs source tree this process runs from."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimulationJobAdministrator:
    """
    SimulationJobAdministrator: deploys test cases onto a simulation cluster built by
    the Simulation System Administrator.

    Parameters
    ----------
    cluster_name : string
        name of the simulation cluster; empty means the installer default.
    """

    def __init__(self, cluster_name=""):
        self.cluster_name = cluster_name or DEFAULT_CLUSTER_NAME
        self.control_plane = f"{self.cluster_name}-control-plane"

    def kubectl(self, *args, stdin=None):
        """run kubectl in the control-plane container; return its stdout."""
        cmd = ["docker", "exec", "-i", self.control_plane, "kubectl", *args]
        return _run(cmd, input=stdin, text=True).stdout

    def nodes(self):
        """
        Returns
        -------
        list
            (name, is_edge, is_ready) of every node of the cluster
        """
        items = json.loads(self.kubectl("get", "nodes", "-o", "json"))["items"]
        nodes = []
        for item in items:
            conditions = item.get("status", {}).get("conditions", [])
            labels = item["metadata"].get("labels", {})
            nodes.append((item["metadata"]["name"], EDGE_ROLE_LABEL in labels,
                          any(c.get("type") == "Ready" and c.get("status") == "True"
                              for c in conditions)))
        return nodes

    def select_node(self):
        """a node for the simulation jobs: an edge node, else a cloud worker, else the
        control plane."""
        # an edge node can be NotReady for a while when it reconnects to the cloud
        deadline = time.time() + EDGE_READY_TIMEOUT_S
        while True:
            nodes = self.nodes()
            if any(is_edge and ready for _, is_edge, ready in nodes) or \
                    not any(is_edge for _, is_edge, _ in nodes):
                break
            if time.time() >= deadline:
                LOGGER.warning("no edge node of cluster %s is Ready; the test cases run on "
                               "a cloud node", self.cluster_name)
                break
            time.sleep(POLL_INTERVAL_S)
        nodes = [(name, is_edge) for name, is_edge, ready in nodes if ready]
        if not nodes:
            raise RuntimeError(f"simulation cluster {self.cluster_name} has no Ready node")
        edge = [name for name, is_edge in nodes if is_edge]
        workers = [name for name, is_edge in nodes
                   if not is_edge and name != self.control_plane]
        return sorted(edge or workers or [self.control_plane])[0]

    def node_container(self, node):
        """the docker container of a node of the cluster."""
        if node.startswith(f"{self.cluster_name}-"):
            # a kind node: the container has the node's name
            return node
        listed = _run(["docker", "ps", "--filter", f"label={EDGE_CONTAINER_LABEL}", "--format",
                       f'{{{{.Names}}}} {{{{.Label "{EDGE_CLUSTER_LABEL}"}}}}'],
                      text=True).stdout.split("\n")
        names = {}
        for line in listed:
            if line.strip():
                name, _, cluster = line.partition(" ")
                names[name] = cluster.strip()
        # labeled by cluster (newer installer), else named <cluster>-<node>, else the
        # installer's former fixed name sedna-mini-<node>
        for name, cluster in names.items():
            if cluster == self.cluster_name and self._hostname(name) == node:
                return name
        for name in (f"{self.cluster_name}-{node}", f"{DEFAULT_CLUSTER_NAME}-{node}"):
            if name in names:
                return name
        raise RuntimeError(f"no docker container found for node {node}")

    @staticmethod
    def _hostname(container):
        return _run(["docker", "inspect", "--format", "{{.Config.Hostname}}", container],
                    text=True).stdout.strip()

    @staticmethod
    def build_image(profile):
        """
        build the image the test cases run in: the Ianvs core, its requirements and the
        extra requirement files of the sandbox profile. A prebuilt profile.image is used
        as it is. Returns the image name.
        """
        if profile.image:
            return profile.image

        root = ianvs_root()
        with tempfile.TemporaryDirectory(prefix="ianvs-image-") as context:
            for source in _IMAGE_SOURCES:
                src = os.path.join(root, source)
                dst = os.path.join(context, source)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
                elif os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    raise RuntimeError(f"cannot build the sandbox image: {src} not found")
            extra = ""
            for i, requirement in enumerate(profile.requirements):
                if not os.path.isfile(requirement):
                    raise RuntimeError(f"sandbox requirements file {requirement} not found")
                if requirement.endswith(".whl"):
                    # a local wheel, such as an example's algorithm package, installed
                    # like the examples install it on the host: without the dependencies
                    # it declares, which are often pinned for an older Python; its real
                    # dependencies come from a requirements file. pip needs the wheel's
                    # file name unchanged
                    wheel = f"wheels-{i}/{os.path.basename(requirement)}"
                    os.makedirs(os.path.join(context, f"wheels-{i}"))
                    shutil.copy2(requirement, os.path.join(context, wheel))
                    extra += f" \\\n    && pip install --no-cache-dir --no-deps {wheel}"
                    continue
                shutil.copy2(requirement, os.path.join(context, f"requirements-extra-{i}.txt"))
                extra += f" \\\n    && pip install --no-cache-dir -r requirements-extra-{i}.txt"
            with open(os.path.join(context, "Dockerfile"), "w", encoding="utf-8") as file:
                file.write(DOCKERFILE.format(base_image=profile.base_image,
                                             extra_requirements=extra))

            image = f"{IMAGE_REPOSITORY}:{_digest(context)}"
            if subprocess.run(["docker", "image", "inspect", image], check=False,
                              capture_output=True).returncode == 0:
                LOGGER.info("sandbox image %s is up to date", image)
                return image
            LOGGER.info("building sandbox image %s", image)
            _run(["docker", "build", "-t", image, context], capture_output=False)
        return image

    @staticmethod
    def load_image(image, container):
        """import an image of the host into the containerd of a node."""
        LOGGER.info("loading image %s into node container %s", image, container)
        with subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE) as save:
            _run(["docker", "exec", "-i", container,
                  "ctr", "-n", "k8s.io", "images", "import", "-"], stdin=save.stdout)
            save.stdout.close()
            if save.wait() != 0:
                raise RuntimeError(f"docker save {image} failed")

    @staticmethod
    def disable_swap(container):
        """
        turn off swap for the pods of a node, so a memory limit is a hard limit.

        The kubelet of the simulation clusters leaves memory.swap.max unset on cgroup v2:
        a test case over its limit swaps instead of being OOM-killed, and the thrashing
        node can lose its edge-cloud connection. The limit of the kubepods cgroup
        applies to every pod under it.
        """
        script = ("f=$(find /sys/fs/cgroup -maxdepth 4 -type d -name 'kubepods*' "
                  "| awk '{ print length, $0 }' | sort -n | head -1 | cut -d' ' -f2-)"
                  "/memory.swap.max; [ -f \"$f\" ] && echo 0 > \"$f\" && echo \"$f\"")
        try:
            LOGGER.info("swap turned off for pods: %s:%s", container,
                        _run(["docker", "exec", container, "sh", "-c", script],
                             text=True).stdout.strip())
        except RuntimeError:
            LOGGER.warning("could not turn off swap for the pods of %s; a test case over "
                           "its memory limit may swap instead of failing", container)

    @staticmethod
    def stage_paths(container, stage, paths):
        """copy host paths onto a node, each to <stage>/<host path>."""
        for path in paths:
            target = stage + path
            if os.path.isdir(path):
                # copy the content, so it merges with a part of it that is staged already
                _run(["docker", "exec", container, "mkdir", "-p", target])
                _run(["docker", "cp", os.path.join(path, "."), f"{container}:{target}"])
            else:
                _run(["docker", "exec", container, "mkdir", "-p", os.path.dirname(target)])
                _run(["docker", "cp", path, f"{container}:{target}"])

    @staticmethod
    # pylint: disable=too-many-arguments
    def job_manifest(job_name, node, image, command, mounts, resources, workdir):
        """
        the Kubernetes Job of a test case.

        Parameters
        ----------
        mounts : list
            (node path, host path) pairs: each staged copy on the node is mounted at its
            host path, so the paths in the test case configs resolve as they do on the host
        resources : tuple
            (max_memory_mb, max_cpu_cores)
        """
        memory_mb, cpus = resources
        volumes, volume_mounts = [], []
        for i, (node_path, path) in enumerate(mounts):
            volumes.append({"name": f"path-{i}", "hostPath": {"path": node_path}})
            volume_mounts.append({"name": f"path-{i}", "mountPath": path})
        limits = {"memory": f"{int(memory_mb)}Mi", "cpu": str(cpus)}
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "labels": {"app": "ianvs-sandbox"}},
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": {"app": "ianvs-sandbox", "job-name": job_name}},
                    "spec": {
                        "restartPolicy": "Never",
                        "nodeSelector": {"kubernetes.io/hostname": node},
                        # edge and control-plane nodes are tainted
                        "tolerations": [{"operator": "Exists"}],
                        "containers": [{
                            "name": "testcase",
                            "image": image,
                            # the image is imported into the node, never pulled
                            "imagePullPolicy": "Never",
                            "command": command,
                            "workingDir": workdir,
                            "resources": {"limits": limits, "requests": limits},
                            "volumeMounts": volume_mounts,
                            "terminationMessagePolicy": "FallbackToLogsOnError",
                        }],
                        "volumes": volumes,
                    },
                },
            },
        }

    def run_job(self, manifest, timeout, on_poll=None):
        """
        create the Job and watch its pod until it terminates.

        Parameters
        ----------
        on_poll : callable
            called with the pod (as JSON) at every poll, e.g. to sample it

        Returns
        -------
        dict
            the terminated state of the pod's container: reason, exitCode, message
        """
        job_name = manifest["metadata"]["name"]
        self.kubectl("apply", "-f", "-", stdin=json.dumps(manifest))
        deadline = time.time() + timeout
        while time.time() < deadline:
            pods = json.loads(self.kubectl("get", "pods", "-l", f"job-name={job_name}",
                                           "-o", "json"))["items"]
            for pod in pods:
                if on_poll is not None:
                    on_poll(pod)
                for status in pod.get("status", {}).get("containerStatuses", []):
                    terminated = status.get("state", {}).get("terminated")
                    if terminated:
                        return terminated
                    waiting = status.get("state", {}).get("waiting", {})
                    if waiting.get("reason") in ("ErrImageNeverPull", "CreateContainerError",
                                                 "CreateContainerConfigError"):
                        raise JobFailure("EXEC_FAILURE",
                                         f"{waiting['reason']}: {waiting.get('message', '')}")
            time.sleep(POLL_INTERVAL_S)
        raise JobFailure("TIMEOUT", f"job {job_name} did not finish within {timeout} s")

    @staticmethod
    def pod_cgroup(container, pod_uid):
        """
        the cgroup directory of a pod on its node, e.g.
        /sys/fs/cgroup/kubepods.slice/kubepods-pod<uid>.slice (systemd driver, with the
        uid's dashes as underscores) or /sys/fs/cgroup/kubepods/pod<uid> (cgroupfs).
        """
        names = f"-name '*pod{pod_uid}*' -o -name '*pod{pod_uid.replace('-', '_')}*'"
        found = _run(["docker", "exec", container, "sh", "-c",
                      f"find /sys/fs/cgroup -maxdepth 5 -type d \\( {names} \\) | head -1"],
                     text=True).stdout.strip()
        if not found:
            raise RuntimeError(f"cgroup of pod {pod_uid} not found on {container}")
        return found

    @staticmethod
    def pod_pid(container, cgroup):
        """a process of the pod, for entering its network namespace."""
        pid = _run(["docker", "exec", container, "sh", "-c",
                    f"cat {cgroup}/*/cgroup.procs {cgroup}/cgroup.procs 2>/dev/null | head -1"],
                   text=True).stdout.strip()
        if not pid:
            raise RuntimeError(f"no process in cgroup {cgroup}")
        return pid

    @staticmethod
    def pod_ip(container, pid):
        """the pod's IPv4 address, read in its network namespace on the node; "" until
        its network is set up. Edge nodes report a pod's status to the API server late,
        so the node, not the API, tells when the pod can be shaped."""
        try:
            return _run(["docker", "exec", container, "sh", "-c",
                         f"nsenter -t {pid} -n ip -4 -o addr show eth0 2>/dev/null"
                         " | awk '{print $4}' | cut -d/ -f1"], text=True).stdout.strip()
        except RuntimeError:
            return ""

    @staticmethod
    def apply_network(container, pid, pod_ip, bandwidth_mbps, latency_ms):
        """
        emulate a network condition for a pod with tc/netem: on the pod's eth0 (the
        traffic it sends) the bandwidth and the added round-trip delay, and on the
        node's side of its veth pair (the traffic it receives) the bandwidth.
        """
        rate = f"rate {bandwidth_mbps:g}mbit"
        delay = f"delay {latency_ms:g}ms " if latency_ms else ""
        script = (
            f"nsenter -t {pid} -n tc qdisc replace dev eth0 root netem {delay}{rate} && "
            f"dev=$(ip -o route get {pod_ip} | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p') && "
            f"[ -n \"$dev\" ] && tc qdisc replace dev \"$dev\" root netem {rate} && echo \"$dev\"")
        peer = _run(["docker", "exec", container, "sh", "-c", script], text=True).stdout.strip()
        LOGGER.info("network emulation for pod %s: %s mbit/s, +%s ms (pod eth0, node %s)",
                    pod_ip, f"{bandwidth_mbps:g}", f"{latency_ms:g}", peer)

    @staticmethod
    def signal_network_ready(container, stage, output_dir):
        """tell the job's runner that its network emulation is in place."""
        _run(["docker", "exec", container, "touch",
              f"{stage}{output_dir}/{NETWORK_READY_FILE}"])

    @staticmethod
    def sample_pod(container, cgroup, pid, du_path):
        """
        one resource sample of a pod: the node's time, memory current/peak (bytes), CPU
        usage (µs), block I/O read/written (bytes), eth0 received/sent (bytes) and the
        size of a directory (KiB), as key=value lines (see parse_sample).
        """
        script = (
            f"c={cgroup}; echo t=$(date +%s.%N); "
            "echo mem=$(cat $c/memory.current 2>/dev/null); "
            "echo peak=$(cat $c/memory.peak 2>/dev/null); "
            "echo cpu=$(awk '/^usage_usec/ {print $2}' $c/cpu.stat 2>/dev/null); "
            "echo io=$(awk '{for (i = 2; i <= NF; i++) {split($i, a, \"=\"); "
            "if (a[1] == \"rbytes\") r += a[2]; if (a[1] == \"wbytes\") w += a[2]}} "
            "END {print r + 0, w + 0}' $c/io.stat 2>/dev/null); "
            f"echo net=$(nsenter -t {pid} -n cat /proc/net/dev 2>/dev/null | "
            "awk '/eth0:/ {sub(/.*eth0:/, \"\"); print $1, $9}'); "
            f"echo du=$(du -sk {du_path} 2>/dev/null | cut -f1)")
        return _run(["docker", "exec", container, "sh", "-c", script], text=True).stdout

    @staticmethod
    def collect(container, stage, path):
        """copy a path back from its staged copy on a node to the host."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isdir(path):
            # docker cp copies a directory into an existing one; replace its content
            shutil.rmtree(path)
        _run(["docker", "cp", f"{container}:{stage}{path}", path])

    @staticmethod
    def save_logs(container, job_name, path):
        """save the log of a job's container, which its node keeps, to a host file;
        best-effort, as the pod and its log are deleted with the job."""
        try:
            raw = _run(["docker", "exec", container, "sh", "-c",
                        "cat " + POD_LOG_GLOB.format(job=job_name)],
                       encoding="utf-8", errors="replace").stdout
        except RuntimeError as err:
            LOGGER.warning("could not save the log of simulation job %s: %s", job_name, err)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(cri_log_text(raw))

    def delete_job(self, container, job_name):
        """delete the Job, its pod and the staged files; best-effort."""
        for cmd in (["docker", "exec", "-i", self.control_plane, "kubectl", "delete", "job",
                     job_name, "--ignore-not-found", "--wait=false",
                     "--cascade=foreground"],
                    ["docker", "exec", container, "rm", "-rf",
                     f"{NODE_STAGE_ROOT}/{job_name}"]):
            try:
                _run(cmd)
            except RuntimeError as err:
                LOGGER.warning("cleanup of simulation job %s: %s", job_name, err)


def cri_log_text(raw):
    """
    the text of a container log in the CRI format, where each line is
    `<time> <stdout|stderr> <F|P> <text>` and P marks a line split across entries.
    """
    text = []
    for line in raw.splitlines():
        parts = line.split(" ", 3)
        if len(parts) == 4 and parts[1] in ("stdout", "stderr") and parts[2] in ("F", "P"):
            text.append(parts[3] + ("\n" if parts[2] == "F" else ""))
        else:
            text.append(line + "\n")
    return "".join(text)


def _digest(directory):
    """a short digest of a directory's files, so an unchanged image is not rebuilt."""
    sha = hashlib.sha256()
    for base, dirs, files in os.walk(directory):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(base, name)
            sha.update(os.path.relpath(path, directory).encode())
            with open(path, "rb") as file:
                sha.update(file.read())
    return sha.hexdigest()[:12]
