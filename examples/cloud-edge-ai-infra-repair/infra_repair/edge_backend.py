# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real KubeEdge serving gateway and isolated GPU worker on one WSL host.

Cloud/edge are distinct Docker nodes. The model worker binds only their bridge,
and validation traverses the edge gateway. The evaluator's Docker connection is
the out-of-band recovery channel, never exposed to the model agent.
"""

import fcntl
import json
from pathlib import Path
import subprocess
import time
from urllib.error import URLError

from .backend import ProcessBackend
from .io_utils import http_json, save_json, stream_tokens


class EdgeBackend(ProcessBackend):
    cluster = "ianvs-repair"
    node = "ianvs-repair-worker"
    cloud = "ianvs-repair-control-plane"
    good_image = "ianvs-inference-gateway:amd64"
    bad_image = "ianvs-inference-gateway:arm64"
    endpoint = "http://127.0.1.1:18080"

    def __init__(self, *args, kubeconfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.kubeconfig = str(Path(kubeconfig).resolve())
        bundled_client = Path(self.kubeconfig).parent / "bin/kubectl"
        self.kubectl = str(bundled_client) if bundled_client.is_file() else "kubectl"
        self.namespace = "ir-" + self.run_dir.name[:20]
        self.revision = "R2"
        self.image = self.good_image
        self.cloud_ip = None
        self.lock = None
        self.created_namespace = False
        self.node_architecture = None
        self.node_images = {}

    def _command(self, command, payload=None, check=True, timeout=60):
        result = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError("Command failed: " + result.stderr[-3000:])
        return result

    def _kubectl(self, *args, payload=None, check=True):
        return self._command([self.kubectl, "--kubeconfig", self.kubeconfig, "--request-timeout=30s",
                              "-n", self.namespace, *args], payload, check)

    def _node_info(self, node):
        info = json.loads(self._command(["docker", "inspect", node]).stdout)[0]
        if info["Config"]["Labels"].get("io.x-k8s.kind.cluster") != self.cluster:
            raise RuntimeError("Refusing access to a node outside the dedicated experiment cluster")
        self.node_images[node] = info["Image"]
        return next(iter(info["NetworkSettings"]["Networks"].values()))

    def prepare(self):
        if not Path(self.kubeconfig).is_file():
            raise ValueError("A dedicated KubeEdge kubeconfig is required")
        context = self._kubectl("config", "current-context").stdout.strip()
        if context != "kind-" + self.cluster:
            raise ValueError("Refusing kubeconfig for a different cluster")
        self.lock = Path(self.kubeconfig + ".infra-repair.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.host = self._node_info(self.node)["Gateway"]
        self.cloud_ip = self._node_info(self.cloud)["IPAddress"]
        node = json.loads(self._kubectl("get", "node", self.node, "-o", "json").stdout)
        if "kubeedge" not in node["status"]["nodeInfo"]["kubeletVersion"]:
            raise RuntimeError("Target node is not managed by EdgeCore")
        self.node_architecture = node["status"]["nodeInfo"]["architecture"]
        if self.node_architecture != "amd64":
            raise RuntimeError("This fixture requires an AMD64 edge node")
        self.image_architectures = {}
        image_ids = {}
        for image in (self.good_image, self.bad_image):
            image_info = json.loads(self._command(["docker", "image", "inspect", image]).stdout)[0]
            self.image_architectures[image] = image_info["Architecture"]
            image_ids[image] = image_info["Id"]
        if self.image_architectures != {self.good_image: "amd64", self.bad_image: "arm64"}:
            raise RuntimeError("Fixture image architectures differ from the experiment contract")
        self.runtime_identity = {**self.runtime_identity, "architecture": self.node_architecture, "images": image_ids,
            "node_images": self.node_images,
            "edgecore": node["status"]["nodeInfo"]["kubeletVersion"],
            "container_runtime": node["status"]["nodeInfo"]["containerRuntimeVersion"]}
        save_json(self.run_dir / "edge-environment.json", {
            "node_info": node["status"]["nodeInfo"], "image_architectures": self.image_architectures,
            "topology": "Two Docker nodes and one bridge-bound GPU worker on a single WSL host"})
        save_json(self.run_dir / "edge-journal.json", {
            "namespace": self.namespace, "run_id": self.run_dir.name, "kubeconfig": self.kubeconfig,
            "node": self.node, "cloud_ip": self.cloud_ip, "cluster": self.cluster})
        super().prepare()
        if self.port is None:
            raise RuntimeError("Inference worker failed before deployment")
        namespace = {"apiVersion": "v1", "kind": "Namespace", "metadata": {
            "name": self.namespace, "labels": {"ianvs.infra-repair/run": self.run_dir.name}}}
        self._kubectl("create", "-f", "-", payload=json.dumps(namespace))
        self.created_namespace = True
        self._publish()
        self._deploy()
        self._wait_revision("R2")

    def _publish(self):
        document = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "serving"},
                    "data": {"service.json": json.dumps({"revision": self.revision,
                              "backend_url": "http://%s:%d" % (self.host, self.port)})}}
        self._kubectl("apply", "-f", "-", payload=json.dumps(document))

    def _deploy(self):
        self._kubectl("delete", "pod", "target", "--ignore-not-found", "--wait=true", "--timeout=45s")
        pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "target",
               "labels": {"ianvs.infra-repair/run": self.run_dir.name}}, "spec": {
            "nodeName": self.node, "hostNetwork": True, "dnsPolicy": "Default",
            "automountServiceAccountToken": False, "terminationGracePeriodSeconds": 1,
            "containers": [{"name": "gateway", "image": self.image, "imagePullPolicy": "Never",
                "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                                    "capabilities": {"drop": ["ALL"]}},
                "resources": {"requests": {"cpu": "50m", "memory": "32Mi"},
                              "limits": {"cpu": "500m", "memory": "128Mi"}},
                "volumeMounts": [{"name": "config", "mountPath": "/config", "readOnly": True}]}],
            "volumes": [{"name": "config", "configMap": {"name": "serving"}}]}}
        self._kubectl("create", "-f", "-", payload=json.dumps(pod))

    def _health(self):
        try:
            return http_json(self.endpoint + "/health", timeout=3)
        except (URLError, TimeoutError, ConnectionError):
            return {}

    def _wait_revision(self, revision, timeout=180):
        deadline = time.monotonic() + min(timeout, self.startup_timeout)
        while time.monotonic() < deadline:
            health = self._health()
            if health.get("revision") == revision and health.get("backend_url") == "http://%s:%d" % (self.host, self.port):
                return
            time.sleep(1)
        raise TimeoutError("Edge serving configuration did not converge to " + revision)

    def _rule(self, operation, direction="OUTPUT"):
        return ["docker", "exec", self.node, "iptables", operation, direction,
                "-d" if direction == "OUTPUT" else "-s", self.cloud_ip,
                "-p", "tcp", "--dport" if direction == "OUTPUT" else "--sport", "10000:10002", "-m", "comment", "--comment",
                "ianvs-" + self.run_dir.name, "-j", "DROP"]

    def _blocked(self):
        return bool(self.cloud_ip) and all(self._command(self._rule("-C", direction), check=False).returncode == 0
                                          for direction in ("OUTPUT", "INPUT"))

    def _unblock(self):
        if self.cloud_ip:
            for direction in ("OUTPUT", "INPUT"):
                if self._command(self._rule("-C", direction), check=False).returncode == 0:
                    self._command(self._rule("-D", direction))

    def inspect(self):
        # Diagnose observable connectivity, never expose the injection rule's
        # private comment or existence as a model-facing answer.
        reachable = self._command(["docker", "exec", self.node, "timeout", "3", "bash", "-c",
            'exec 3<>/dev/tcp/$1/10000', "probe", self.cloud_ip], check=False).returncode == 0
        config = json.loads(self._kubectl("get", "configmap", "serving", "-o", "json").stdout)
        desired = json.loads(config["data"]["service.json"])["revision"]
        return {**super().inspect(), "node_architecture": self.node_architecture, "image_architecture":
                self.image_architectures[self.image], "desired_revision": desired,
                "edge_revision": self._health().get("revision"), "control_link_reachable": reachable}

    def logs(self):
        logs = super().logs()
        if self.created_namespace:
            # EdgeCore's optional streaming server may be disabled. Read only
            # this run's container through the evaluator's recovery channel.
            containers = self._command(["docker", "exec", self.node, "crictl", "ps", "-a",
                "--label", "io.kubernetes.pod.namespace=" + self.namespace, "-o", "json"], check=False)
            if containers.returncode == 0:
                for container in json.loads(containers.stdout).get("containers", []):
                    if container.get("labels", {}).get("io.kubernetes.pod.name") == "target":
                        output = self._command(["docker", "exec", self.node, "crictl", "logs",
                            "--tail=20", container["id"]], check=False)
                        logs += "\n" + output.stdout + output.stderr
            events = self._kubectl("get", "events", "--field-selector=involvedObject.name=target", "-o", "json", check=False)
            if events.returncode == 0:
                messages = [event.get("message", "") for event in json.loads(events.stdout).get("items", [])]
                logs += "\n" + "\n".join(messages[-8:])
        return logs[-8000:]

    def generate(self, prompt, timeout=30):
        return stream_tokens(self.endpoint + "/generate-stream", {"prompt": prompt}, timeout)

    def inject(self, scenario):
        if scenario not in ("S4", "S5"):
            return super().inject(scenario)
        if scenario == "S4":
            self.image = self.bad_image
            self._deploy()
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if "exec format error" in self.logs().lower() and not self._health():
                    return True
                time.sleep(2)
            return False
        self.revision = "R1"
        self._publish()
        self._wait_revision("R1")
        for direction in ("OUTPUT", "INPUT"):
            self._command(self._rule("-I", direction))
        self.revision = "R2"
        self._publish()
        time.sleep(3)
        unavailable = False
        try:
            self.generate("probe", timeout=5)
        except URLError as error:
            unavailable = getattr(error, "code", None) == 503
        return self._blocked() and self._health().get("revision") == "R1" and unavailable

    def repair(self, action):
        if action == "restore_image":
            self.image = self.good_image
            self._deploy()
        elif action == "reconnect_control":
            self._unblock()
        else:
            result = super().repair(action)
            if not result["started"]:
                return result
            self._publish()
        self._wait_revision(self.revision)
        return {"started": True, "edge_revision": self._health().get("revision")}

    def rollback(self):
        self._unblock()
        self.configured_model = self.model
        self.gpu_memory_fraction = 0.4
        ProcessBackend.repair(self, "restore_artifacts")
        self.revision = "R2"
        self.image = self.good_image
        self._publish()
        self._deploy()
        self._wait_revision("R2")
        return all(self.integrity().values())

    def cleanup(self):
        edge_error = None
        try:
            self._unblock()
            if self.created_namespace:
                namespace = json.loads(self._kubectl("get", "namespace", self.namespace, "-o", "json").stdout)
                if namespace["metadata"]["labels"].get("ianvs.infra-repair/run") != self.run_dir.name:
                    raise RuntimeError("Namespace ownership changed; refusing deletion")
                self._kubectl("delete", "pod", "target", "--ignore-not-found", "--wait=true", "--timeout=45s")
                self._kubectl("delete", "namespace", self.namespace, "--wait=true", "--timeout=45s")
                if self._health():
                    raise RuntimeError("Edge gateway still responds after cleanup")
        except Exception as error:
            edge_error = error
        finally:
            worker_clean = super().cleanup()
            if self.lock:
                self.lock.close()
        if edge_error:
            raise edge_error
        return worker_clean


def recover_edge(run_dir):
    """Remove only journal-owned edge resources after an evaluator crash.

    This is emergency cleanup, not a claim that model rollback was verified.
    """
    from .backend import recover
    run_dir = Path(run_dir).resolve()
    state = json.loads((run_dir / "edge-journal.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if (state.get("run_id") != run_dir.name or state.get("cluster") != EdgeBackend.cluster
            or state.get("node") != EdgeBackend.node
            or state.get("namespace") != "ir-" + run_dir.name[:20]):
        raise RuntimeError("Edge recovery journal ownership mismatch")
    backend = EdgeBackend(run_dir, config["source"], config["device"], config["startup_timeout"],
                          kubeconfig=state["kubeconfig"])
    with Path(backend.kubeconfig + ".infra-repair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if backend._kubectl("config", "current-context").stdout.strip() != "kind-" + backend.cluster:
            raise RuntimeError("Recovery kubeconfig does not address the dedicated cluster")
        backend._node_info(backend.node)
        cloud_ip = backend._node_info(backend.cloud)["IPAddress"]
        if cloud_ip != state["cloud_ip"]:
            raise RuntimeError("Cluster identity changed since the journal was written")
        backend.cloud_ip = cloud_ip
        backend._unblock()
        namespace = backend._kubectl("get", "namespace", backend.namespace, "--ignore-not-found", "-o", "json")
        if namespace.stdout.strip():
            labels = json.loads(namespace.stdout)["metadata"].get("labels", {})
            if labels.get("ianvs.infra-repair/run") != run_dir.name:
                raise RuntimeError("Refusing recovery of an unowned namespace")
            backend._kubectl("delete", "pod", "target", "--ignore-not-found", "--wait=true", "--timeout=45s")
            backend._kubectl("delete", "namespace", backend.namespace, "--wait=true", "--timeout=45s")
        result = recover(run_dir)
        if backend._health() or backend._blocked():
            raise RuntimeError("Recovery isolation check failed")
        save_json(run_dir / "emergency-recovery.json", result)
        return result
