# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Provision only the dedicated, disposable two-node KubeEdge experiment."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = "ianvs-repair"
KEINK = "5a35bdb57230b7ad077dc200ae1684b8238976ee"
KUBEEDGE = "1a7ee88d3c61dc781cc9c9053066ab3d38519dd5"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def checkout(url, path, revision):
    if not path.exists():
        run("git", "clone", "--filter=blob:none", "--no-checkout", url, path)
        run("git", "-C", path, "checkout", "--detach", revision)
    actual = run("git", "-C", path, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise RuntimeError("Dependency revision differs: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "status", "delete", "images"])
    parser.add_argument("--workspace", default="workspace/infra-repair")
    parser.add_argument("--go-cache", help="Optional reusable Go build/download cache")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Use Linux/WSL with Docker")
    work = Path(args.workspace).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / "kubeconfig.infra-repair.lock").open("a") as lock:
        if args.action != "status":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        kubeconfig = work / "kubeconfig"
        client = work / "bin/kubectl"
        kubectl = str(client) if client.is_file() else "kubectl"
        nodes = run("docker", "ps", "-a", "--filter", "label=io.x-k8s.kind.cluster=" + CLUSTER,
                    "--format", "{{.Names}}", capture_output=True, text=True).stdout.split()
        if args.action == "status":
            print(json.dumps({"cluster": CLUSTER, "nodes": nodes}))
            if nodes:
                run(kubectl, "--kubeconfig", kubeconfig, "get", "nodes", "-o", "wide")
            return
        if args.action == "delete":
            allowed = {CLUSTER + "-control-plane", CLUSTER + "-worker"}
            if not set(nodes).issubset(allowed):
                raise RuntimeError("Unexpected dedicated cluster membership; refusing deletion")
            if nodes:
                run("docker", "rm", "-f", *nodes)
            return
        if nodes and args.action == "create":
            raise RuntimeError("Dedicated cluster already exists; use status or images")
        if args.action == "images" and set(nodes) != {CLUSTER + "-control-plane", CLUSTER + "-worker"}:
            raise RuntimeError("Both owned dedicated nodes are required before image import")
        keink = work / "third_party/keink"
        source = work / "third_party/kubeedge"
        cache = Path(args.go_cache).resolve() if args.go_cache else work / "go-cache"
        cache.mkdir(parents=True, exist_ok=True)

        def go(directory, *command, env=()):
            run("docker", "run", "--rm", "--network", "host", "--label", "ianvs.infra-repair=build",
                "-e", "GOTOOLCHAIN=go1.22.12", *env, "-v", str(cache) + ":/go",
                "-v", str(directory) + ":/src", "-w", "/src", "golang:1.25-bookworm", *command)

        if args.action == "create":
            checkout("https://github.com/kubeedge/keink.git", keink, KEINK)
            checkout("https://github.com/kubeedge/kubeedge.git", source, KUBEEDGE)
            go(keink, "go", "build", "-buildvcs=false", "-o", "bin/keink", ".")
            build = work / "node-build"
            (build / "crds").mkdir(parents=True, exist_ok=True)
            for path in (source / "build/crds").rglob("*.yaml"):
                shutil.copyfile(path, build / "crds" / path.name)
            for path in (keink / "build/tools").glob("*.service"):
                shutil.copyfile(path, build / path.name)
            service = build / "cloudcore.service"
            service.write_text(service.read_text().replace("ExecStart=", "ExecStartPre=/usr/bin/kubectl --kubeconfig=/etc/kubernetes/admin.conf apply -f /etc/kubeedge/crds/\nExecStart="))
            shutil.copyfile(ROOT / "deploy/Dockerfile.node", build / "Dockerfile")
            run("docker", "build", "-t", "ianvs-repair-node:v1.23.1", build)
            config = work / "cluster.yaml"
            shutil.copyfile(ROOT / "deploy/cluster.yaml", config)
            run(keink / "bin/keink", "create", "kubeedge", "--name", CLUSTER,
                "--image", "ianvs-repair-node:v1.23.1", "--config", config,
                "--kubeconfig", kubeconfig, "--wait", "180s",
                env={**os.environ, "KUBECONFIG": str(kubeconfig)})
            client.parent.mkdir(exist_ok=True)
            run("docker", "cp", CLUSTER + "-control-plane:/usr/bin/kubectl", client)
            client.chmod(0o755)
            kubectl = str(client)
            # The edge fixture serves through host networking. Its EdgeCore does
            # not inject Kubernetes service environment variables into kindnet.
            # Keep this CNI daemon on the kubelet-managed control-plane node only.
            patch = {"spec": {"template": {"spec": {"nodeSelector": {
                "node-role.kubernetes.io/control-plane": ""}}}}}
            run(kubectl, "--kubeconfig", kubeconfig, "-n", "kube-system", "patch", "daemonset",
                "kindnet", "--type=merge", "-p", json.dumps(patch))
        build = work / "gateway-build"
        build.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "deploy/gateway.go", build / "gateway.go")
        shutil.copyfile(ROOT / "deploy/Dockerfile.gateway", build / "Dockerfile")
        images = []
        for architecture in ("amd64", "arm64"):
            go(build, "go", "build", "-o", "gateway-" + architecture, "gateway.go",
               env=("-e", "CGO_ENABLED=0", "-e", "GOOS=linux", "-e", "GOARCH=" + architecture))
            image = "ianvs-inference-gateway:" + architecture
            run("docker", "build", "--provenance=false", "--platform", "linux/" + architecture,
                "--build-arg", "BINARY=gateway-" + architecture, "-t", image, build)
            images.append(image)
        archive = work / "gateway-images.tar"
        run("docker", "save", "-o", archive, *images)
        run("docker", "cp", archive, CLUSTER + "-worker:/root/ianvs-gateway-images.tar")
        run("docker", "exec", CLUSTER + "-worker", "ctr", "-n", "k8s.io", "images", "import",
            "--all-platforms", "--snapshotter", "overlayfs", "/root/ianvs-gateway-images.tar")
        run(kubectl, "--kubeconfig", kubeconfig, "get", "nodes", "-o", "wide")


if __name__ == "__main__":
    main()
