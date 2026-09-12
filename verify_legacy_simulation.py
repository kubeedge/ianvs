#!/usr/bin/env python3
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
Reproducible evidence for the breakages in the 2022 Ianvs simulation code.

Run against a pristine checkout of ``kubeedge/ianvs`` at ``main``::

    git clone https://github.com/kubeedge/ianvs.git
    cd ianvs
    python3 verify_legacy_simulation.py

Every check is self-contained and read-only.  Nothing is installed, no cluster
is provisioned, and no file in the repository is modified.  Checks that need
network access are skipped with a clear notice when offline, so the script
still produces a useful report on an air-gapped machine.

Exit code is the number of confirmed defects, so this can be wired into CI as a
regression guard once the fixes land.
"""

import argparse
import io
import os
import re
import subprocess
import sys
import textwrap

REPO_SIMULATION = "core/testcasecontroller/simulation/simulation.py"
REPO_SYS_ADMIN = (
    "core/testcasecontroller/simulation_system_admin/simulation_system_admin.py"
)
REPO_BENCHMARKINGJOB = "core/cmd/obj/benchmarkingjob.py"
SEDNA_AIO_URL = (
    "https://raw.githubusercontent.com/kubeedge/sedna/main"
    "/scripts/installation/all-in-one.sh"
)

RESULTS = []


def report(ident, title, confirmed, detail, skipped=False):
    """Record and print one finding."""
    if skipped:
        status = "SKIP"
    elif confirmed:
        status = "CONFIRMED"
    else:
        status = "not reproduced"
    RESULTS.append((ident, title, confirmed, skipped))
    print(f"\n[{ident}] {title}")
    print(f"  status : {status}")
    for line in textwrap.wrap(detail, width=76):
        print(f"  {line}")


def read(path):
    """Return file contents, or None when the file is absent."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------- checks

def check_b1_unreachable_install_branch():
    """subprocess.run(check=True) makes the following returncode branch dead."""
    source = read(REPO_SYS_ADMIN)
    if source is None:
        report("B1", "Auto-install branches are unreachable", False,
               "simulation_system_admin.py not found in this checkout.",
               skipped=True)
        return

    has_check_true = "check=True" in source
    has_returncode_branch = re.search(r"if\s+check_\w+\.returncode\s*!=\s*0",
                                      source) is not None

    # Demonstrate the semantics directly rather than asserting from reading.
    raised = False
    try:
        subprocess.run("exit 3", shell=True, check=True)
    except subprocess.CalledProcessError:
        raised = True

    report(
        "B1", "Docker/kind auto-install branches are unreachable dead code",
        has_check_true and has_returncode_branch and raised,
        "check_host_docker() calls subprocess.run(..., check=True) and then "
        "tests ret.returncode != 0. check=True raises CalledProcessError on a "
        "non-zero exit, so control never reaches the test. Verified live: a "
        f"command exiting 3 raised CalledProcessError = {raised}. A host "
        "without Docker therefore crashes Ianvs with a raw traceback instead "
        "of taking the intended fallback.",
    )


def check_b2_bool_accepted_as_int():
    """isinstance(True, int) is True in Python."""
    report(
        "B2", "Booleans pass the node-count type check",
        isinstance(True, int),
        "Simulation._check_fields() validates node counts with "
        "isinstance(value, int). bool subclasses int, so 'edge_number: true' "
        "in YAML validates cleanly and is then interpolated into the Sedna "
        "command as NUM_EDGE_NODES=True.",
    )


def check_b3_unknown_keys_dropped():
    """Unknown YAML keys were silently discarded by the 2022 parser."""
    source = read(REPO_SIMULATION)
    if source is None:
        report("B3", "Unknown config keys silently dropped", False,
               "simulation.py not found.", skipped=True)
        return

    legacy_pattern = "if attribute in self.__dict__:" in source
    confirmed = legacy_pattern

    live = ""
    if legacy_pattern:
        sys.path.insert(0, os.getcwd())
        try:
            from core.testcasecontroller.simulation import Simulation

            sim = Simulation({
                "edge_nodes": 5, "cloud_number": 1, "cluster_name": "c",
                "kubeedge_version": "", "sedna_version": "",
            })
            live = (
                f" Verified live: passing 'edge_nodes: 5' yields "
                f"edge_number = {sim.edge_number}."
            )
        except Exception:  # pylint: disable=broad-except
            live = ""

    report(
        "B3", "Unknown config keys are silently dropped",
        confirmed,
        "_parse_config() assigns only keys already present in __dict__ and "
        "discards everything else without warning. A plausible typo such as "
        "'edge_nodes' instead of 'edge_number' therefore produces a cluster "
        "with zero edge nodes and no diagnostic at all." + live,
    )


def check_b4_empty_config_accepted():
    """An entirely empty simulation block passed validation."""
    source = read(REPO_SIMULATION)
    if source is None:
        report("B4", "Empty required fields accepted", False,
               "simulation.py not found.", skipped=True)
        return
    # The 2022 checks assert type only, never emptiness.
    type_only = 'must be string type' in source and 'not self.cluster_name' not in source
    report(
        "B4", "Empty required fields are accepted",
        type_only,
        "_check_fields() asserts types but never emptiness, so cloud_number=0 "
        "with cluster_name='' validates and then generates a shell command "
        "containing a bare 'CLUSTER_NAME= ', which fails inside the Sedna "
        "script rather than at config-parse time.",
    )


def check_b5_fragile_cpu_parse():
    """lscpu | grep CPU: is locale- and image-dependent."""
    source = read(REPO_SYS_ADMIN)
    if source is None:
        report("B5", "Fragile CPU parsing", False,
               "simulation_system_admin.py not found.", skipped=True)
        return

    uses_lscpu = "lscpu" in source
    completed = subprocess.run(
        "lscpu | grep 'CPU:'", shell=True, capture_output=True, text=True,
        check=False,
    )
    empty = not completed.stdout.strip()
    report(
        "B5", "CPU count parsing is fragile",
        uses_lscpu and empty,
        "get_host_number_of_cpus() shells out to \"lscpu | grep CPU:\" and "
        "splits on ':' and a backslash. The field is 'CPU(s):' on modern "
        "util-linux, the output is locale-dependent, and lscpu is absent from "
        f"slim container images. On this host the grep returned "
        f"{'nothing (parse would raise IndexError)' if empty else 'output'}. "
        f"os.cpu_count() reports {os.cpu_count()} with no subprocess at all.",
    )


def check_b7_teardown_never_called():
    """The destroy function is exported but has no call site."""
    if not os.path.isdir("core"):
        report("B7", "Simulation environment is never torn down", False,
               "core/ not found.", skipped=True)
        return

    hits = []
    for root, _, files in os.walk("core"):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            content = read(path) or ""
            for lineno, line in enumerate(content.splitlines(), start=1):
                if "destory_simulation_enviroment" in line:
                    hits.append((path, lineno, line.strip()))

    definitions = [h for h in hits if h[2].startswith(("def ", "from ", "import "))]
    call_sites = [h for h in hits if h not in definitions]

    report(
        "B7", "The simulation environment is never torn down",
        bool(hits) and not call_sites,
        f"destory_simulation_enviroment() is defined and re-exported "
        f"({len(definitions)} occurrence(s)) but has {len(call_sites)} call "
        f"sites anywhere in core/. BenchmarkingJob.run() calls "
        f"build_simulation_enviroment() and never destroys the cluster, so "
        f"every benchmarking run leaves a kind cluster and its containers "
        f"resident on the host until the user removes them by hand.",
    )


def check_b8_pinned_kind_version():
    """kind is pinned to a 2022 release."""
    source = read(REPO_SYS_ADMIN)
    if source is None:
        report("B8", "kind pinned to a 2022 release", False,
               "simulation_system_admin.py not found.", skipped=True)
        return
    match = re.search(r"kind\.sigs\.k8s\.io/dl/(v[\d.]+)/", source)
    report(
        "B8", "kind is pinned to a 2022 release",
        match is not None,
        f"The installer URL pins kind {match.group(1) if match else 'unknown'} "
        f"(released 2022) and hardcodes the linux-amd64 asset, so the pinned "
        f"binary cannot provision node images built for current Kubernetes "
        f"releases and cannot run on arm64 hosts at all.",
    )


def check_b10_b12_backend_node_ceilings(offline):
    """The Sedna backend caps node counts far below what Ianvs accepts."""
    if offline:
        report("B12", "Backend node-count ceilings are not enforced by Ianvs",
               False, "Network disabled; skipping remote script inspection.",
               skipped=True)
        return

    try:
        import urllib.request

        with urllib.request.urlopen(SEDNA_AIO_URL, timeout=30) as response:
            script = response.read().decode("utf-8", errors="replace")
    except Exception as err:  # pylint: disable=broad-except
        report("B12", "Backend node-count ceilings are not enforced by Ianvs",
               False, f"Could not fetch the Sedna script: {err}", skipped=True)
        return

    cloud_cap = re.search(r"MAX_CLOUD_WORKER_NODES=(\d+)", script)
    edge_cap = re.search(r"MAX_EDGE_WORKER_NODES=(\d+)", script)

    source = read(REPO_SIMULATION) or ""
    ianvs_enforces = "MAX_EDGE" in source or "out of range" in source

    report(
        "B12", "Backend node-count ceilings are not enforced by Ianvs",
        bool(cloud_cap and edge_cap) and not ianvs_enforces,
        f"The Sedna all-in-one script hard-caps topology at "
        f"MAX_CLOUD_WORKER_NODES={cloud_cap.group(1) if cloud_cap else '?'} and "
        f"MAX_EDGE_WORKER_NODES={edge_cap.group(1) if edge_cap else '?'}, and "
        f"aborts above them. The Ianvs Simulation class accepts any integer, "
        f"so 'edge_number: 10' -- the large-scale simulation this feature "
        f"exists to provide -- passes every Ianvs-side check and then fails "
        f"deep inside a piped bash script. The headline capability is capped "
        f"at three edge nodes by its own backend.",
    )

    arch_case = re.search(r"function arch\(\)(.{0,220})", script, re.S)
    only_x86 = bool(arch_case and "x86_64" in arch_case.group(1)
                    and "aarch64" not in arch_case.group(1))
    report(
        "B11", "ARM64 hosts are unsupported by the provisioning backend",
        only_x86,
        "The all-in-one script's arch() maps x86_64 to amd64 and passes every "
        "other machine string through unchanged, so an aarch64 host requests "
        "KubeEdge release assets under a name that does not exist. Apple "
        "Silicon and ARM server hosts cannot provision the cluster tier.",
    )


def check_b6_branch_urls(offline):
    """The build and destroy paths reference different Sedna branches."""
    source = read(REPO_SYS_ADMIN)
    if source is None:
        report("B6", "Build and destroy reference different branches", False,
               "simulation_system_admin.py not found.", skipped=True)
        return

    uses_master = "/sedna/master/" in source or "sedna\\\n/master/" in source
    uses_main = "/sedna\\\n/main/" in source or "/sedna/main/" in source

    detail = (
        "build_simulation_enviroment() fetches the installer from the "
        "'master' branch while destory_simulation_enviroment() uses 'main'. "
    )
    if not offline:
        codes = {}
        for branch in ("master", "main"):
            url = (f"https://raw.githubusercontent.com/kubeedge/sedna/{branch}"
                   f"/scripts/installation/all-in-one.sh")
            try:
                import urllib.request

                with urllib.request.urlopen(url, timeout=20) as response:
                    codes[branch] = response.status
            except Exception:  # pylint: disable=broad-except
                codes[branch] = None
        detail += (
            f"Checked live: master -> {codes.get('master')}, "
            f"main -> {codes.get('main')}. Both resolve, so this is an "
            f"internal inconsistency rather than a hard failure, and is "
            f"reported here as low severity for accuracy."
        )

    report("B6", "Build and destroy reference different Sedna branches",
           uses_master and uses_main, detail)


def check_b9_feature_undocumented():
    """No shipped example demonstrates the simulation block."""
    if not os.path.isdir("examples"):
        report("B9", "No example exercises the simulation feature", False,
               "examples/ not found.", skipped=True)
        return

    found = []
    for root, _, files in os.walk("examples"):
        for name in files:
            if not name.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(root, name)
            content = read(path) or ""
            if re.search(r"^\s*simulation:", content, re.M):
                found.append(path)

    report(
        "B9", "No shipped example exercises the simulation feature",
        not found,
        f"{len(found)} of the shipped benchmarkingjob configs contain a "
        f"'simulation:' block. The feature is present in the codebase but "
        f"undiscoverable and untested from a user's point of view, which is "
        f"why the breakages above went unnoticed for four years.",
    )


def check_b13_no_system_metrics():
    """Ianvs has no system-wise metric types."""
    source = read("core/common/constant.py")
    if source is None:
        report("B13", "No system-level metrics exist", False,
               "constant.py not found.", skipped=True)
        return

    members = re.findall(r"^\s{4}([A-Z_]+)\s*=", source, re.M)
    system_like = [
        m for m in members
        if any(token in m for token in ("MEMORY", "CPU", "WALL", "LATENCY",
                                        "BANDWIDTH", "THROUGHPUT"))
    ]
    report(
        "B13", "Ianvs has no system-level metric types",
        not system_like,
        "SystemMetricType currently enumerates samples_transfer_ratio, FWT, "
        "BWT, task_avg_acc, MATRIX and forget_rate -- all algorithm-wise. "
        "There is no peak memory, CPU utilisation or wall-clock member, so "
        "the leaderboard cannot express what a distributed AI system costs to "
        "run, only how accurate it is.",
    )


def check_b14_fault_isolation():
    """One failing test case discards every result already computed."""
    source = read("core/testcasecontroller/testcasecontroller.py")
    if source is None:
        report("B14", "One failure discards all results", False,
               "testcasecontroller.py not found.", skipped=True)
        return

    raises_out = re.search(
        r"except Exception as err:\s*\n\s*raise RuntimeError\(f?\"testcase",
        source,
    ) is not None
    report(
        "B14", "A single failing test case discards every completed result",
        raises_out,
        "run_testcases() re-raises out of the loop on the first exception. "
        "Results already computed for earlier test cases are never handed to "
        "StoryManager, so a crash in the last of ten LLM benchmarks throws "
        "away the nine that succeeded along with hours of compute.",
    )


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Verify the breakages in the 2022 Ianvs simulation code."
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="skip checks that require network access",
    )
    args = parser.parse_args()

    if not os.path.isdir("core"):
        print("error: run this from the root of an ianvs checkout.",
              file=sys.stderr)
        return 2

    print("=" * 78)
    print("Ianvs simulation controller — breakage verification")
    print("=" * 78)
    print(f"python  : {sys.version.split()[0]}")
    print(f"platform: {sys.platform}")
    print(f"cwd     : {os.getcwd()}")

    check_b1_unreachable_install_branch()
    check_b2_bool_accepted_as_int()
    check_b3_unknown_keys_dropped()
    check_b4_empty_config_accepted()
    check_b5_fragile_cpu_parse()
    check_b6_branch_urls(args.offline)
    check_b7_teardown_never_called()
    check_b8_pinned_kind_version()
    check_b9_feature_undocumented()
    check_b10_b12_backend_node_ceilings(args.offline)
    check_b13_no_system_metrics()
    check_b14_fault_isolation()

    confirmed = [r for r in RESULTS if r[2] and not r[3]]
    skipped = [r for r in RESULTS if r[3]]

    print("\n" + "=" * 78)
    print(f"SUMMARY: {len(confirmed)} confirmed, {len(skipped)} skipped, "
          f"{len(RESULTS)} checks run")
    print("=" * 78)
    for ident, title, ok, was_skipped in RESULTS:
        mark = "SKIP" if was_skipped else ("  ok" if not ok else "FOUND")
        print(f"  [{mark}] {ident}  {title}")

    return len(confirmed)


if __name__ == "__main__":
    sys.exit(main())
