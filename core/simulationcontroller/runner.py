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
Entrypoint of a simulation job's pod: rebuild one test case from the same config
files the host read, run it, and report the result.

The result is written as JSON to the pod's termination message, which the kubelet
(or edgecore on an edge node) reports in the pod status, so the host reads it through
the Kubernetes API; pods on KubeEdge edge nodes cannot reach the API server to write
a ConfigMap themselves. A copy is kept in the test case's output directory.
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid

RESULT_FILE = "sandbox_result.json"
TERMINATION_LOG = "/dev/termination-log"
# the kubelet keeps at most 4096 bytes of a termination message
MAX_MESSAGE_BYTES = 4096
# the file the host writes into the output directory once the pod's network emulation
# is in place; the runner waits for it, so the test case never runs unshaped
NETWORK_READY_FILE = ".sandbox-network-ready"
NETWORK_WAIT_S = 300


def wait_for_network(output_dir, timeout=NETWORK_WAIT_S, interval=0.5):
    """wait until the host has applied the network emulation; True when it has."""
    marker = os.path.join(output_dir, NETWORK_READY_FILE)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(marker):
            try:
                os.remove(marker)
            except OSError:
                pass
            return True
        time.sleep(interval)
    return False


def to_jsonable(value):
    """convert metric values (numpy scalars and arrays included) to JSON types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    for method in ("item", "tolist"):
        if hasattr(value, method):
            try:
                return to_jsonable(getattr(value, method)())
            except (TypeError, ValueError):
                continue
    return str(value)


def _report(result, output_dir, termination_log):
    payload = json.dumps(result)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, RESULT_FILE), "w", encoding="utf-8") as file:
            file.write(payload)
    if len(payload.encode("utf-8")) > MAX_MESSAGE_BYTES:
        # too large for the pod status; the host falls back to the copied result file
        payload = json.dumps({"status": result["status"], "result_file": RESULT_FILE})
    try:
        with open(termination_log, "w", encoding="utf-8") as file:
            file.write(payload)
    except OSError:
        # not running in a pod, e.g. in tests
        pass


def run(args):
    """rebuild and run the test case; returns the result dict."""
    # imported here so a broken environment is reported as an EXEC_FAILURE result
    # pylint: disable=import-outside-toplevel
    from core.common import utils
    from core.testenvmanager.testenv import TestEnv
    from core.testcasecontroller.algorithm import Algorithm
    from core.testcasecontroller.testcase import TestCase

    test_env = TestEnv(utils.yaml2dict(args.testenv))
    test_env.prepare()

    algorithm = Algorithm(args.algorithm_name, utils.yaml2dict(args.algorithm_url))
    # the host expanded the hyperparameters into the same ordered list of combinations
    algorithm.modules = algorithm.modules_list[args.modules_index]

    testcase = TestCase(test_env, algorithm)
    # keep the host's id, so the output directory is the one the host expects
    testcase.id = uuid.UUID(args.testcase_id)
    # the host already created (and mounted) that directory, and TestCase looks for
    # one that does not exist yet, so it is pinned here
    output_dir = os.path.join(args.workspace, args.algorithm_name, args.testcase_id)
    testcase._get_output_dir = lambda _workspace: output_dir  # pylint: disable=protected-access
    metrics = testcase.run(args.workspace)
    return {"testcase_id": args.testcase_id, "status": "SUCCESS",
            "metrics": to_jsonable(metrics), "output_dir": testcase.output_dir}


def main(argv=None, termination_log=TERMINATION_LOG):
    """parse arguments, run the test case and report; returns the exit code."""
    parser = argparse.ArgumentParser(description="run one Ianvs test case in a simulation job")
    parser.add_argument("--testenv", required=True)
    parser.add_argument("--algorithm-name", required=True)
    parser.add_argument("--algorithm-url", required=True)
    parser.add_argument("--modules-index", type=int, required=True)
    parser.add_argument("--testcase-id", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wait-network", action="store_true",
                        help="wait until the host has applied the network emulation")
    args = parser.parse_args(argv)

    if args.wait_network and not wait_for_network(args.output_dir):
        result = {"testcase_id": args.testcase_id, "status": "EXEC_FAILURE",
                  "error": "the network emulation was not applied within "
                           f"{NETWORK_WAIT_S} s; the test case did not run"}
        _report(result, args.output_dir, termination_log)
        return 1

    # start and end lines in the pod's log (saved as sandbox.log), so a test case that is
    # killed, e.g. for exceeding its memory limit, still shows how far it got
    print(f"testcase(id={args.testcase_id}) of {args.algorithm_name} started", flush=True)
    start = time.monotonic()
    try:
        result = run(args)
        code = 0
    except Exception as err:  # pylint: disable=broad-except
        traceback.print_exc()
        result = {"testcase_id": args.testcase_id, "status": "EXEC_FAILURE",
                  "error": str(err)[-1000:]}
        code = 1
    # the test case's own wall time, for the system metrics
    result["wall_time_s"] = round(time.monotonic() - start, 3)
    _report(result, args.output_dir, termination_log)
    print(f"testcase(id={args.testcase_id}) finished: {result['status']}", flush=True)
    return code


if __name__ == "__main__":
    # the algorithm under test may parse the command line itself (e.g. TensorFlow 1 flags
    # in the pcb-aoi FPN model), so it must not see the runner's own arguments
    RUNNER_ARGS, sys.argv = sys.argv[1:], sys.argv[:1]
    sys.exit(main(RUNNER_ARGS))
