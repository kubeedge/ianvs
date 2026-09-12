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

"""The sandbox's inner worker process: runs exactly one test case.

Invoked as ``python -m core.testcasecontroller.simulation.worker <payload>
<result>`` by ``ProcessSandbox``, and identically inside a cluster-tier Job's
container. When ``IANVS_RESULT_CONFIGMAP`` is set, the result is additionally
published via ``kubectl create configmap`` so the Job Administrator's
list-watch (``job_admin.py``) can retrieve it -- that path is implemented but
has not been exercised against a live kind/KubeEdge/Sedna cluster; see the
proposal's "Honest limitations" section.
"""

import base64
import os
import pickle
import subprocess
import sys
import traceback


def _publish_configmap(outcome):
    name = os.environ.get("IANVS_RESULT_CONFIGMAP")
    if not name:
        return
    namespace = os.environ.get("IANVS_RESULT_NAMESPACE", "default")
    encoded = base64.b64encode(pickle.dumps(outcome)).decode("ascii")
    subprocess.run(
        ["kubectl", "create", "configmap", name, "-n", namespace,
         f"--from-literal=result={encoded}"],
        check=False,
    )


def _run(payload_path, result_path):
    with open(payload_path, "rb") as handle:
        payload = pickle.load(handle)

    testcase = payload["testcase"]
    workspace = payload["workspace"]

    outcome = {"succeeded": False, "result": None, "error": None}
    try:
        outcome["result"] = testcase.run(workspace)
        outcome["succeeded"] = True
    except Exception as err:  # pylint: disable=broad-except
        # A test case's own failure must not propagate as a worker crash
        # with no diagnostic: capture it so the parent can record it as a
        # normal (non-OOM) failure and continue with the rest of the job.
        outcome["error"] = f"{err}\n{traceback.format_exc()}"

    with open(result_path, "wb") as handle:
        pickle.dump(outcome, handle)
    _publish_configmap(outcome)

    return 0 if outcome["succeeded"] else 1


def main():
    """Entry point: worker.py <payload.pkl> <result.pkl>."""
    if len(sys.argv) != 3:
        print("usage: worker.py <payload.pkl> <result.pkl>", file=sys.stderr)
        return 2
    return _run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
