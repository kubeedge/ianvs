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

"""Process-tier sandbox: one worker process per test case, no cluster,
Docker or root required.

Isolation, concretely:

- Python dependencies: a throwaway ``venv`` per test case, seeded from the
  parent's site-packages so heavy shared deps are not re-downloaded
  (``isolation: venv``), or the parent interpreter reused for speed when
  the caller knows dependencies do not conflict (``isolation: none``).
- ``sys.path``/``PYTHONPATH``: rebuilt from scratch in the child.
- Environment variables: an **allowlist**, not a denylist — a leak is then
  a bug in one visible list rather than an unbounded surface.
- Wall clock: ``timeout`` sends SIGTERM to the worker's process group, then
  SIGKILL after a grace period.
"""

import os
import pickle
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import venv

from core.testcasecontroller.simulation.profiler import (
    ProfileResult, TreeSampler, detect_oom_kill,
)
from core.testcasecontroller.simulation.sandbox.base import Sandbox

# The parent's environment does not leak into a worker except for these.
_ALLOWLISTED_ENV_VARS = (
    "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR",
    "VIRTUAL_ENV", "CONDA_PREFIX", "SSL_CERT_FILE", "SSL_CERT_DIR",
)

_KILL_GRACE_PERIOD_S = 5


class ProcessSandbox(Sandbox):
    """Runs each test case as a child process with its own transient runtime."""

    def __init__(self, config, ianvs_root):
        super().__init__(config, ianvs_root)
        self._workdirs = []

    def _make_workdir(self):
        workdir = tempfile.mkdtemp(prefix="ianvs-sbx-")
        self._workdirs.append(workdir)
        return workdir

    def _build_env(self, workdir):
        """
        Build the allowlisted environment for one worker.

        ``PYTHONPATH`` points at the test case's own private workdir, not
        the Ianvs checkout: the worker module itself is found because it is
        launched with ``python -m`` from ``cwd=ianvs_root``, which Python
        puts on ``sys.path`` automatically.
        """
        env = {
            key: os.environ[key]
            for key in _ALLOWLISTED_ENV_VARS if key in os.environ
        }
        env["PYTHONPATH"] = workdir
        env["IANVS_SANDBOX"] = "1"
        return env

    def _interpreter_for(self, workdir, requirements_file=None):
        if self.config.isolation != "venv":
            return sys.executable

        venv_dir = os.path.join(workdir, ".venv")
        venv.create(venv_dir, with_pip=True, system_site_packages=True)
        python = os.path.join(venv_dir, "bin", "python")
        if requirements_file and os.path.isfile(requirements_file):
            subprocess.run(
                [python, "-m", "pip", "install", "-q", "-r", requirements_file],
                check=True,
            )
        return python

    @staticmethod
    def _make_preexec_fn(memory_bytes, cpus):
        """
        Build the child-side setup that turns a quota from a number into an
        enforced ceiling.

        ``RLIMIT_AS`` bounds *virtual* address space, and ML runtimes
        routinely reserve far more of that than they touch, so this can
        reject a workload that would have fit comfortably in its resident
        footprint. cgroup v2 ``memory.max`` would be the more accurate
        mechanism, but it requires a writable cgroup delegation that is not
        guaranteed to be available; ``RLIMIT_AS`` needs nothing beyond a
        POSIX process and is honestly reported as the fallback it is, not
        silently assumed to be the accurate one.
        """
        def _limit():
            if memory_bytes:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                except (ValueError, OSError):
                    pass
            if cpus:
                try:
                    available = sorted(os.sched_getaffinity(0))
                    pinned = available[:max(1, int(cpus))] or available
                    os.sched_setaffinity(0, pinned)
                except (AttributeError, OSError, ValueError):
                    pass
        return _limit

    @staticmethod
    def _kill_process_group(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=_KILL_GRACE_PERIOD_S)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # pylint: disable=too-many-locals,too-many-statements
    def run_testcase(self, testcase, workspace):
        """Run one test case as a resource-bounded, isolated child process."""
        workdir = self._make_workdir()
        profile = ProfileResult()
        profile.quota_memory_bytes = self.config.quota.memory_bytes

        payload_path = os.path.join(workdir, "testcase.pkl")
        result_path = os.path.join(workdir, "result.pkl")
        with open(payload_path, "wb") as handle:
            pickle.dump({"testcase": testcase, "workspace": workspace}, handle)

        requirements_file = getattr(testcase.algorithm, "requirements_file", None)
        python = self._interpreter_for(workdir, requirements_file)
        env = self._build_env(workdir)
        cmd = [python, "-m", "core.testcasecontroller.simulation.worker",
               payload_path, result_path]

        preexec_fn = self._make_preexec_fn(
            self.config.quota.memory_bytes, self.config.quota.cpus)

        start = time.time()
        # pylint: disable=consider-using-with
        # preexec_fn's usual multithreading hazard doesn't apply here: no
        # sampler or other thread exists yet in this process at fork time,
        # since TreeSampler.start() below only runs after Popen returns.
        # pylint: disable-next=subprocess-popen-preexec-fn
        process = subprocess.Popen(
            cmd, cwd=self.ianvs_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, text=True, preexec_fn=preexec_fn,
        )
        sampler = TreeSampler(pid=process.pid)
        sampler.start()

        timeout = self.config.quota.timeout
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            _, stderr = process.communicate()
            profile.error = f"test case exceeded its {timeout}s timeout and was killed."
        finally:
            sampler.stop()

        wall_time_s = round(time.time() - start, 3)
        profile.wall_time_s = wall_time_s
        profile.exit_code = process.returncode

        sample = sampler.result(wall_time_s=wall_time_s)
        profile.peak_memory_bytes = sample["peak_memory_bytes"]
        profile.mean_memory_bytes = sample["mean_memory_bytes"]
        profile.memory_source = sample["memory_source"]
        profile.sample_count = sample["sample_count"]
        profile.cpu_time_s = sample["cpu_time_s"]
        profile.cpu_utilization_pct = sample["cpu_utilization_pct"]

        # worker.py always writes result_path before it exits, whether the
        # test case succeeded or raised -- a non-zero exit with a result
        # file present means the test case failed *inside* a still-
        # functioning worker, which has already captured a precise
        # traceback (e.g. the RLIMIT_AS quota rejecting an allocation
        # raises a catchable Python MemoryError, not a kernel SIGKILL).
        # Prefer that traceback over a generic "exited with code N", and
        # only fall back to raw stderr/returncode when result_path is
        # missing, meaning the worker never got that far (import error,
        # segfault, or an actual kernel-level SIGKILL).
        outcome = None
        if os.path.isfile(result_path):
            with open(result_path, "rb") as handle:
                outcome = pickle.load(handle)
            profile.succeeded = outcome.get("succeeded", False)
            profile.error = outcome.get("error")
        else:
            profile.succeeded = False
            profile.error = profile.error or (
                stderr.strip()[-2000:] if stderr else
                f"worker exited with code {process.returncode} and produced no result.")

        profile.oom_killed = detect_oom_kill(
            process.returncode, f"{stderr or ''}\n{profile.error or ''}")

        if profile.oom_killed:
            profile.succeeded = False
            quota_mib = (profile.quota_memory_bytes or 0) / 2 ** 20
            peak_mib = (profile.peak_memory_bytes or 0) / 2 ** 20
            profile.error = (
                "test case was OOM-killed inside the sandbox. The declared "
                f"memory quota was {quota_mib:.0f}MiB; peak observed was "
                f"{peak_mib:.2f}MiB. The Ianvs process itself was "
                f"unaffected. (worker detail: {profile.error})")
            return None, profile

        if not profile.succeeded:
            return None, profile

        return outcome.get("result"), profile

    def teardown(self):
        """Remove every test case's private workdir, unless keep_workdir is set."""
        if getattr(self.config, "keep_workdir", False):
            return
        for workdir in self._workdirs:
            shutil.rmtree(workdir, ignore_errors=True)
        self._workdirs = []
