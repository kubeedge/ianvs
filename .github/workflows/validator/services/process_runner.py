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

"""Run validator commands with bounded cleanup of their POSIX process group."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Mapping, Optional, Sequence


OUTPUT_DRAIN_TIMEOUT_SECONDS = 1


def run_command(
    command: Sequence[str],
    *,
    cwd: str,
    timeout: float,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Capture combined output, killing the command's group on timeout.

    POSIX children inherit a private session so a timed-out preparation or
    benchmark cannot leave its ordinary workers running. Processes that detach
    into their own session are outside this boundary. On non-POSIX platforms
    only the direct child is terminated, as with subprocess.run().
    """
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=os.name == "posix",
    ) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _kill_command(process)
            try:
                output, _ = process.communicate(timeout=OUTPUT_DRAIN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as drain_error:
                # A detached descendant may still own the pipe. Do not let
                # draining its output turn a command timeout into an idle hang.
                output = drain_error.output or error.output or b""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
            raise subprocess.TimeoutExpired(command, timeout, output=output) from error
        except BaseException:
            # Cancellation should clean up the same workers as a timeout.
            _kill_command(process)
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout=output)


def _kill_command(process: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            # Even if the group leader has exited, its workers can still be
            # alive and holding stdout open. Do not guard this with poll().
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()
