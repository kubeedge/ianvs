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

"""Module ②, the Sedna Contract Check.

Lifelong learning is the paradigm that hands control to Sedna, and Sedna is
vendored as a wheel rather than declared as a dependency. This module checks
that the vendored wheel is present, that the installed `sedna` is that
vendored version (not the unrelated PyPI package), and that every Sedna
symbol Ianvs core imports resolves in the installed Sedna.

The installed Sedna is inspected in a subprocess using the runtime Python, so
the validator process never imports it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

VENDORED_WHEEL_DIR = "resources/third_party"
WHEEL_VERSION_RE = re.compile(r"^sedna-(?P<version>[^-]+)-.*\.whl$")

_PROBE = r"""
import importlib, importlib.metadata, json, sys
symbols = json.loads(sys.argv[1])
try:
    version = importlib.metadata.version("sedna")
except importlib.metadata.PackageNotFoundError:
    version = None
missing = []
if version is not None:
    for module_name, attribute in symbols:
        try:
            target = importlib.import_module(module_name)
            for part in [p for p in attribute.split(".") if p]:
                target = getattr(target, part)
        except Exception as exc:  # report every failure, not only ImportError
            missing.append(f"{module_name}{'.' + attribute if attribute else ''}: "
                           f"{type(exc).__name__}: {exc}")
print(json.dumps({"version": version, "missing": missing}))
"""


@dataclass
class ContractResult:
    name: str
    status: str  # "PASS", "FAIL" or "ERROR"
    message: str
    details: List[str] = field(default_factory=list)


def vendored_versions(repo_root: Path) -> List[str]:
    wheel_dir = repo_root / VENDORED_WHEEL_DIR
    if not wheel_dir.is_dir():
        return []
    versions = []
    for wheel in sorted(wheel_dir.glob("sedna-*.whl")):
        match = WHEEL_VERSION_RE.match(wheel.name)
        if match:
            versions.append(match.group("version"))
    return versions


def core_sedna_symbols(repo_root: Path) -> List[Tuple[str, str]]:
    """(module, attribute) pairs for every `sedna` import in Ianvs core."""
    symbols = set()
    for path in sorted((repo_root / "core").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "sedna" or node.module.startswith("sedna."):
                    for alias in node.names:
                        if alias.name != "*":
                            symbols.add((node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "sedna" or alias.name.startswith("sedna."):
                        symbols.add((alias.name, ""))
    return sorted(symbols)


def check_sedna_contract(
    repo_root: Path,
    python: Optional[str] = None,
    timeout_seconds: int = 120,
    extra_symbols: Sequence[Tuple[str, str]] = (),
) -> List[ContractResult]:
    results = []
    versions = vendored_versions(repo_root)
    results.append(ContractResult(
        name="Vendored Sedna wheel present",
        status="PASS" if versions else "FAIL",
        message=(
            f"Vendored Sedna wheel(s) found in {VENDORED_WHEEL_DIR}: {', '.join(versions)}."
            if versions else
            f"No sedna-*.whl found in {VENDORED_WHEEL_DIR}. Impact: lifelong learning "
            "cannot be installed reproducibly."
        ),
    ))

    symbols = sorted(set(core_sedna_symbols(repo_root)) | set(extra_symbols))
    try:
        completed = subprocess.run(
            [python or sys.executable, "-c", _PROBE, json.dumps(symbols)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout_seconds,
            check=False,
        )
        probe = json.loads(completed.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        results.append(ContractResult(
            name="Installed Sedna matches vendored wheel",
            status="ERROR",
            message=f"Could not inspect the installed Sedna: {type(exc).__name__}.",
        ))
        return results

    installed = probe.get("version")
    if installed is None:
        status, message = "FAIL", "Sedna is not installed in the runtime environment."
    elif installed in versions:
        status, message = "PASS", f"Installed Sedna {installed} is the vendored version."
    else:
        status = "FAIL"
        message = (
            f"Installed Sedna {installed} is not a vendored version "
            f"({', '.join(versions) or 'none found'}). Impact: core may run against a "
            "different Sedna API, or the unrelated PyPI package."
        )
    results.append(ContractResult("Installed Sedna matches vendored wheel", status, message))

    if installed is None:
        return results
    missing = probe.get("missing", [])
    results.append(ContractResult(
        name="Sedna symbols used by core resolve",
        status="FAIL" if missing else "PASS",
        message=(
            f"{len(missing)} of {len(symbols)} Sedna symbols imported by core do not resolve. "
            "Impact: the paradigm fails when it reaches them."
            if missing else
            f"All {len(symbols)} Sedna symbols imported by core resolve."
        ),
        details=missing,
    ))
    return results
