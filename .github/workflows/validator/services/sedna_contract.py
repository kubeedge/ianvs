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
symbol Ianvs core imports resolves in the installed Sedna. Imports guarded by
`try` / `except ImportError` are alternatives: one of them must resolve, and a
fallback without a Sedna import makes the import optional.

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
            missing.append([module_name, attribute, f"{type(exc).__name__}: {exc}"])
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


_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _core_trees(repo_root: Path) -> List[ast.AST]:
    trees = []
    for path in sorted((repo_root / "core").rglob("*.py")):
        try:
            trees.append(ast.parse(path.read_text(encoding="utf-8")))
        except (SyntaxError, UnicodeDecodeError):
            continue
    return trees


def _sedna_symbols_in(nodes: Sequence[ast.AST]) -> List[Tuple[str, str]]:
    symbols = set()
    for root in nodes:
        for node in ast.walk(root):
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


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(getattr(t, "id", getattr(t, "attr", None)) in _IMPORT_ERRORS for t in types)


def _guarded_tries(tree: ast.AST) -> List[ast.Try]:
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Try) and any(map(_catches_import_error, node.handlers))]


def core_sedna_symbols(repo_root: Path) -> List[Tuple[str, str]]:
    """(module, attribute) pairs for every unguarded `sedna` import in Ianvs core."""
    symbols = set()
    for tree in _core_trees(repo_root):
        guarded = set(_sedna_symbols_in(_guarded_tries(tree)))
        symbols.update(set(_sedna_symbols_in([tree])) - guarded)
    return sorted(symbols)


def core_sedna_alternatives(repo_root: Path) -> List[List[List[Tuple[str, str]]]]:
    """For each `try` / `except ImportError` around a `sedna` import in Ianvs
    core, the symbols of each branch: the `try` body first, then each handler.
    An empty branch is a fallback that needs no Sedna."""
    groups = []
    for tree in _core_trees(repo_root):
        for node in _guarded_tries(tree):
            branches = [_sedna_symbols_in(node.body)] + [
                _sedna_symbols_in(handler.body)
                for handler in node.handlers if _catches_import_error(handler)]
            if any(branches):
                groups.append(branches)
    return groups


def _describe(symbol: Tuple[str, str], error: str) -> str:
    module_name, attribute = symbol
    return f"{module_name}{'.' + attribute if attribute else ''}: {error}"


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

    required = set(core_sedna_symbols(repo_root)) | set(extra_symbols)
    groups = core_sedna_alternatives(repo_root)
    symbols = sorted(required | {s for branches in groups for b in branches for s in b})
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
    errors = {(module_name, attribute): error
              for module_name, attribute, error in probe.get("missing", [])}
    missing = [_describe(s, errors[s]) for s in sorted(required) if s in errors]
    for branches in groups:
        # A group fails only when no branch resolves; an empty branch always does.
        if all(any(s in errors for s in branch) for branch in branches):
            missing.extend(_describe(s, errors[s]) for s in branches[0] if s in errors)
    results.append(ContractResult(
        name="Sedna symbols used by core resolve",
        status="FAIL" if missing else "PASS",
        message=(
            f"{len(missing)} Sedna symbols imported by core do not resolve. "
            "Impact: the paradigm fails when it reaches them."
            if missing else
            f"All {len(required)} Sedna symbols imported by core resolve, and each of the "
            f"{len(groups)} imports guarded by `except ImportError` has a working alternative."
        ),
        details=missing,
    ))
    return results
