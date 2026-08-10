#!/usr/bin/env python3
"""Static validator for Ianvs example configs.

Walks every benchmarkingjob.yaml under examples/, and for each one:
  - confirms it's valid YAML
  - confirms the referenced testenv.yaml exists
  - confirms every algorithm config (algorithms[].config_file / url) exists
  - confirms every module's `url:` (Python file) referenced in the
    algorithm config actually exists on disk
  - confirms testenv.yaml's dataset file references
    (train_url/test_url/train_index/test_index/train_data/test_data/
    train_data_info/test_data_info) exist on disk

This never imports torch/sedna/etc. and never executes example code, so it
runs in seconds and needs no GPU, no network, no heavy dependencies —
suitable as a fast CI gate to catch the class of "broken path / stale
config" bugs that otherwise only surface when a user tries to run an
example and hits a FileNotFoundError.

Usage:
    python3 scripts/validate_example_configs.py [--strict]

Exit code is non-zero if any problems were found (useful for CI).
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

DATASET_URL_KEYS = [
    "train_url", "test_url",
    "train_index", "test_index",
    "train_data", "test_data",
    "train_data_info", "test_data_info",
]


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def oneline(msg: str) -> str:
    """Collapse a possibly-multi-line message (e.g. a YAML parser error)
    into a single line, so every entry in `problems`/`dataset_notes` is
    exactly one line - required for the baseline file's one-problem-per-line
    format to round-trip correctly.

    Also strips the machine-specific absolute REPO_ROOT prefix that PyYAML
    embeds directly in its error messages (e.g. 'in "/Users/you/ianvs/..."').
    Without this, the exact same error produces a different string on every
    contributor's machine and on the CI runner, permanently breaking
    baseline comparison across environments.
    """
    text = " ".join(str(msg).split())
    text = text.replace(str(REPO_ROOT), ".")
    return text


def resolve(base: Path, ref: str) -> Path:
    """Resolve a path referenced inside a config file.

    Ianvs itself never resolves these paths relative to the referencing
    config file - it resolves them relative to the process's current
    working directory (see core/cmd/benchmarking.py -> utils.yaml2dict /
    utils.is_local_file), and every README instructs running
    `ianvs -f examples/.../benchmarkingjob.yaml` from the repo root.
    So: absolute paths are used as-is, and everything else is resolved
    relative to REPO_ROOT, not relative to `base`'s directory.
    """
    ref = ref.strip()
    p = Path(ref)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def check_benchmarkingjob(bj_path: Path, problems: list, dataset_notes: list, strict: bool):
    prefix = str(bj_path.relative_to(REPO_ROOT))
    try:
        cfg = load_yaml(bj_path)
    except yaml.YAMLError as e:
        problems.append(oneline(f"{prefix}: INVALID YAML - {e}"))
        return

    if not cfg or "benchmarkingjob" not in cfg:
        problems.append(f"{prefix}: missing top-level 'benchmarkingjob' key")
        return
    bj = cfg["benchmarkingjob"]

    testenv_ref = bj.get("testenv")
    if not testenv_ref:
        problems.append(f"{prefix}: no 'testenv' key under benchmarkingjob")
    else:
        testenv_path = resolve(bj_path, testenv_ref)
        if not testenv_path.exists():
            problems.append(f"{prefix}: testenv file does not exist -> {testenv_ref}")
        else:
            check_testenv(testenv_path, dataset_notes)

    algorithms = (bj.get("test_object") or {}).get("algorithms", [])
    if not algorithms and strict:
        problems.append(f"{prefix}: no algorithms declared under test_object")

    for algo in algorithms or []:
        algo_name = algo.get("name", "<unnamed>")
        cfg_ref = algo.get("url")
        if not cfg_ref:
            problems.append(f"{prefix}: algorithm '{algo_name}' has no 'url'")
            continue
        algo_cfg_path = resolve(bj_path, cfg_ref)
        if not algo_cfg_path.exists():
            problems.append(
                f"{prefix}: algorithm '{algo_name}' config does not exist -> {cfg_ref}"
            )
            continue
        check_algorithm_config(algo_cfg_path, problems, prefix, algo_name)


def check_testenv(testenv_path: Path, dataset_notes: list):
    """Dataset file references are checked separately from structural
    problems: it's normal/expected for these to be absent in a fresh
    clone (data is downloaded separately per each example's README), so
    these are informational notes, not CI-blocking problems."""
    prefix = str(testenv_path.relative_to(REPO_ROOT))
    try:
        cfg = load_yaml(testenv_path)
    except yaml.YAMLError as e:
        dataset_notes.append(oneline(f"{prefix}: INVALID YAML - {e}"))
        return
    te = (cfg or {}).get("testenv", {})
    dataset = te.get("dataset", {})
    for key in DATASET_URL_KEYS:
        val = dataset.get(key)
        if val:
            path = resolve(testenv_path, val)
            if not path.exists():
                dataset_notes.append(f"{prefix}: dataset.{key} not present locally -> {val}")


def check_algorithm_config(algo_cfg_path: Path, problems: list, bj_prefix: str, algo_name: str):
    prefix = str(algo_cfg_path.relative_to(REPO_ROOT))
    try:
        cfg = load_yaml(algo_cfg_path)
    except yaml.YAMLError as e:
        problems.append(oneline(f"{prefix}: INVALID YAML - {e}"))
        return
    algorithm = (cfg or {}).get("algorithm", {})
    modules = algorithm.get("modules", [])
    for mod in modules:
        mod_type = mod.get("type", "<unknown type>")
        url = mod.get("url")
        if not url:
            continue
        module_path = resolve(algo_cfg_path, url)
        if not module_path.exists():
            problems.append(
                f"{prefix}: module '{mod_type}' url does not exist -> {url} "
                f"(referenced from {bj_prefix}, algorithm '{algo_name}')"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                         help="also flag missing-but-optional fields")
    parser.add_argument("--baseline", type=str, default=None,
                         help="path to a baseline file listing currently-known "
                              "problems (one exact problem string per line, "
                              "'#'-prefixed lines and blank lines ignored). "
                              "Problems in the baseline are reported but do not "
                              "cause a non-zero exit; only NEW problems do. This "
                              "lets the check be added to CI immediately without "
                              "blocking on pre-existing, already-tracked breakage, "
                              "while still catching new regressions right away.")
    parser.add_argument("--write-baseline", type=str, default=None,
                         help="instead of validating, write every current "
                              "problem to this path as a fresh baseline file.")
    args = parser.parse_args()

    bj_files = sorted(EXAMPLES_DIR.rglob("benchmarkingjob*.yaml"))
    print(f"Found {len(bj_files)} benchmarkingjob config(s) under examples/\n")

    problems = []
    dataset_notes = []
    for bj in bj_files:
        check_benchmarkingjob(bj, problems, dataset_notes, args.strict)

    if args.write_baseline:
        with open(args.write_baseline, "w", encoding="utf-8") as f:
            f.write("# Baseline of known structural problems as of the date this\n")
            f.write("# file was generated. New problems not listed here will fail\n")
            f.write("# CI; problems listed here are still reported but non-blocking\n")
            f.write("# until fixed and removed from this file.\n")
            for p in problems:
                f.write(p + "\n")
        print(f"Wrote {len(problems)} problem(s) to baseline file: {args.write_baseline}")
        return 0

    baseline = set()
    if args.baseline and Path(args.baseline).exists():
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline = {line.strip() for line in f
                        if line.strip() and not line.strip().startswith("#")}

    new_problems = [p for p in problems if p not in baseline]
    known_problems = [p for p in problems if p in baseline]

    if dataset_notes:
        print(f"{len(dataset_notes)} dataset file(s) not present locally "
              f"(expected - normally downloaded separately per each README):\n")
        for n in dataset_notes:
            print(f"  - {n}")
        print()

    if known_problems:
        print(f"{len(known_problems)} known, already-tracked problem(s) "
              f"(non-blocking, see scripts/example_config_baseline.txt):\n")
        for p in known_problems:
            print(f"  - {p}")
        print()

    if new_problems:
        print(f"{len(new_problems)} NEW structural problem(s) found "
              f"(not in baseline - this is what fails CI):\n")
        for p in new_problems:
            print(f"  - {p}")
        print()
        return 1

    print("No new structural problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
