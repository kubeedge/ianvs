#!/usr/bin/env python3
"""Static validator for Ianvs example configs.

Walks every examples/*/benchmarkingjob.yaml and checks that its testenv,
algorithm, and module (python `url`) references all point to files that
exist. No torch/sedna/GPU needed, runs in seconds - built to catch the
"stale/broken path" bug class before it reaches a user.

Usage: python3 scripts/validate_example_configs.py [--baseline FILE] [--strict]
Exit code is non-zero if any (non-baselined) problems were found.
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
    """Collapse to one line and strip REPO_ROOT (PyYAML embeds the absolute
    path in errors, which would otherwise differ per machine/CI runner and
    break baseline matching)."""
    text = " ".join(str(msg).split())
    text = text.replace(str(REPO_ROOT), ".")
    return text


def resolve(base: Path, ref: str) -> Path:
    """Resolve relative to REPO_ROOT, not `base`'s directory - matches how
    Ianvs itself resolves config paths (core/cmd/benchmarking.py ->
    utils.yaml2dict/is_local_file), since every README runs `ianvs -f
    examples/...` from the repo root."""
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
    """Dataset paths are reported separately, non-blocking: data is
    downloaded per-README, not committed, so absence is expected."""
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
                         help="known-problems file (one per line). Baselined "
                              "problems don't fail the build; only new ones do.")
    parser.add_argument("--write-baseline", type=str, default=None,
                         help="write all current problems to this path as a baseline.")
    args = parser.parse_args()

    bj_files = sorted(EXAMPLES_DIR.rglob("benchmarkingjob*.yaml"))
    print(f"Found {len(bj_files)} benchmarkingjob config(s) under examples/\n")

    problems = []
    dataset_notes = []
    for bj in bj_files:
        check_benchmarkingjob(bj, problems, dataset_notes, args.strict)

    if args.write_baseline:
        with open(args.write_baseline, "w", encoding="utf-8") as f:
            f.write("# Known problems as of generation. New ones fail CI;\n")
            f.write("# these don't, until fixed and removed from this file.\n")
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