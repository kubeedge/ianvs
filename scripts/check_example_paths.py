import argparse
import os
import sys
from pathlib import Path
import yaml
def find_benchmarkingjob_files(examples_dir):
    results = []
    for root, _dirs, files in os.walk(examples_dir):
        for filename in files:
            if (filename.startswith("benchmarkingjob")
                    and (filename.endswith(".yaml")
                         or filename.endswith(".yml"))):
                results.append(Path(root) / filename)
    return sorted(results)
def parse_yaml_safe(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception as exc:  
        print(f"  [WARN] Could not parse {filepath}: {exc}", file=sys.stderr)
        return None
def resolve_path(raw_path, repo_root):
    if raw_path.startswith("./"):
        return repo_root / raw_path[2:]
    if os.path.isabs(raw_path):
        return Path(raw_path)
    return repo_root / raw_path
def check_path_exists(raw_path, repo_root, source_file, field_name,
                      broken_paths, verbose=False):
    if not raw_path or not isinstance(raw_path, str):
        return
    resolved = resolve_path(raw_path, repo_root)
    rel_source = source_file.relative_to(repo_root)
    if resolved.exists():
        if verbose:
            print(f"  [OK]     {rel_source}: {field_name} -> {raw_path}")
    else:
        record = {
            "source": str(rel_source),
            "field": field_name,
            "path": raw_path,
        }
        broken_paths.append(record)
        print(f"  [BROKEN] {rel_source}: {field_name} -> {raw_path}")
def check_testenv(testenv_path, repo_root, broken_paths, verbose=False):
    data = parse_yaml_safe(testenv_path)
    if not data:
        return
    testenv = data.get("testenv", data)
    metrics = testenv.get("metrics")
    if isinstance(metrics, list):
        for idx, metric in enumerate(metrics):
            if isinstance(metric, dict):
                url = metric.get("url")
                if url:
                    check_path_exists(
                        url, repo_root, testenv_path,
                        f"metrics[{idx}].url", broken_paths, verbose
                    )
    model_eval = testenv.get("model_eval")
    if isinstance(model_eval, dict):
        model_metric = model_eval.get("model_metric")
        if isinstance(model_metric, dict):
            url = model_metric.get("url")
            if url:
                check_path_exists(
                    url, repo_root, testenv_path,
                    "model_eval.model_metric.url", broken_paths, verbose
                )
def check_algorithm(algorithm_path, repo_root, broken_paths, verbose=False):
    data = parse_yaml_safe(algorithm_path)
    if not data:
        return
    algorithm = data.get("algorithm", data)
    modules = algorithm.get("modules")
    if isinstance(modules, list):
        for idx, module in enumerate(modules):
            if isinstance(module, dict):
                url = module.get("url")
                if url:
                    check_path_exists(
                        url, repo_root, algorithm_path,
                        f"modules[{idx}].url", broken_paths, verbose
                    )
def check_benchmarkingjob(filepath, repo_root, broken_paths, verbose=False):
    data = parse_yaml_safe(filepath)
    if not data:
        return
    job = data.get("benchmarkingjob", data)
    testenv_raw = job.get("testenv")
    if testenv_raw:
        testenv_resolved = resolve_path(testenv_raw, repo_root)
        check_path_exists(
            testenv_raw, repo_root, filepath,
            "testenv", broken_paths, verbose
        )
        if testenv_resolved.exists():
            check_testenv(testenv_resolved, repo_root, broken_paths, verbose)
    test_object = job.get("test_object", {})
    algorithms = test_object.get("algorithms", [])
    if isinstance(algorithms, list):
        for idx, algo in enumerate(algorithms):
            if not isinstance(algo, dict):
                continue
            url = algo.get("url")
            if url:
                algo_resolved = resolve_path(url, repo_root)
                check_path_exists(
                    url, repo_root, filepath,
                    f"algorithms[{idx}].url", broken_paths, verbose
                )
                if algo_resolved.exists():
                    check_algorithm(
                        algo_resolved, repo_root, broken_paths, verbose
                    )
def load_allowlist(filepath):
    entries = set()
    path = Path(filepath)
    if not path.exists():
        print(f"[WARN] Allowlist file not found: {filepath}", file=sys.stderr)
        return entries
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                entries.add(stripped)
    return entries
def record_to_allowlist_key(record):
    source = record["source"].replace("\\", "/")
    return f"{source}:{record['field']}:{record['path']}"
def main():
    parser = argparse.ArgumentParser(
        description="Validate example config paths in the Ianvs repository."
    )
    parser.add_argument(
        "--allowlist",
        help="Path to a file listing known broken paths to skip.",
        default=None,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all checked paths, not just broken ones.",
    )
    parser.add_argument(
        "--repo-root",
        help="Repository root directory (auto-detected if not specified).",
        default=None,
    )
    args = parser.parse_args()
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        print(f"[ERROR] examples/ directory not found at {examples_dir}",
              file=sys.stderr)
        sys.exit(2)
    allowlist = set()
    if args.allowlist:
        allowlist = load_allowlist(args.allowlist)
        if allowlist:
            print(f"Loaded {len(allowlist)} known broken path(s) "
                  f"from allowlist.\n")
    job_files = find_benchmarkingjob_files(examples_dir)
    print(f"Found {len(job_files)} benchmarkingjob file(s) to check.\n")
    all_broken = []
    for job_file in job_files:
        rel_path = job_file.relative_to(repo_root)
        if args.verbose:
            print(f"Checking {rel_path} ...")
        check_benchmarkingjob(job_file, repo_root, all_broken, args.verbose)
    new_broken = []
    known_broken = []
    for record in all_broken:
        key = record_to_allowlist_key(record)
        if key in allowlist:
            known_broken.append(record)
        else:
            new_broken.append(record)
    print(f"\n{'=' * 60}")
    print("Config Path Validation Summary")
    print(f"{'=' * 60}")
    print(f"Files checked:        {len(job_files)}")
    print(f"Total broken paths:   {len(all_broken)}")
    print(f"Known (allowlisted):  {len(known_broken)}")
    print(f"NEW broken paths:     {len(new_broken)}")
    print(f"{'=' * 60}")
    if new_broken:
        print("\nNEW broken paths (not in allowlist):\n")
        for record in new_broken:
            source = record["source"].replace("\\", "/")
            print(f"  {source}:{record['field']}:{record['path']}")
        print("\nTo allowlist these, add the lines above to your "
              "allowlist file.")
        sys.exit(1)
    if all_broken and not new_broken:
        print("\nAll broken paths are in the allowlist. CI passes.")
    if not all_broken:
        print("\nNo broken paths found. All config references are valid!")
    sys.exit(0)
if __name__ == "__main__":
    main()
