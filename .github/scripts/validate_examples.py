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

"""Validation script for Ianvs example YAML and Python files."""

import os
import sys
import py_compile

try:
    import yaml
except ImportError:
    yaml = None


def validate_examples(examples_dir):
    failed = False
    yaml_count = 0
    py_count = 0

    print(f"Validating Ianvs examples in {examples_dir}...")

    for root, _, files in os.walk(examples_dir):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, examples_dir)

            if file.endswith((".yaml", ".yml")):
                yaml_count += 1
                if yaml is not None:
                    try:
                        with open(filepath, "r", encoding="utf-8") as yf:
                            yaml.safe_load(yf)
                    except Exception as err:
                        print(f"[FAIL] YAML syntax error in {rel_path}: {err}")
                        failed = True

            elif file.endswith(".py"):
                py_count += 1
                try:
                    py_compile.compile(filepath, doraise=True)
                except py_compile.PyCompileError as err:
                    print(f"[FAIL] Python syntax error in {rel_path}: {err}")
                    failed = True

    print(f"Validated {yaml_count} YAML files and {py_count} Python files.")

    if failed:
        print("Example validation FAILED.")
        sys.exit(1)
    else:
        print("All example YAML and Python files passed validation successfully.")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    examples_path = os.path.join(repo_root, "examples")
    validate_examples(examples_path)
