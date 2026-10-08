# Run Example Validation Locally

Run all commands from the Ianvs repository root. The local CLI uses the same inventory and validator modules as CI.

See [validation rules](validation_rules.md) for what each stage checks and [classification policy](classification_policy.md) for how CI compares base and pull request results.

## Prerequisites

- Python 3.8 or the `python_version` declared by the selected inventory entry
- Git
- A virtual environment for dependency installation and dynamic validation

Install the lightweight validator dependency first:

```bash
python -m venv .venv-validator
. .venv-validator/bin/activate
python -m pip install -r .github/workflows/validator/requirements.txt
```

Dynamic smoke execution also needs Ianvs itself:

```bash
python -m pip install -r requirements.txt
python -m pip install resources/third_party/sedna-0.6.0.1-py3-none-any.whl
python -m pip install -e . --no-deps
```

## Common commands

Run static validation for all active inventory entries:

```bash
python .github/workflows/validator/validation_runner.py --static --all
```

Run static validation for one benchmark unit. `--example` accepts its inventory name, example path, or benchmark file:

```bash
python .github/workflows/validator/validation_runner.py \
  --static \
  --example examples/llm_simple_qa
```

Validate dependency declarations without installing them:

```bash
python .github/workflows/validator/validation_runner.py \
  --dependency \
  --example examples/llm_simple_qa
```

Ask pip to resolve the dependencies without installing them:

```bash
python .github/workflows/validator/validation_runner.py \
  --dependency \
  --pip-install-check \
  --example examples/llm_simple_qa
```

Run the dynamic stages used by CI for `llm_simple_qa`:

```bash
python .github/workflows/validator/validation_runner.py \
  --dependency \
  --pip-install \
  --prepare-env \
  --smoke \
  --example examples/llm_simple_qa
```

This command matches the dynamic stages in the current workflow:

1. `--dependency --pip-install` validates and installs the example dependencies into the active environment.
2. `--prepare-env` executes the ordered `prepare_env.steps` declared by the inventory.
3. `--smoke` validates the configured JSONL dataset structure and then executes the runtime smoke test.

The configured Mock LLM runtime is enabled automatically. Its result is
`mocked_llm` integration evidence, not real-model validation. Use a disposable
environment because `--pip-install` changes installed packages.

Use the standalone `--jsonl` stage when dataset validation is needed without
runtime smoke execution:

```bash
python .github/workflows/validator/validation_runner.py \
  --prepare-env \
  --jsonl \
  --example examples/llm_simple_qa
```

CLI `--timeout` controls dependency installation or resolution checks and the
runtime smoke command. Each `prepare_env.steps[].timeout` controls only that
preparation step and is not overridden by CLI `--timeout`.

For the supported `prepare_env.steps` schema, step validation rules, and legacy compatibility behavior, see the [environment preparation contract](validation_rules.md#environment-preparation-contract).

## Reports and exit codes

Markdown is printed to standard output by default. Save either Markdown or JSON with `--report`:

```bash
python .github/workflows/validator/validation_runner.py \
  --static \
  --example examples/llm_simple_qa \
  --format json \
  --report /tmp/llm-simple-qa-validation.json
```

The command exits `0` when no check has `FAIL` or `ERROR`, and `1` otherwise.
`WARNING` and `SKIP` remain visible but do not change the exit code. GitHub
Actions combines these structured results into Step Summary and artifact
reports; see [classification policy](classification_policy.md) for how CI uses
base and head results.

## Detect affected examples

Fetch the current upstream baseline before comparing with `upstream/main`:

```bash
git remote get-url upstream
git fetch upstream main
```

If `upstream` is not configured, add the official Ianvs repository and fetch it:

```bash
git remote add upstream https://github.com/kubeedge/ianvs.git
git fetch upstream main
```

Use the inventory loader to reproduce CI's target selection between two Git revisions:

```bash
python .github/workflows/validator/services/inventory_loader.py \
  --mode static \
  --base-ref upstream/main \
  --head-ref HEAD

python .github/workflows/validator/services/inventory_loader.py \
  --mode dynamic \
  --base-ref upstream/main \
  --head-ref HEAD
```

The loader reports selected inventory entries and the validation matrix. See
[CI coverage](validation_rules.md#ci-coverage) for the authoritative static
and dynamic file matching, run-all prefixes, workflow triggers, and inactive
entry behavior.

## Optional workflow inspection with act

With Docker and [`act`](https://github.com/nektos/act), contributors can inspect
workflow definitions and list individual jobs:

```bash
act -l -W .github/workflows/static_code_requirement_cicd.yaml
act -l -W .github/workflows/dynamic_code_cicd.yaml
```

The complete multi-job workflow cannot currently be reproduced reliably with
`act`: cross-job `actions/upload-artifact@v7` and
`actions/download-artifact@v8` handoff is affected by
[`nektos/act#6114`](https://github.com/nektos/act/issues/6114). Run
`validation_runner.py` directly for local validation and confirm the final CI
result in GitHub Actions.

## Troubleshooting

- **No inventory examples matched:** check the selector against `name`, `path`, or `benchmark_file` in the inventory.
- **Dynamic validation was skipped:** the selected inventory entry is not `active`.
- **PyYAML unavailable:** install `.github/workflows/validator/requirements.txt` and rerun static validation.
- **Dataset file missing:** run `--prepare-env` before `--jsonl` or `--smoke`, and confirm the inventory dataset layout.
- **Mock Runtime not loaded:** confirm both Mock Runtime directories exist and that the example fixture declares a supported adapter.
- **Dependency installation polluted the environment:** recreate the disposable virtual environment; do not use `--pip-install` in a shared environment.
- **Local action differs from CI:** inspect the uploaded JSON/Markdown artifacts in GitHub Actions and reproduce the underlying `validation_runner.py` command directly.
