# Example Validator Rules

This document describes the checks implemented by the Ianvs example validator. The inventory at [`.github/workflows/validator/data/example_inventory.yaml`](../../.github/workflows/validator/data/example_inventory.yaml) is the source of truth for validation targets and example-specific metadata.

See also:

- [Classification policy](classification_policy.md)
- [Local validation](local_validation.md)
- [Example status directions](status_directions.md)

## Validation concepts

The validator keeps three machine concepts separate:

- **Inventory lifecycle status** is maintained metadata and policy input. An
  exact `status: active` value makes an entry eligible for dynamic stages;
  other values describe inactive lifecycle states.
- **Individual validator result** is the outcome of one check: `PASS`, `FAIL`,
  `ERROR`, `WARNING`, or `SKIP`.
- **Dynamic validation eligibility** determines whether selected entries execute
  dynamic stages. A selected inactive entry receives a
  `Dynamic validation eligibility: SKIP` result instead of executing those
  stages.

These concepts do not directly define pull-request impact or a published health
badge. See [classification policy](classification_policy.md) and
[status directions](status_directions.md), respectively.

## Result levels

Each validator emits one of the following check results:

| Result | Meaning | Directly fails a validator run |
| --- | --- | --- |
| `PASS` | The check completed and found no issue. | No |
| `FAIL` | An execution or validation requirement failed. | Yes |
| `ERROR` | A required file, configuration, or structural rule failed. | Yes |
| `WARNING` | A portability or maintenance risk was found. | No |
| `SKIP` | The check was not applicable or could not run. | No |

`FAIL` and `ERROR` are blocking results inside a single validation report. Whether they block a pull request is decided separately by the regression policy.

## Inventory rules

Every benchmark unit is represented by an inventory entry. An entry should declare:

- a unique benchmark `name` and its top-level `example` group;
- `path`, `benchmark_file`, and `readme_file` when available;
- `requirements_file` when the example has specific dependencies;
- `python_version` for active dynamic validation targets;
- an inventory lifecycle `status`;
- dataset metadata, including `root` and `structure` when JSONL validation is supported;
- optional ordered `prepare_env.steps`;
- optional Mock LLM runtime paths.

The validation unit is a benchmark job, normally identified by one benchmarking YAML file. A top-level example may therefore have several inventory entries and several matrix rows. Do not collapse multiple jobs into one result merely because they share an example directory: their configurations and health can differ.

Affected-example detection may select both active and inactive inventory
entries. Static validation may inspect an explicitly selected inactive entry;
dynamic eligibility follows the rules above.

`prepare_env.steps` is the supported environment-preparation schema for active,
migrated targets. Legacy, unvalidated entries may still contain fields such as
`dataset.prepare_script: null` while awaiting migration; that legacy field is
not the recommended schema for new or activated targets. When an entry has no
`prepare_env` mapping, smoke validation retains a backward-compatible
`dataset.prepare_script` preparation fallback.

## Static validation

Static checks do not execute the example. They inspect the entry, YAML, and Python files under the example path.

| Check | Rule | Failure level |
| --- | --- | --- |
| Example and benchmark paths | The example directory and declared benchmark file must exist. | `ERROR` |
| Requirements and preparation paths | Declared files and scripts must exist. | `ERROR` |
| YAML syntax | Every `.yaml` and `.yml` file must parse with PyYAML. | `ERROR`; `SKIP` if PyYAML is unavailable |
| Repository-local references | Referenced repository Python/YAML paths must resolve. | `ERROR` |
| Other repository path parents | Parent directories for non-code references should exist. | `WARNING` |
| Hardcoded local paths | Contributor-specific absolute POSIX or Windows paths should not appear in Python or YAML. | `WARNING` |
| Local model paths | Model settings should use a portable model ID or a documented override, not `/home/...`, `/Users/...`, or a local `models/` path. | `WARNING` |
| Device selection | Code that selects CUDA must also provide an availability check and CPU fallback. | `WARNING` |
| Metric safety | Metrics that divide by a collection length should guard an empty collection. | `WARNING`; `SKIP` when no metric file exists |

The current static scanner covers `.py`, `.yaml`, and `.yml` files. It does not
scan Markdown or enforce README-specific rules.

### Why a check is an error or a warning

The primary boundary is whether the detected condition is sufficient to prevent the configured validation path from running:

- use `ERROR` or `FAIL` for a missing required file, invalid configuration, failed preparation/install command, failed runtime, or another condition that directly prevents execution;
- use `WARNING` for a portability, maintainability, security, or heuristic finding that may still allow execution;
- do not promote a heuristic to an error merely because the pattern is undesirable. A false positive must not block a pull request.

## Environment preparation contract

If `prepare_env` is present, it must contain a valid `working_directory` and a non-empty ordered `steps` list. Every step requires:

```yaml
prepare_env:
  working_directory: examples/llm_simple_qa
  steps:
    - name: prepare_dataset
      type: dataset
      script: scripts/02_prepare_dataset.py
      args:
        - --output-dir
        - ../../dataset/llm_simple_qa
      timeout: 300
```

`args` must be an array of strings, `timeout` must be a positive integer, and the script must exist below the working directory. The environment preparation validator executes steps in order, without `shell=True`, stops at the first failure, and reports the step name and type. Each `prepare_env.steps[].timeout` applies to that step and is not overridden by the validation runner's CLI `--timeout`.

## Mock Runtime contract

When `mock_runtime.enabled` is `true`, both `shared_pythonpath` and `example_pythonpath` must be non-empty path arrays whose directories exist inside the repository. Adapter selection and semantic responses belong to the example fixture, not the inventory.

## Dependency validation

Dependency validation checks the declared requirements file independently of environment preparation. It verifies:

- the declared file exists and is not empty;
- requirement lines are syntactically valid;
- environment markers allow at least one supported Python version in the
  validator's default `3.8`, `3.9`, and `3.10` matrix;
- imports used by the example runtime are covered by dependency declarations;
- optionally, pip can resolve or install the requirements.

The install modes are:

| CLI option | Behavior |
| --- | --- |
| no install option | Validate declarations only. |
| `--pip-install-check` | Run pip's dry-run resolution check. |
| `--pip-install` | Install the declared requirements into the current environment. |

The validation runner's CLI `--timeout` applies to pip resolution or
installation commands and runtime smoke execution. It is also passed to the
legacy `dataset.prepare_script` fallback, but not to `prepare_env.steps`.

## Dataset and JSONL validation

The validator discovers JSONL files from inventory `dataset.root` plus `dataset.structure`, falling back to the test environment configuration when necessary.

`--jsonl` runs this validation as an independent stage without runtime smoke
execution. `--smoke` performs the same JSONL structure checks before it starts
the runtime command, so the current CI dynamic command does not also pass
`--jsonl`.

The implemented JSONL rules are:

- every declared file must exist;
- test data must not be empty;
- blank rows are invalid;
- each physical line must contain one complete JSON value;
- every row must decode to a JSON object;
- an empty training file is allowed for examples that do not train.

Field-level schema validation, such as requiring `question` and `answer`, is
not currently enforced by the shared validator.

## Smoke validation

The default smoke command is:

```bash
python benchmarking.py -f <benchmark-file>
```

Before execution, the validator verifies the benchmark file and JSONL structure. A non-zero exit status or timeout is a `FAIL`.

For an inventory entry with Mock LLM enabled, the smoke subprocess receives `IANVS_LLM_MOCK=1` and a composed `PYTHONPATH`. Python loads the shared `sitecustomize.py`, which installs the adapters declared by the example fixture. The report labels this check `Runtime smoke test (mocked_llm)`.

A mocked run proves that the unchanged inference integration and benchmark flow execute with deterministic substitute responses. It does not prove model quality, model availability, provider availability, network access, GPU behavior, or benchmark accuracy.

Examples that require an external API key and do not have a supported Mock Runtime cannot run a meaningful credential-free smoke test. Classify that limitation explicitly; do not publish a real-provider passing status from a substituted response.

## CI coverage

The repository uses these validation levels:

| Tier | Selection | Checks |
| --- | --- | --- |
| T0 | Inventory examples with changed `.py`, `.yaml`, or `.yml` files below their example path | Static checks |
| T1 | Changed inventory entries | Active entries execute dynamic validation; inactive entries are reported as `SKIP` |
| T2 | Changes below `core/**`, `.github/workflows/validator/**`, or `.github/workflows/dynamic_code_cicd.yaml` | All inventory entries are selected; active entries execute dynamic validation and inactive entries are reported as `SKIP` |
| T3 | Scheduled run, push to `main`, or manual workflow dispatch | Dynamic validation for all active inventory targets and health snapshot publication |

The static workflow currently triggers for changed example Python or YAML files. The dynamic workflow event filter covers `examples/**`, `core/**`, `.github/workflows/validator/**`, and `.github/workflows/dynamic_code_cicd.yaml`. Although `inventory_loader.py` treats every `.github/workflows/` path as a dynamic run-all prefix when invoked, changes to other workflow files do not trigger the current dynamic workflow. Scheduled planning runs daily and uses a seven-day broad-validation cadence. Generated reports and status snapshots are the evidence for classification; a passing mocked check must retain its `mocked_llm` label.

For static and dynamic pull-request validation, the base target set is selected
from the base revision's inventory and the head target set from the head
inventory, using the same changed-file range. The sets may differ when a
benchmark unit is added, removed, or renamed. T2 base-health completeness is
measured against the complete base target set, not the head target count.

T0 does not select documentation-only changes and does not perform
Markdown-specific validation, deep parsing of runtime GPU declarations, or
broader static semantic analysis. T2/T3 status is based on dynamic evidence,
not inferred from a T0 pass.
