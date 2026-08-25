# KubeEdge Ianvs Example Classification CI Validation Framework

Automated Example Classification, Validation, and `llm_simple_qa` Restoration for Sustainable Ianvs Example Maintenance

---

## Background

Ianvs is the KubeEdge SIG AI distributed benchmark toolkit. It provides benchmark examples for edge AI, cloud-edge collaborative inference, federated learning, lifelong learning, LLM benchmarking, robotics, and other distributed AI scenarios.

As the number of Ianvs examples continues to grow, the project faces increasing usability and maintenance challenges. Historical examples may fail due to Python version changes, evolving dependencies, third-party library updates, outdated runtime configurations, missing datasets, hardcoded local paths, and documentation that no longer matches the actual implementation.

These problems create difficulties for both maintainers and users. Maintainers cannot easily determine which examples are currently healthy, which are broken, or which require external resources or special hardware. New developers and enterprise users may also fail to run examples from a clean environment because there is no clear classification of example status.

This proposal introduces a CI-based example classification and validation framework for Ianvs. The goal is to make example status visible, repeatable, and maintainable. The CI pipeline will act as a validation and reporting layer around Ianvs examples, helping maintainers classify example health and helping contributors understand whether their changes introduce regressions.

This proposal keeps the main focus on example classification and CI validation, but it also includes concrete restoration work for `examples/llm_simple_qa`. The example is selected as the initial repair target because it exposes a representative set of problems that the framework should detect, report, and verify after repair: hardcoded local paths, unclear dataset setup, invalid JSONL risks, local-only model paths, CUDA-only assumptions, incomplete dependency documentation, and metric edge cases.

The project does not aim to restore every broken example in Ianvs. Broad example repair should still be handled by separate proposals or follow-up tasks. The exception in this proposal is `examples/llm_simple_qa`: this project will repair it to become a portable, clean-environment reference example, and the CI framework will verify that restoration.

---

## Goals

The goals of this project are:

* Build an example inventory and classification matrix.
* Introduce automated CI validation for example status classification.
* Detect hardcoded paths, missing files, dependency issues, dataset issues, model-loading assumptions, hardware assumptions, and runtime failures.
* Support multi-version Python validation where practical.
* Provide local validation tools for contributors.
* Prevent pull requests from breaking already validated examples.
* Distinguish PR-introduced regressions from pre-existing or time-based failures.
* Reduce unnecessary CI runtime through tiered validation.
* Generate readable example health reports for maintainers.
* Use `examples/llm_simple_qa` as the initial validation target, repair target, and reference example for LLM-related example checks.
* Repair `examples/llm_simple_qa` so it becomes portable and reproducible from a clean Ianvs clone, then define CI-verifiable checks to prevent regression.

For maintainers, the project aims to reduce manual review burden and provide better visibility into example health.

For contributors, the project aims to provide clear local and CI feedback about whether their changes affect validated examples.

For developers and enterprise users, the project aims to make the status of Ianvs examples easier to understand before they attempt to run or reuse them.

---

## Problem Statement

Ianvs currently faces four major classes of example maintenance problems.

### 1. Unclear Example Health Status

Some examples may be runnable, while others may be partially runnable, broken, dependent on large datasets, or dependent on GPU or special hardware.

Without a systematic classification mechanism, maintainers and users cannot easily know:

* Which examples are currently validated
* Which examples are known to fail
* Which examples require external datasets
* Which examples require GPU or special hardware
* Which examples are affected by dependency drift
* Which examples should be excluded from normal PR-blocking checks
* Which examples have clean-environment validation evidence

### 2. Example Execution Failures

Some examples fail to execute due to stale paths, missing configuration files, outdated dependencies, or runtime incompatibilities.

Common failure patterns include:

* Broken YAML paths
* Hardcoded local paths
* Missing datasets
* Missing model resources
* Invalid or undocumented dataset formats
* Python version incompatibility
* Framework API changes
* Device assumptions such as CUDA-only execution
* Metric evaluation edge cases

`examples/llm_simple_qa` is a useful initial target because it contains many of these failure patterns in a compact example. The current known blockers include outdated README/YAML paths, references to local paths such as `/home/icyfeather/...`, unclear dataset preparation, invalid multi-line JSONL risks, local Qwen model paths, CUDA-only assumptions, incomplete dependency documentation, and possible metric failure when no valid prediction-answer pairs exist.

This project will detect and classify these failures. It will also repair `examples/llm_simple_qa` so that it becomes a portable reference example. Broad restoration of all examples remains outside the scope of this proposal.

### 3. Lack of Automated Validation

Currently, example failures are often discovered manually by users or contributors. Pull requests may unintentionally break previously validated examples because there is no systematic CI validation for example status.

For LLM examples, this problem is amplified by model loading, tokenizer availability, device selection, external datasets, and heavyweight dependencies. CI should therefore distinguish between lightweight static requirement checks, dependency installation checks, dataset preparation or availability checks, and optional runtime smoke tests.

### 4. Increasing Maintainer Burden

As the number of examples increases, manually checking and classifying each example becomes unrealistic. Maintainers need automated feedback to understand which examples are healthy, which examples are broken, and which PRs introduce regressions.

In addition, GitHub Actions workflows that can affect CI execution often require maintainer approval before they can run on pull requests from new contributors. Requiring a maintainer to repeatedly inspect and approve obviously safe pull request updates creates operational overhead and delays validation feedback.

---

## Proposal

This project proposes a non-intrusive CI validation framework for Ianvs examples.

The CI pipeline will not replace Ianvs' core execution logic. Instead, it will validate and classify Ianvs examples by checking configuration files, dependencies, datasets, documentation consistency, hardware assumptions, model-loading assumptions, and selected smoke-test executions.

The proposal contains three major parts:

1. Example inventory and classification
2. CI validation framework
3. Reporting and contributor feedback system

The CI framework shall detect examples affected by a pull request and execute the corresponding validation workflow only for those examples.

Lightweight static requirement checks will run across relevant examples on every pull request. More expensive dependency installation checks, dataset preparation or availability checks, and smoke tests will run only for changed examples or examples affected by shared code changes.

A broader validation suite will run on a scheduled basis to detect time-based failures such as dependency drift, dataset unavailability, model download failure, or CI environment changes.

A possible future extension is an automatic review bot for pull requests. If maintainers later decide the extra automation is worthwhile, the bot can inspect the diff against the `main` branch and auto-approve workflow execution only when CI-sensitive code such as workflow YAML files under `.github/workflows/` or validator automation under `.github/workflows/validator/` is unchanged. Pull requests that touch CI-sensitive paths would still remain pending for maintainer review and approval.

This proposal focuses on classification, validation, reporting, and one concrete restoration target: `examples/llm_simple_qa`. Fixing every broken example, replacing datasets for all examples, or rewriting all outdated documentation is out of scope and should be handled by separate restoration proposals or follow-up issues.

For `examples/llm_simple_qa`, this proposal adds a focused target: the validation framework should be able to confirm whether the example can run from a clean clone with portable paths, reproducible dataset setup, valid JSONL, configurable model loading, CUDA/MPS/CPU fallback, documented dependencies, and robust metric behavior.

LLM Smoke Tests may otherwise require a GPU, model download, external API credentials, network access, or substantial CI time. This proposal therefore adds an opt-in Mock LLM runtime: the Validator owns shared SDK adapters, each Example owns its responses, and `sitecustomize.py` injects them without changing Example inference code. This proposal implements both the Hugging Face pattern used by `examples/llm_simple_qa` and the OpenAI Chat Completions API pattern.

---

## Scope

### In Scope

The project will include:

* Example inventory and classification
* Static requirement validation scripts
* Dependency validation
* Dataset and JSONL validation support for selected examples
* Model path and hardware assumption checks for LLM examples
* Example smoke testing for selected examples
* GitHub Actions workflow
* Local validation commands
* Example health reporting
* Failure classification
* Tiered validation strategy for PR and scheduled workflows
* Documentation for validation rules and local validation usage
* Initial validation coverage for `examples/llm_simple_qa`
* Restoration of `examples/llm_simple_qa` portability and clean-environment execution
* An opt-in Smoke Test Mock Runtime with Example-owned responses, Hugging Face support for `llm_simple_qa`, and OpenAI Chat Completions support

### Out of Scope

The project will not:

* Restore or repair every broken example in the repository beyond `examples/llm_simple_qa`
* Rewrite all outdated example implementations
* Replace missing datasets or model resources for all examples
* Redesign the benchmark execution framework
* Replace Ianvs core architecture
* Replace KubeEdge or edge-cloud synergy components
* Rewrite all examples at once
* Run every example fully on every pull request
* Introduce core code changes unless repeated CI failures reveal a framework-level issue
* Guarantee that every classified example becomes runnable during this project
* Use predefined LLM responses as evidence that a real model, external provider, or model-quality evaluation has passed
* Treat missing `preprocess()` as an active blocker for `llm_simple_qa`, because PR #407 already addressed the relevant core-side `_preprocess()` behavior

The design principle is:

```text
Classify broadly, repair `llm_simple_qa` in this proposal, and repair other examples in separate proposals.
```

A secondary principle is:

```text
CI first, core changes only when necessary.
```

For `examples/llm_simple_qa`, the framework should classify failures and verify restoration targets. If restoration changes are implemented in a related PR, this proposal should reuse or merge them. Otherwise, this project will implement the missing `llm_simple_qa` restoration changes directly and validate them through CI.

---

## Target Users

### User Group A: Ianvs Maintainers

Maintainers need automated validation to understand example health and prevent validated examples from being broken by new changes.

Main needs:

* Classify example status
* Detect example regressions early
* Understand which examples are runnable, broken, skipped, resource-dependent, model-dependent, or hardware-dependent
* Review PRs with CI evidence
* Track example health over time
* Reduce manual debugging and classification effort
* Avoid blocking unrelated PRs because of pre-existing or time-based failures

### User Group B: Contributors

Contributors need a clear way to validate their changes before opening a pull request.

Main needs:

* Run local validation commands
* Understand why validation fails
* Reproduce CI failures locally
* Know whether a failure was introduced by their PR or already existed
* Avoid being responsible for unrelated historical failures
* Understand which examples are validated and which are classified as known failures

### User Group C: Developers and Enterprise Users

Developers and enterprise users need to know whether examples are likely to run before adopting them.

Main needs:

* Identify validated examples
* Understand the dataset, dependency, model, and hardware requirements
* Avoid spending time on examples already classified as broken or resource-dependent
* Use example status reports to select suitable examples
* Trust that validated examples are monitored by CI

---

## Design Details

### User Story Roadmap

The following roadmap connects the main user roles, the primary validation entry points, and the downstream decision flow across the several use cases. It provides a high-level view of how contributors, maintainers, and developers or enterprise users move through the validation and example health workflow.

![User Story Roadmap](images/User-Story-Roadmap.drawio.png)

### Relationship Between CI and Ianvs

The CI pipeline is a validation and classification layer around Ianvs examples.

```text
Ianvs = benchmark execution framework
CI = automated validation and classification mechanism for Ianvs examples
```

CI will call Ianvs commands, inspect example configuration files, verify dependencies, validate datasets where practical, check documentation consistency, and report whether examples remain runnable or require special classification.

The CI pipeline mainly interacts with:

* `examples/`
* `benchmarkingjob.yaml`
* `testenv.yaml`
* `testalgorithms/`
* example README files
* dependency files
* dataset path configuration
* model configuration
* runtime execution commands
* evaluation metric files
* related documents

The first version should avoid core Ianvs changes. Core changes should only be considered when multiple examples fail due to the same framework-level behavior, and such changes should be discussed separately.

### Use Cases

The proposal covers four top-level validation use cases. UC-01 is the parent pull request workflow and contains specialized validation and regression-classification sub-scenarios.

#### UC-01: Pull Request Validation and Regression Handling

A contributor submits a pull request and needs the validation system to validate the change, distinguish a regression introduced by the pull request from a failure that already exists in the base branch, and determine whether the detected result should block merge.

The goal is to make pull request feedback fair and actionable by blocking only PR-introduced regressions while still surfacing pre-existing failures to maintainers and contributors.

##### UC-01.1: Document-Only Pull Request Validation

A contributor submits a pull request that changes documentation only, such as example README files, usage guides, or related proposal documents.

In this use case, the CI workflow runs static documentation validation instead of example execution-oriented validation. The workflow generates a documentation validation report, and maintainers review the result to decide whether the pull request can be merged or whether documentation fixes are still needed. If validation fails, the failure should be classified as a documentation issue rather than as an example runtime regression.

The goal is to avoid unnecessary runtime validation for document-only changes while still preserving documentation quality and consistency.

![UC-01.1 Document-Only PR Validation Use Case](images/use-case/Use-Case-Diagram-Document-Change-PR-Validation.drawio.png)

##### UC-01.2: Single Example Change Pull Request Validation

A contributor submits a pull request that changes one example without modifying the shared Ianvs core or CI-sensitive framework code.

In this use case, the workflow detects the changed example, runs affected-example validation, applies the appropriate validation tier, and generates a pull request health report. If validation fails, the workflow should classify the failure so maintainers can distinguish a newly introduced regression from a pre-existing or unrelated problem. Maintainers then review the validation result and decide whether to merge the pull request or request fixes.

The goal is to keep pull request validation targeted, efficient, and actionable for ordinary example-level contributions.

![UC-01.2 Single Example Change PR Validation Use Case](images/use-case/Use-Case-Diagram-Example-Change-PR-Validation.drawio.png)

##### UC-01.3: Core Code Change Pull Request Validation

A contributor submits a pull request that changes shared Ianvs core code, common execution logic, or other paths that may affect multiple examples at once.

In this use case, the CI workflow runs a broader regression validation scope than example-only changes. It generates a pull request health report for maintainer review, and any validation failure should be classified so the project can distinguish a true cross-example regression from an unrelated environmental issue. Maintainers use the result to decide whether the pull request is safe to merge or requires additional fixes.

The goal is to protect validated examples from framework-level regressions when shared code changes have a wider blast radius.

![UC-01.3 Core Code Change PR Validation Use Case](images/use-case/Use-Case-Diagram-Core-Code-Change-PR-Validation.drawio.png)

##### UC-01.4: Regression Classification and Blocking Decision

In this use case, pull request validation is triggered automatically after the contributor submits the pull request. The workflow compares the base result and the pull request result, then generates a regression report for the contributor to read. If the comparison shows that the pull request introduced a new regression, the workflow should block the pull request until the contributor fixes it. If the failure already exists in the base branch, the workflow should report the pre-existing failure without blocking the pull request for that specific issue.

The regression-classification and blocking logic should cover:

* PR-introduced regression
* Existing base failure
* Unrelated failure
* Blocking versus non-blocking decision

![UC-01.4 Regression Classification and Blocking Decision Use Case](images/use-case/Use-Case-Diagram-Pull-Request-Regression-Handling.drawio.png)

#### UC-02: Local Validation Before Pull Request Submission

A contributor wants to validate changes locally before opening or updating a pull request.

In this use case, the contributor runs the local validation workflow with `nektos/act` or an equivalent local entry point. The workflow syncs with the upstream baseline, prepares a temporary validation branch, runs validation locally, and generates a local report. The validation flow should always include static validation and may extend to smoke testing when the changed example or validation tier requires runtime execution. If the local run fails, the contributor should fix the failure before opening or updating the pull request.

The goal is to give contributors fast feedback before CI review, reduce avoidable pull request failures, and make CI results easier to reproduce locally.

![UC-02 Local Validation Use Case](images/use-case/Use-Case-Diagram-Local-Validation.drawio.png)

#### UC-03: Scheduled Validation and Time-Based Failure Triage

A maintainer wants the project to periodically re-validate examples even when no pull request is open, so the team can detect dependency drift, dataset availability problems, model download failures, and other time-based breakages.

The goal is to give maintainers continuous visibility into example health, surface long-term ecosystem drift early, and provide a structured response path for scheduled validation failures.

![UC-03 Scheduled Validation Use Case](images/use-case/Use-Case-Diagram-Scheduled-Validation.drawio.png)

##### UC-03.1: Scheduled Full Validation

In this use case, a scheduled CI workflow runs the broader validation suite and generates a health report for maintainer review.

##### UC-03.2: Time-Based Failure Classification

In this use case, the scheduled workflow detects time-based failures, classifies likely drift causes, and distinguishes scheduled drift detection from pull request regression detection so contributors are not blamed for failures introduced by the external environment over time.

##### UC-03.3: Maintainer Triage for Drift Failures

When a scheduled run identifies a failure, maintainers triage the result and choose an appropriate follow-up action, such as marking the example as a known failure, creating a follow-up issue, or quarantining a broken example until it is repaired.

#### UC-04: Example Status Management and Classification Review

A maintainer wants to review the current status of examples and update their classification based on validation evidence, environment requirements, and reported failures.

The goal is to make example health classification explicit, keep maintainer decisions consistent, and ensure the project records whether a failure is blocking, expected, or caused by special runtime prerequisites.

![UC-04 Example Status Management Use Case](images/use-case/Use-Case-Diagram-Example-Status-Management.drawio.png)

##### UC-04.1: Review Example Health Report

In this use case, the maintainer reviews the example report and views the current example status.

##### UC-04.2: Update Example Classification

Based on the validation result, the maintainer may update the example classification to reflect operational constraints such as GPU requirements, external dataset requirements, or model download requirements.

##### UC-04.3: Decide Blocking / Non-Blocking Status

In this use case, the maintainer decides whether a failure should block a pull request.

##### UC-04.4: Create Follow-up Issue

When the reported status reveals a follow-up maintenance task, the maintainer may create a follow-up issue to track restoration or cleanup work.

### Automatic Workflow Approval Bot

As a future extension, the project can add a lightweight GitHub-integrated review bot that reduces the need for maintainers to manually approve workflow execution on pull requests from new contributors.

The bot should:

* Trigger when a pull request is opened, synchronized, or reopened.
* Compare the pull request diff against the current `main` branch.
* Detect whether the pull request changes CI-sensitive paths, especially workflow YAML files under `.github/workflows/` and validator automation under `.github/workflows/validator/`.
* Automatically approve workflow execution when CI-sensitive paths are not changed.
* Leave the pull request for maintainer review when CI-sensitive paths are changed.
* Re-run the same decision on every pull request update so approval reflects the latest diff.

This bot is not intended to replace code review. Its purpose is only to automate workflow approval gating for low-risk pull requests so contributors can receive CI feedback faster while maintainers retain control over workflow and tooling changes.

The interaction is:

```mermaid
sequenceDiagram
    participant Contributor as new Contributor
    participant GitHub
    participant Bot

    Contributor->>+GitHub: Pull Request
    GitHub->>+Bot: [Webhook] new Pull Request
    Bot->>-GitHub: [API] Approve workflow
    GitHub->>GitHub: Run workflow
    GitHub-->>-Contributor: Result
```

In this design, workflow approval is based on path-level risk classification:

* Safe for automatic approval: pull requests that do not modify workflow YAML files under `.github/workflows/` or validator automation under `.github/workflows/validator/`
* Requires maintainer review: pull requests that modify workflow YAML files under `.github/workflows/`, validator automation under `.github/workflows/validator/`, or other paths later designated as CI-sensitive

This keeps the approval rule simple, auditable, and aligned with the goal of protecting CI execution logic while removing repetitive maintainer work for ordinary example or documentation changes.

---

## Architecture and Modules

The proposed framework adds a validation layer around existing Ianvs examples.

### Software Architecture Overview

The following architecture diagram shows the main system layers of the proposed framework. At a high level, CI/CD event triggers invoke the validation modules under `.github/workflows/validator/`, shared storage provides the inventory and validation state used across the system, and the existing Ianvs node-side managers remain the execution environment whose behavior is observed and classified by the validation pipeline. Within the CI/CD layer, the key proposal-specific responsibilities are static validation, dynamic affected-example validation, regression detection, local validation support, and report generation.

![Software Architecture](images/Software-Architecture.png)

Although the software architecture presents GitHub Actions as the CI/CD execution layer, the repository does not introduce a separate `.github/CICD/` directory because GitHub Actions only discovers workflow definitions under `.github/workflows/` ([document](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/approve-runs-from-forks)). Each validation concern therefore remains represented by a workflow file under `.github/workflows/`, including renamed CI/CD workflow definitions such as `static_code_pylint_cicd.yaml`, `static_code_requirement_cicd.yaml`, `third_party_codeql_analysis_cicd.yaml`, `third_party_fossa_cicd.yaml`, and `dynamic_code_cicd.yaml`. The static example-requirement checks are separated into `static_code_requirement_cicd.yaml`, while `dynamic_code_cicd.yaml` is reserved for validation that prepares an execution environment, installs dependencies, or runs example smoke tests. The General Validators are third-party integrations, so they require only their corresponding workflow files and do not introduce additional implementation files in the repository. In contrast, the Example Validator contains project-specific reusable logic, which is colocated under `.github/workflows/validator/` and invoked by the static requirement and dynamic validation workflows.

```diff
Ianvs Repository
  ├── examples/
+ │   ├── README.md (Show validation time and status matrix)
  │   ├── llm_simple_qa/
+ │   │   ├── scripts/
+ │   │   │   ├── 01_install_requirements.sh
+ │   │   │   ├── 02_prepare_dataset.py
+ │   │   │   ├── ...
+ │   │   │   └── mock_runtime/
+ │   │   │       └── ianvs_mock_fixture.py
  │   ├── example A/
  │   ├── example B/
  │   └── ...
  │
  ├── docs/
+ │   └── example_validator/
+ │       ├── validation_rules.md
+ │       ├── classification_policy.md
+ │       ├── local_validation.md
+ │       └── status_directions.md
  │
  └── .github/
      └── workflows/
-         ├── main-doc.yaml                           (Delete)
M         ├── static_code_pylint_cicd.yaml            (Rename)
M         ├── third_party_codeql_analysis_cicd.ymal   (Rename)
M         ├── third_party_fossa_cicd.yaml             (Rename)
+         ├── dynamic_code_cicd.yaml
+         ├── static_code_requirement_cicd.yaml
+         └── validator/
+             ├── validation_runner.py
+             ├── static_validator.py
+             ├── dependency_validator.py
+             ├── smoke_test_validator.py
+             ├── services/
+             │   ├── validation_branch_manager.py
+             │   ├── inventory_loader.py
+             │   ├── regression_detector.py
+             │   └── report_generator.py
+             ├── scripts/
+             │   └── mock_runtime/
+             │       ├── sitecustomize.py
+             │       └── adapters/
+             │           ├── __init__.py
+             │           ├── huggingface_adapter.py
+             │           └── openai_adapter.py
+             └── data/
+                 └── example_inventory.yaml
```

> [!WARNING]
> Renaming workflow files marked as `(Rename)` will hide the previous GitHub Actions run history for those workflows.

> [!NOTE]
> Delete `main-doc.yaml` because it does not perform any meaningful CI or validation work in practice, so the proposal removes the inactive workflow instead of carrying it forward.

The responsibilities of the proposed files are:

| Path | Responsibility |
|---|---|
| `examples/` | Stores Ianvs example projects, including their runnable configurations, documentation, dependency references, dataset references, and algorithm-related files. These directories are the validation targets of the framework. |
| `examples/README.md` | Serves as the maintainer-facing summary of current example health. It should keep the latest T2/T3 validation time and the example status matrix, link or point to the underlying CI evidence when needed, and provide a stable place to track whether an example is validated, degraded, quarantined, external-resource-dependent, or awaiting follow-up repair work. Detailed status explanations should live in `docs/example_validator/status_directions.md`, and `examples/README.md` should link to that file. |
| `docs/example_validator/status_directions.md` | Documents the example status model, badge legend, broken-status subtypes, and interpretation notes. This keeps detailed status explanations out of `examples/README.md`, where only the latest validation time and status matrix should remain. A later documentation update should make `examples/README.md` link to this file. |
| `.github/workflows/validator/validation_runner.py` | Serves as the main entry point for local and CI validation. It should parse CLI arguments, load the inventory, select validation stages, invoke the validation modules that live directly under `validator/`, and coordinate report generation. |
| `.github/workflows/validator/static_validator.py` | Performs lightweight static checks without executing examples. It should detect problems such as missing files, invalid YAML, broken relative paths, hardcoded local paths, outdated repository layout references, README and configuration mismatches, local-only model paths, and CUDA-only assumptions. |
| `.github/workflows/validator/dependency_validator.py` | Validates `requirements_file` syntax, Python-version compatibility, dependency drift, and installation in a clean environment. It does not prepare datasets or run other environment-preparation scripts. |
| `.github/workflows/validator/smoke_test_validator.py` | Runs lightweight execution validation for selected examples. For Mock LLM runs, it sets `IANVS_LLM_MOCK=1`, composes `PYTHONPATH` from inventory metadata, and starts each Example in a separate subprocess. |
| `.github/workflows/validator/scripts/mock_runtime/sitecustomize.py` | Loads the current Example fixture and calls `install(responses)` on each declared adapter when Mock LLM mode is enabled. |
| `.github/workflows/validator/scripts/mock_runtime/adapters/` | Contains the implemented `huggingface_adapter.py` and `openai_adapter.py` runtime adapters. Adapter filenames avoid shadowing installed SDK packages. |
| `examples/llm_simple_qa/scripts/mock_runtime/ianvs_mock_fixture.py` | Stores the Example-owned adapter selection and semantic responses used only by Mock Smoke Tests. |
| `.github/workflows/validator/services/validation_branch_manager.py` | Wraps local validation with the branch preparation and cleanup steps shown in the local validation flowchart. It should run around `validation_runner.py`, checking whether the `upstream` remote exists, adding it when missing, fetching `upstream/main`, finding the merge-base against the contributor's local `HEAD`, detecting changed files, creating a temporary validation branch, rebasing that branch onto `upstream/main`, and deleting the temporary branch after validation completes. |
| `.github/workflows/validator/services/inventory_loader.py` | Loads and manages the example inventory. It should provide structured metadata access, helper logic for selecting changed or affected examples, and shared inventory operations used by the validation pipeline. |
| `.github/workflows/validator/services/regression_detector.py` | Compares validation failures from the pull request result against the baseline result from the `main` branch. It should identify which failures are newly introduced by the pull request, which failures already exist on `main`, and which differences should be classified as non-blocking baseline debt rather than PR regressions. |
| `.github/workflows/validator/services/report_generator.py` | Converts validation results into human-readable CI summaries and example health reports, including failure classifications, reproduction commands, and suggested next actions for contributors and maintainers. |
| `.github/workflows/validator/data/example_inventory.yaml` | Stores the example inventory and classification metadata, including each example's paths, `requirements_file`, ordered `prepare_env.steps`, validation level, dataset requirements, model requirements, hardware requirements, current status, expected dataset structure, and whether the dataset is external when preparation is unavailable. Its optional `mock_runtime` metadata provides only the shared and Example fixture paths used to compose `PYTHONPATH`; adapter selection and responses remain Example-owned. |
| `docs/example_validator/validation_rules.md` | Documents the validation rules implemented by the framework, including what each validator checks, why the rule exists, and how maintainers should interpret its result. |
| `docs/example_validator/classification_policy.md` | Defines the example status model and failure classification policy, including which failure types block pull requests and which should be treated as known, pre-existing, or time-based failures. |
| `docs/example_validator/local_validation.md` | Documents how contributors run validation locally, including example commands, expected usage patterns, local troubleshooting, and optional workflow-level local verification guidance. |
| `.github/workflows/static_code_requirement_cicd.yaml` | Defines the GitHub Actions workflow for Tier 0 static validation. It should run non-execution checks such as required-file presence, YAML and README consistency, dependency-file declarations, dataset and model path declarations, hardcoded local path detection, and hardware-assumption detection by invoking the reusable static validation logic under `.github/workflows/validator/`. |
| `.github/workflows/dynamic_code_cicd.yaml` | Defines the GitHub Actions workflow for execution-oriented example validation tiers. It should run only the dynamic portions of validation, such as dependency installation validation, the separate environment-preparation stage, smoke tests, regression comparison, result collection, and CI summaries or report artifacts. |

For local contributor validation, `validation_branch_manager.py` should act as a wrapper around `validation_runner.py`. Before validation starts, it should ensure the `upstream` remote is available, synchronize with `upstream/main`, compute the change set against the merge-base, and create the temporary rebased validation branch. After validation finishes, it should handle cleanup, including deleting the temporary validation branch.

For pull request validation, `validation_runner.py` should invoke `regression_detector.py` after the selected validation modules complete so the framework can compare the PR result against the current `main` branch baseline before deciding whether a failure should block merge.

---

## Module Details

### 1. Example Inventory Module

Purpose:

* Track all Ianvs examples and their validation status.

Responsibilities:

* List examples
* Record benchmark configuration path
* Record dataset requirements
* Record dependency requirements
* Record hardware requirements
* Record model requirements
* Record validation level
* Classify example status
* Record whether failures are known, newly introduced, or time-based
* Record clean-environment validation evidence when available

Output:

* Example Classification Matrix

The status matrix should follow a single lifecycle-oriented state model so maintainers can distinguish current state from machine-assigned failure reasons.

![Example Status Transition Standard](images/example-status-STD.png)

Example status categories:

* Runnable
* Broken
* Not validated yet
* Requires external dataset or model download
* Requires GPU or special hardware
* Quarantined
* Known issue

Auto-classified broken subtypes:

* Dataset or resource unavailable
* Dependency drift
* Documentation issue

Example inventory metadata may include:

```yaml
examples:
  - name: simple_qa_singletask_learning
    example: llm_simple_qa
    status: active
    path: examples/llm_simple_qa
    readme_file: examples/llm_simple_qa/README.md
    benchmark_file: examples/llm_simple_qa/benchmarkingjob.yaml
    requirements_file: examples/llm_simple_qa/requirements.txt

    mock_runtime:
      enabled: true
      shared_pythonpath:
        - .github/workflows/validator/scripts/mock_runtime
      example_pythonpath:
        - examples/llm_simple_qa/scripts/mock_runtime

    prepare_env:
      working_directory: examples/llm_simple_qa
      steps:
        - name: install_requirements
          type: dependency
          script: scripts/01_install_requirements.sh
          args:
            - requirements.txt
          timeout: 600

        - name: prepare_dataset
          type: dataset
          script: scripts/02_prepare_dataset.py
          args:
            - --output-dir
            - ../../dataset/llm_simple_qa
            - --overwrite
          timeout: 300
```

The inventory provides the shared and Example fixture paths; `ianvs_mock_fixture.py` declares adapters and responses.

`prepare_env.steps` is an ordered environment-preparation contract. Each step supports `name`, `type`, `script`, `args`, and `timeout`. `args` must be an array of strings and must be passed directly to the subprocess argument vector; the runner must not use `shell=True`. The separate environment-preparation stage runs these steps in order from `working_directory` and reports the failing step, if any. The Mock Runtime is not a `prepare_env` step; it is injected only into the Smoke Test subprocess environment.

---

### 2. Static Validation Module

Purpose:

* Detect common configuration and path problems before runtime execution.

Checks:

* Hardcoded absolute paths
* Missing YAML files
* Missing README files
* README contains setup steps, execution commands, and troubleshooting information when such checks can be performed statically
* Broken dataset paths
* Broken algorithm paths
* Broken test environment paths
* Invalid YAML syntax
* Outdated repository layout references
* README commands that do not match existing files
* Local model paths
* CUDA-only hardcoding where CPU fallback should be supported
* Missing model override documentation
* Missing CPU fallback
* Missing tokenizer/model dependency documentation
* Metric edge cases caused by empty predictions or empty answer pairs when the risk can be detected statically
* Dataset format mismatch between README, YAML, and runtime code
* README contains dependency installation instructions
* README contains dataset preparation instructions
* README describes the inventory-defined `prepare_env` flow when the example supports automated environment preparation
* README contains JSONL format when applicable
* README contains model configuration instructions when applicable
* README paths match YAML paths
* Dependency documentation matches actual requirements

Example checks:

```text
/home/username/...
/home/icyfeather/...
/home/.*/models
examples/old_path/...
examples/llm/singletask_learning_bench/simple_qa
missing benchmarkingjob.yaml
missing testenv.yaml
device = "cuda"
```

Output:

* Static validation report
* Classification update for affected examples

Example static validation report:

```md
# Static Validation Report

## Example

examples/llm_simple_qa

### Validation Result

| Check | Result |
|---|---|
| benchmarkingjob.yaml exists | PASS |
| testenv.yaml exists | PASS |
| README exists | PASS |
| Hardcoded path check | FAIL |
| Dataset path validation | FAIL |
| Local model path check | FAIL |
| CUDA-only device check | FAIL |

### Failure Details

#### Hardcoded Path

File:

examples/llm_simple_qa/benchmarkingjob.yaml

Detected:

`/home/user/...`
```

Static validation should be lightweight enough to run across relevant examples on every pull request.

For `examples/llm_simple_qa`, static validation should also confirm:

* The README explains the example overview, setup steps, dependency installation, dataset preparation, JSONL format, model configuration, run command, expected output, and troubleshooting.
* The documented dataset layout matches the inventory and the README describes any applicable `prepare_env` dataset step.
* Model loading uses a portable model ID or a documented override mechanism instead of local-only paths.
* Device selection supports CUDA, MPS, and CPU fallback rather than assuming CUDA-only execution.
* Metric handling avoids crashes when no valid prediction-answer pairs exist, for example by returning `0.0` and logging a warning instead of triggering `ZeroDivisionError`.
* Related PRs from Issue/PR #452 that already solve part of the restoration work are reflected consistently in the updated documentation and example configuration.

---

### 3. Dependency Validation Module

Purpose:

* Verify whether declared dependencies are syntactically valid, installable, and compatible; environment preparation is handled separately.

Checks:

* `requirements_file` availability and requirements syntax
* Python version compatibility
* Dependency drift and dependency conflicts
* Clean-environment installation validation
* Missing runtime packages and example-specific dependency documentation

Initial Python matrix:

* Python 3.8
* Python 3.9
* Python 3.10

For `examples/llm_simple_qa`, the validation framework should recognize the inventory-declared dependency file:

```text
examples/llm_simple_qa/requirements.txt
```

Planned content:

```text
# Machine Learning Libraries
transformers >= 4.45.0
torch >= 2.0.0
accelerate >= 1.0.0
```

The dependency validator must not execute `prepare_env` steps. It is limited to requirements syntax, Python compatibility, dependency drift, and installation validation; the separate environment-preparation stage executes the inventory-defined steps after dependency validation.

Output:

* Dependency compatibility report
* Dependency-related classification result

Dependency validation should run mainly for changed examples or examples affected by shared dependency changes.

---

### 4. Environment Preparation Module

Purpose:

* Prepare an example's runtime environment through the ordered `prepare_env.steps` declared in the inventory.

Execution rules:

* Run each step sequentially from `prepare_env.working_directory`.
* Require every step to provide `name`, `type`, `script`, `args`, and `timeout`.
* Require `args` to be an array of strings; pass the script and arguments directly to `subprocess` without `shell=True`.
* Apply each step's timeout and stop the stage on failure, reporting the step name and type.
* Keep dependency validation separate: a `type: dependency` preparation step may perform example-specific setup, but it does not expand the responsibilities of `dependency_validator.py`.

This stage may install example prerequisites, prepare datasets, or perform other documented setup required before smoke testing. It should run only when the selected validation tier requires environment preparation.

---

### 5. Dataset and JSONL Validation Module

Purpose:

* Verify whether dataset files are present, documented, and structurally valid when lightweight validation is practical.

Checks:

* Dataset path exists or is declared in `example_inventory.yaml`
* Dataset path matches README and YAML references
* Applicable `prepare_env` dataset steps are declared and their documented output location matches the inventory
* `example_inventory.yaml` declares the expected dataset directory structure
* If automated dataset setup is unavailable, the example inventory marks the dataset as `external: true`
* JSONL files are not empty
* Each JSONL line is a complete JSON object
* Required fields are present
* The relevant `prepare_env` dataset step produces or documents the expected dataset layout when data is not committed

For `examples/llm_simple_qa`, the expected dataset layout may be:

```text
dataset/llm_simple_qa/
├── train_data/
│   └── data.jsonl
└── test_data/
    └── data.jsonl
```

Each JSONL line should be one complete JSON object, for example:

```json
{"question": "If Xiao Ming has 5 apples and gives 3 to Xiao Hua, how many apples does Xiao Ming have left?\nA. 2\nB. 3\nC. 4\nD. 5", "answer": "A"}
```

Example validation commands:

```bash
cd examples/llm_simple_qa
scripts/01_install_requirements.sh requirements.txt
python scripts/02_prepare_dataset.py --output-dir ../../dataset/llm_simple_qa --overwrite
python scripts/validate_jsonl.py ../../dataset/llm_simple_qa/train_data/data.jsonl
python scripts/validate_jsonl.py ../../dataset/llm_simple_qa/test_data/data.jsonl
```

Output:

* Dataset validation report
* Dataset/resource classification result

---

### 5. Smoke Test Module

Purpose:

* Run selected examples in CI to verify that they can execute.

Smoke tests should be lightweight and should not require large datasets or long GPU execution.

Validation target:

```bash
ianvs -f examples/<example_name>/benchmarkingjob.yaml
```

For `examples/llm_simple_qa`, preferred validation command:

```bash
ianvs -f examples/llm_simple_qa/benchmarkingjob.yaml
```

Alternative command if required by current Ianvs documentation:

```bash
python3 benchmarking.py -f examples/llm_simple_qa/benchmarkingjob.yaml
```

For examples that require large datasets or large model downloads, CI should mark the example accordingly or use an existing lightweight validation mode if already available. Creating new datasets or repairing dataset pipelines for all examples is outside the scope of this proposal.

For `examples/llm_simple_qa`, the Validator may run the same command in Mock LLM mode by setting only the following subprocess environment:

```bash
IANVS_LLM_MOCK=1 \
PYTHONPATH=.github/workflows/validator/scripts/mock_runtime:examples/llm_simple_qa/scripts/mock_runtime \
ianvs -f examples/llm_simple_qa/benchmarkingjob.yaml
```

The two `PYTHONPATH` entries provide the shared Runtime and current Example fixture. Each Example runs in a separate subprocess, and no third environment variable is required.

The Smoke Test flow is:

```mermaid
flowchart TD
    A[Validator] --> B[Compose PYTHONPATH and set IANVS_LLM_MOCK]
    B --> C[Start Example subprocess]
    C --> D[Python loads sitecustomize]
    D --> E[Load Example fixture and declared adapters]
    E --> F[Patch supported SDK APIs]
    F --> G[Run unchanged Example flow]
    G --> H[Return Example-owned Mock Response]
```

Output:

* Runtime validation report
* Example pass/fail status
* Updated example classification

Smoke tests should run for:

* Changed examples
* Examples affected by shared code changes
* Representative examples for core framework changes
* Broader example sets in scheduled validation

---

### 6. Mock LLM Runtime for Smoke Tests

Purpose:

* Make the Python `llm_simple_qa` Smoke Test deterministic, offline, and low cost without downloading or executing a real Hugging Face model.

#### Runtime Injection

The Validator sets `IANVS_LLM_MOCK=1` and composes `PYTHONPATH` from `.github/workflows/validator/scripts/mock_runtime` and the current Example's `scripts/mock_runtime`. Python then imports the shared `sitecustomize.py`, which loads the Example-owned fixture and each declared adapter. This remains outside `prepare_env` and does not change Example inference code.

Adapters expose `install(responses)`. `sitecustomize.py` invokes it for each entry in `ADAPTERS`. The Hugging Face adapter patches the two `from_pretrained()` calls and implements only the Model, Tokenizer, batch, `generate()`, and `batch_decode()` behavior used by `llm_simple_qa`. The OpenAI adapter patches `OpenAI(...)` and `AsyncOpenAI(...)` clients and implements `client.chat.completions.create(...)` for synchronous, asynchronous, and streaming calls without credentials or network access.

The adapter owns SDK compatibility; the Example fixture owns semantic responses:

```python
ADAPTERS = ["huggingface"]
RESPONSES = {
    "huggingface": {"default": "A"},
}
```

Examples may list `huggingface`, `openai`, or both; each adapter constructs its SDK-specific response shape from its grouped fixture data.

Requirements:

* Mock LLM mode must be explicitly enabled with `IANVS_LLM_MOCK=1`; it must be disabled by default.
* The Validator must inject the Mock Runtime only when starting a Smoke Test and must not register it as a `prepare_env` step.
* Example inference code must remain unchanged, and the version-controlled fixture must own adapter selection and semantic responses.
* Adapters must implement `install(responses)` and preserve the response shape expected by existing Example code.
* The OpenAI adapter must support `OpenAI(...)`, `AsyncOpenAI(...)`, and `client.chat.completions.create(...)`, including non-streaming and streaming response shapes.
* OpenAI Mock Runtime tests must run without an API key and must fail if they attempt external network access.
* Validation reports must identify runs that used substituted responses as `mocked_llm`; those runs must not be classified as real Hugging Face model, GPU, external-provider, or model-quality validation.

#### `examples/llm_simple_qa` Repair Example

`llm_simple_qa` keeps its existing Hugging Face `from_pretrained()` calls and `_infer()` flow. Its fixture selects Hugging Face and provides the fixed answer, allowing offline execution without a model download or GPU. Real model validation remains a separate resource-dependent tier.

#### Limitations

* The first phase applies only to Python Examples.
* Hugging Face support covers only the API pattern currently used by `llm_simple_qa`; other Transformers patterns require adapter extensions.
* OpenAI support covers the Chat Completions API pattern; other OpenAI SDK resources and endpoints are outside this proposal.
* An Example may declare multiple adapters; different Examples run in separate Smoke Test subprocesses.
* Runtime patches affect only the Python process that loaded `sitecustomize.py` and Python child processes that inherit the same environment; they do not modify installed packages or system-wide behavior.
* Mock LLM mode verifies execution, not model quality, real GPU behavior, resource availability, or benchmark accuracy.

---

### 7. Report Generator

Purpose:

* Provide maintainers and contributors with readable validation and classification feedback.

Report contents:

* Passed checks
* Failed checks
* Failure reason
* Affected example
* Related file path
* Reproduction command
* Failure classification
* Whether the failure blocks the PR
* Suggested next action, such as creating a follow-up issue or marking the example as quarantined
* Clean-environment validation evidence when available

Output:

* CI summary
* Example health report
* Markdown report artifact

Example report:

```markdown
## Example Validation Report

Example: examples/llm_simple_qa
Status: Failed
Classification: Dataset/resource unavailable
PR Blocking: No, unless this PR modified the affected dataset path or example configuration.

Failed Check:
- Runtime smoke test

Reason:
- benchmarkingjob.yaml references a dataset path that is not available in CI.

Suggested Next Action:
- Record the example as dataset-dependent in the example inventory.
- Create a follow-up restoration issue if maintainers decide the example should be repaired.

Reproduction:
python .github/workflows/validator/validation_runner.py --example examples/llm_simple_qa --smoke
```

---

## Tiered Validation Strategy

| Level | Name | Mode | Purpose |
| --- | --- | --- | --- |
| **Tier 0** | **Static Validation** | **Static** | Detect file-level issues without executing examples. |
| **Tier 1** | **Targeted Example Validation** | **Dynamic** | Execute validation for only the examples affected by the current change. |
| **Tier 2** | **Full Example Validation** | **Dynamic** | Execute smoke validation for all examples when core code or validation logic changes. |
| **Tier 3** | **Scheduled Validation** | **Dynamic** | Periodically run smoke validation for all examples to detect time-dependent failures. |

The CI framework should not execute every Ianvs example on every pull request. Instead, it will use a tiered validation strategy.

### Tier 0 — Static Validation

Runs on:

* Every pull request

Coverage:

* Changed examples
* Lightweight repository-wide checks when practical

Purpose:

* Detect low-cost structural problems early.

Checks:

* YAML syntax
* Missing files
* Hardcoded paths
* Broken local references
* README path consistency
* Local model path references
* CUDA-only hardcoding in examples expected to support CPU fallback

PR impact:

* Blocks PR only if a new static validation failure is introduced by the PR.

---

### Tier 1 — Targeted Example Validation

Runs on:

* Every pull request that modifies files under `examples/<example_name>/`

Coverage:

* Changed examples only

Purpose:

* Validate examples directly modified by the PR.

Checks:

* Dependency installation
* Dataset preparation or availability checks when practical
* Lightweight smoke test

Exception:

* If a changed example is too large or too time-consuming to run within the
CI runner limits, Tier 1 may skip the full runtime smoke test for that
example.

Example:

```text
If a PR modifies examples/llm_simple_qa/**, 
CI runs dependency checks, JSONL checks, LLM-specific static checks, and smoke tests for llm_simple_qa.
```

---

### Tier 2 — Full Example Validation

Runs on:

* Pull requests that modify core code, shared configuration, or validation logic

Coverage:

* All examples in the validation target set

Checks:

* Dependency installation
* Example-specific static checks
* Dataset-format validation when lightweight
* Lightweight smoke test
* README consistency

Shared changes may include:

* Ianvs core modules
* Common dataset loader
* Common evaluator
* Common algorithm interface
* Shared dependency files
* GitHub Actions workflow
* Validation framework scripts
* Shared example templates

Example:

```text
If a PR modifies a common evaluator, 
CI runs smoke validation for the full validation target set instead of limiting execution to only the changed example.
```

---

### Tier 3 — Scheduled Validation

Runs on:

* Daily or weekly schedule

Coverage:

* All examples in the validation inventory

Checks:

* Dependency installation when required by the example
* Dataset preparation or availability checks when practical
* Lightweight smoke test across the full inventory

Purpose:

* Detect time-based failures.

Examples of time-based failures:

* Dependency drift
* Dataset URL expiration
* Model download failure
* CI runner image changes
* Python version compatibility changes

PR impact:

* Does not automatically block unrelated PRs.
* Updates the example health report.
* Creates or references maintenance issues if configured by maintainers.

When a pull request runs Tier 2 validation, the Tier 3 schedule for the same target set should be reset from that Tier 2 run. This rescheduling occurs regardless of whether the pull request is later merged into `main`; merge status must not be a prerequisite. This avoids immediately re-running a broad scheduled validation for a target set that was just exercised by Tier 2 while retaining Tier 3 as the periodic time-dependent validation signal.

![Tier 3 Validation Reset After Tier 2](images/Tier3-Validation-Reset-After-Tier2.png)

The timeline shows the reset behavior through three main time points:

* **Previous Tier 3:** The last scheduled Tier 3 run has completed and becomes the current validation baseline for `main`.
* **Tier 2 PR Run:** One day after the previous Tier 3 run, an open pull request runs Tier 2 validation for the same target set. The run resets the Tier 3 timer even if the pull request is never merged.
* **Rescheduled Next Tier 3:** The next Tier 3 timer restarts from the Tier 2 run. With a one-week Tier 3 cadence, the original next Tier 3 time is skipped and the actual next Tier 3 run is pushed back by one day.

This reset rule applies when Tier 2 runs before the originally scheduled Tier 3 time. A later merge, rejection, or closure of the pull request does not restore the original Tier 3 schedule or otherwise change the already rescheduled Tier 3 run.

---

### Validation-Level Detection Workflow

The validation framework should make it easy to see which kinds of repository changes trigger which validation tiers. Pull request validation should begin by scanning changed files and mapping the detected change scope to the appropriate tier combination. Documentation-only changes should run Tier 0 static validation through `static_code_requirement_cicd.yaml` only. Changes within a single example should run Tier 0 plus Tier 1 for that example. Changes in Ianvs core code, shared validation code, workflows, or other shared components should run Tier 0 plus Tier 2 for the full validation target set through `dynamic_code_cicd.yaml`.

Scheduled validation should follow a separate trigger path. Instead of scanning pull request changes, it should scan the full example inventory, refresh Tier 0 static validation results through `static_code_requirement_cicd.yaml`, and run Tier 3 scheduled validation through `dynamic_code_cicd.yaml` as full-inventory smoke validation. In all cases, the resulting failures should pass through the same classification and reporting pipeline so maintainers can compare outcomes consistently across PR-triggered and time-triggered validation.

![Validation-Level Detection Workflow](images/validation-level-detect-workflow.png)

---

## Pull Request Validation Policy

The validation framework should prevent new regressions, not force every contributor to solve all existing maintenance debt.

### PR-Introduced Regression

If a pull request causes a previously validated example to fail, the failure is considered a regression and should be addressed before merge, unless maintainers explicitly approve an exception.

Example:

```text
Base branch: example_b passes
PR branch: example_b fails
Classification: PR regression
```

Expected action:

* PR author should address the regression if it was introduced by their changes.
* If the regression reveals a broader framework issue, maintainers may move the fix to a separate issue or proposal.

---

### Pre-existing Failure

If an example was already marked as broken, not validated, hardware-dependent, dataset-dependent, model-dependent, or expected to fail in the example inventory, the failure should not block unrelated pull requests.

Example:

```text
Base branch: example_b fails
PR branch: example_b fails
Classification: Pre-existing failure
```

Expected action:

* Do not block unrelated PRs.
* Track the failure in the example inventory.
* Create or reference a follow-up issue if maintainers want the example repaired later.

---

### Time-based Maintenance Failure

If a validated example fails due to external dependency drift, Python version changes, changes in dataset availability, model download failures, third-party API updates, or CI runner changes, the failure should be classified as a maintenance failure.

Expected action:

* Record the failure in the example health report.
* Mark the example as broken with a drift/resource subtype, or move it to quarantined or known issue when maintainers decide that normal validation should no longer treat it as an untriaged failure.
* Maintainers may create a follow-up issue or separate restoration proposal.
* The current project does not require unrelated contributors to repair the example.

---

## Failure Classification

CI reports should classify failures rather than just report pass or fail.

Failure types:

```text
Passed
Failed: PR regression
Failed: Known issue
Failed: Pre-existing failure
Failed: Dependency drift
Failed: Dataset/resource drift
Failed: Model/resource drift
Failed: Hardware assumption
Failed: Metric edge case
Failed: CI environment issue
Skipped: Requires external dataset or model download
Skipped: Requires GPU
Quarantined
Unknown
```

Classification method:

```text
If the base branch passes and the PR branch fails:
  classify as potential PR regression.

If both the base branch and PR branch fail:
  classify as pre-existing or time-based failure.

If dependencies changed but the source code did not:
  classify as dependency drift.

If dataset or model download fails:
  classify as dataset/resource drift or model/resource drift.

If the runner image or environment changed:
  classify as CI environment issue.

If an example assumes CUDA but the validation policy requires CPU fallback:
  classify as hardware assumption.

If the metric crashes on empty or malformed results:
  classify as metric edge case.
```

The CI report should include:

* Changed files
* Base branch result
* PR branch result
* Python version
* Dependency versions
* Runner information
* Failure log
* Reproduction command
* Suggested next action
* Whether repair should be handled by a separate issue or proposal

---

## Example Validation Flow

The core CI flow should work without relying on extra approval automation. If maintainers later adopt the workflow approval bot described in Future Work, it can be inserted as a pre-check ahead of the validation jobs.

```text
Pull Request Created
        ↓
Detect Changed Files
        ↓
Load Example Inventory
        ↓
Run Tier 0 Static Validation
        ↓
Run Tier 1 Targeted Example Validation
        ↓
Run Tier 2 Full Example Validation
        ↓
Classify Failures
        ↓
Generate Report
        ↓
Maintainer Review
```

```mermaid
---
config:
  layout: elk
---

flowchart TD
    Contributor[Contributor] --> SubmitPR[Submit Pull Request]
    SubmitPR --> CheckFile

    subgraph ReviewBot[Review Bot]
        CheckFile[Check Change File]
    end

    CheckFile -->|Safe| TriggerValidation
    CheckFile -->|Unsafe| Maintainer[Maintainer]
    Maintainer --> TriggerValidation

    subgraph ExampleRestorationCI[Example Restoration CI System]
        TriggerValidation[Trigger CI Validation]
        ScanExamples[Scan Changed and Related Examples]
        ClassifyIssues[Classify Example Issues]

        TriggerValidation --> ScanExamples
        ScanExamples --> ClassifyIssues
    end

    ClassifyIssues --> ReadReport[Read PR Report]
    ReadReport -->|FAIL| FixIssues[Fix Detected Issues]
    ReadReport -->|PASS| MergeReady[Merge Ready]
    FixIssues --> PushUpdates[Push Updates]
    PushUpdates --> CheckFile
```

If validation passes:

```text
PR can continue review
```

If validation fails due to PR regression:

```text
PR is blocked or requires maintainer decision
Contributor receives feedback
Regression is addressed in the PR or moved to a separate follow-up task
CI runs again
```

If validation fails due to known or time-based failure:

```text
Failure is recorded
Example classification is updated
Maintainer decides whether to quarantine, skip, or create follow-up issue
```

---

## Local Contributor Flow

```text
Contributor modifies an example
        ↓
Runs local validation command
        ↓
Reviews validation and classification result
        ↓
Opens PR
        ↓
CI validates the same rules
        ↓
Maintainer receives validation report
```

The local validation workflow should follow the same comparison model as CI so that changed-file detection and affected-example selection stay consistent. Before validation starts, the local tooling should check whether an `upstream` remote exists for `kubeedge/ianvs`. If it does not exist, the tool should add it. If it already exists, the tool should reuse the configured remote. The tool should then fetch `upstream/main`, select the current `upstream/main` head as the validation target, compute the merge-base between that target and the contributor's local `HEAD`, and detect changed files from the merge-base to the local `HEAD`.

After change detection, the local workflow should create a temporary validation branch and rebase that branch onto `upstream/main` so contributors can validate the effective post-rebase state before opening or updating a pull request. Validation should then run against that rebased temporary branch. If validation completes successfully, the temporary branch should be deleted as part of cleanup.

```mermaid
flowchart TD
    Start([Start Validation])

    CheckUpstream{Check upstream remote}
    AddUpstream[Add upstream remote<br/>kubeedge/ianvs]
    UseUpstream[Use existing<br/>upstream remote]

    FetchUpstream[Fetch upstream/main]
    SelectTarget[Select target<br/>upstream/main HEAD]
    FindMergeBase[Find merge-base]
    DetectChanges[Detect changed files<br/>merge-base to local HEAD]

    CreateValidationBranch[Create temporary validation branch]
    RebaseValidationBranch[Rebase validation branch onto<br/>upstream/main HEAD]

    RunValidation[Run validation]
    ReportValidation[Report validation result]
    CleanupSuccess[Delete validation branch]

    End([End])

    Start --> CheckUpstream

    CheckUpstream -- No --> AddUpstream
    CheckUpstream -- Yes --> UseUpstream

    AddUpstream --> FetchUpstream
    UseUpstream --> FetchUpstream

    FetchUpstream --> SelectTarget
    SelectTarget --> FindMergeBase
    FindMergeBase --> DetectChanges

    DetectChanges --> CreateValidationBranch
    CreateValidationBranch --> RebaseValidationBranch
    RebaseValidationBranch --> RunValidation
    RunValidation --> ReportValidation
    ReportValidation --> CleanupSuccess
    CleanupSuccess --> End
```

The following Mermaid `gitGraph` diagrams make the branch relationship explicit. The happy-path sequence shows why the proposal distinguishes the contributor-owned diff from the rebased validation state. The conflict sequence then shows the same workflow when `upstream/main` has moved in a way that requires manual conflict resolution before validation can proceed.

Happy-path branch evolution:

```mermaid
---
config:
  logLevel: 'debug'
  theme: 'base'
  gitGraph:
    showBranches: true
    showCommitLabel: true
    mainBranchName: 'remote/upstream (KubeEdge/Ianvs)'
---

gitGraph
    commit id: "PR #X"
    commit id: "PR #Y" tag: "Alpha"
    branch "main (fork)"
    commit id: "feat: add new feature"
    commit id: "fix: fix some edge case"
    checkout "remote/upstream (KubeEdge/Ianvs)"
    commit id: "PR #Z" tag: "Beta"
    checkout "main (fork)"
    commit id: "doc: add some doc" tag: "Gamma"
```

```mermaid
---
config:
  logLevel: 'debug'
  theme: 'base'
  gitGraph:
    showBranches: true
    showCommitLabel: true
    mainBranchName: 'remote/upstream (KubeEdge/Ianvs)'
---

gitGraph
    commit id: "PR #X"
    commit id: "PR #Y" tag: "Alpha"
    branch "main (fork)"
    commit id: "feat: add new feature"
    commit id: "fix: fix some edge case"
    commit id: "doc: add some doc" tag: "Gamma"
    checkout "remote/upstream (KubeEdge/Ianvs)"
    commit id: "PR #Z" tag: "Beta"
    branch "temp/validation"
    merge "main (fork)" id: "Rebase main" tag: "Delta"
```

Conflict-path branch evolution:

```mermaid
---
config:
  logLevel: 'debug'
  theme: 'base'
  gitGraph:
    showBranches: true
    showCommitLabel: true
    mainBranchName: 'remote/upstream (KubeEdge/Ianvs)'
---

gitGraph
    commit id: "PR #X"
    commit id: "PR #Y" tag: "Alpha"
    branch "main (fork)"
    commit id: "feat: add new feature"
    commit id: "fix: fix some edge case"
    checkout "remote/upstream (KubeEdge/Ianvs)"
    commit id: "PR with Conflict" tag: "Beta"
    checkout "main (fork)"
    commit id: "doc: add some doc" tag: "Gamma"
```

```mermaid
---
config:
  logLevel: 'debug'
  theme: 'base'
  gitGraph:
    showBranches: true
    showCommitLabel: true
    mainBranchName: 'remote/upstream (KubeEdge/Ianvs)'
---

gitGraph
    commit id: "PR #X"
    commit id: "PR #Y" tag: "Alpha"
    branch "main (fork)"
    commit id: "feat: add new feature"
    commit id: "fix: fix some edge case"
    commit id: "doc: add some doc" tag: "Gamma"
    checkout "remote/upstream (KubeEdge/Ianvs)"
    commit id: "PR with Conflict" tag: "Beta"
    checkout "main (fork)"
    merge "remote/upstream (KubeEdge/Ianvs)" id: "Merge and resolve conflict"
```

```mermaid
---
config:
  logLevel: 'debug'
  theme: 'base'
  gitGraph:
    showBranches: true
    showCommitLabel: true
    mainBranchName: 'remote/upstream (KubeEdge/Ianvs)'
---

gitGraph
    commit id: "PR #X"
    commit id: "PR #Y" tag: "Alpha"
    branch "main (fork)"
    commit id: "feat: add new feature"
    commit id: "fix: fix some edge case"
    checkout "remote/upstream (KubeEdge/Ianvs)"
    commit id: "PR with Conflict" tag: "Beta"
    checkout "main (fork)"
    commit id: "doc: add some doc" tag: "Gamma"
    merge "remote/upstream (KubeEdge/Ianvs)" id: "Merge and resolve conflict"
    checkout "remote/upstream (KubeEdge/Ianvs)"
    branch "temp/validation"
    merge "main (fork)" id: "Rebase main" tag: "Delta"
```

If the local validation workflow encounters a conflict while replaying contributor changes onto the current `upstream/main`, it should fail immediately and instruct the contributor to resolve the conflict first. Validation should be re-run only after the conflict is resolved and the temporary validation branch reflects the intended post-rebase state.

In these diagrams:

* `Alpha`: the old common ancestor between the contributor branch and `upstream/main`
* `Beta`: the current `upstream/main` head at validation time
* `Gamma`: the contributor's local `HEAD`
* `Delta`: the temporary validation branch after rebasing `Gamma` onto `Beta`

This distinction matters because `Alpha -> Gamma` and `Delta` answer different questions:

* `Alpha -> Gamma` tells us which changes belong to the contributor branch itself. That is the right range for merge-base calculation and for identifying PR-owned files.
* `Delta` represents `Gamma` replayed onto the current `Beta`. That is the right validation target for local pre-PR validation, because the eventual merged result must coexist with the current `upstream/main`, not with the older ancestor at `Alpha`.

Two concrete examples:

1. Upstream adds more validation after `Alpha`. Suppose `Gamma` already includes multiple contributor-side changes, such as example code updates, configuration edits, and documentation adjustments. `Alpha -> Gamma` still only tells us what belongs to the contributor branch itself. But if `Beta` adds a stricter smoke test in `.github/workflows/dynamic_code_cicd.yaml`, or a new rule in `.github/workflows/validator/`, the effective validation result is determined by the rebased state `Delta`. In that case, looking only at `Alpha -> Gamma` would still under-estimate the real validation surface, while validating `Gamma` after rebasing it onto `Beta` catches the newly introduced checks.
2. Upstream changes shared core code after `Alpha`. Suppose `Gamma` updates one example, but `Beta` also changes shared runner logic, config parsing, or metric handling used by multiple examples. `Alpha -> Gamma` still shows only the contributor's example edits, but the real question before opening or updating the PR is whether that example still works after those core changes are present. This is why the temporary validation branch rebases onto `Beta`: it can expose rebase conflicts, API mismatches, or extra affected tests that would not appear if validation stayed anchored to `Alpha`.

In short, `Alpha` is still needed to identify the contributor-owned diff, `Beta` is the correct upstream baseline, and `Delta` is the branch that should actually be validated because it represents the contributor changes after rebasing onto the code that the pull request will be merged into.

Local validation commands:

```bash
python .github/workflows/validator/validation_runner.py --static
python .github/workflows/validator/validation_runner.py --example examples/llm_simple_qa
python .github/workflows/validator/validation_runner.py --smoke examples/llm_simple_qa
python .github/workflows/validator/validation_runner.py --example examples/llm_simple_qa --all
python .github/workflows/validator/validation_runner.py --example examples/llm_simple_qa --jsonl
```

For workflow-level verification, contributors should also be able to run the relevant GitHub Actions jobs locally before pushing changes. The proposal should document using `nektos/act` to execute selected workflows or jobs from `.github/workflows/`, so contributors can check whether the same workflow logic used in CI still passes in a local environment.

For contributors who use VS Code, the proposal should also mention the `github-local-actions` extension as a convenient local entry point for running or debugging GitHub Actions jobs backed by `act`. This helps contributors validate workflow behavior before opening a pull request, especially when they changed example validation scripts, workflow definitions, or shared tooling used by CI.

This ensures that CI is not only a maintenance tool but also a contributor workflow tool.

### Maintainer Example Status Flow

Maintainers also need a simple status-driven workflow after CI has already classified example health. The following diagram mirrors the standalone source in `images/maintainer-example-status-user-flow.md` and shows how a maintainer consumes the generated report, decides whether action is needed, and feeds the result back into example status tracking.

The maintainer-facing status model itself is standardized by the diagram below. It defines the primary lifecycle states, the transitions between validation and triage outcomes, and the broken subtypes that CI can auto-classify after a failure has been observed.

![Example Status Transition Standard](images/example-status-STD.png)

```mermaid
flowchart TD

    maintainer[Maintainer\nOpens Repository] --> status[Open examples/README.md]

    status --> decision{Any Failed Examples?}

    decision -- No --> normal[Continue Normal Maintenance]

    decision -- Yes --> report[Open GitHub Actions Report]

    report --> inspect[Inspect Failure Details]

    inspect --> issue[Create Issue or Request Fix]

    issue --> contributor[Contributor Updates Example]

    contributor --> ci[CI Validation Runs]

    ci --> update[Update examples/README.md]

    %% update --> status
```

This flow complements the contributor and CI flow above. The intent is to make `examples/README.md` the maintainer-facing summary, while GitHub Actions reports remain the detailed evidence source for failure triage, follow-up issue creation, and verification after a repair lands.

---

## Related PR Context

This project will reference recent Ianvs example-related work, including PR #406, PR #407, PR #408, Issue/PR #452, and related example maintenance PRs. The purpose is to avoid duplicating previous work and to align the CI validation framework with recent changes in example resource handling, preprocessing behavior, and validation expectations.

The CI framework should treat these PRs as context for designing validation and classification rules, especially around:

* Resource path handling
* Example configuration consistency
* Preprocessing behavior
* Runtime execution assumptions
* Documentation and reproducibility expectations
* LLM example portability

This proposal will not directly repair unrelated examples covered by those PRs unless the change is necessary for the validation framework itself. For `examples/llm_simple_qa`, related PRs identified through Issue/PR #452 should be reviewed and merged or reused when they already solve part of the restoration work. Any remaining blockers should be fixed within this project so that `llm_simple_qa` becomes runnable from a clean clone.

---

## Initial Target Example

The initial validation target will include:

```text
examples/llm_simple_qa
```

This example is suitable as an initial validation target because it exposes common classification cases:

* Hardcoded paths
* Dataset reproducibility issues
* JSONL format risks
* Model loading portability
* Runtime device assumptions
* Evaluation metric edge cases
* Dependency documentation gaps
* README and YAML mismatch

The purpose of using this example is twofold: restore a concrete broken LLM example and validate the CI framework and classification rules against a real repair target.

For this example, the project will implement the following restoration targets and the CI framework should verify that they remain satisfied:

* `examples/llm_simple_qa` runs from a clean Ianvs clone.
* Contributor-specific absolute paths such as `/home/icyfeather/...` are removed.
* README, YAML, dataset, and algorithm paths are aligned.
* Dataset files are provided or generated through documented steps.
* JSONL validation passes.
* Model loading uses a portable model ID or documented override.
* CUDA, MPS, and CPU fallback are supported.
* Example-specific ML dependencies are documented.
* `acc.py` handles empty results without `ZeroDivisionError`.
* Related PRs from Issue/PR #452 are reviewed and reused when applicable.

### `llm_simple_qa` Restoration Tasks

The project will repair `examples/llm_simple_qa` through concrete implementation work, not only validation. The expected restoration tasks include:

* Remove contributor-specific absolute paths such as `/home/icyfeather/...` from README, YAML, dataset references, and model configuration.
* Align README instructions, `benchmarkingjob.yaml`, `testenv.yaml`, dataset paths, and algorithm paths with the current repository layout.
* Provide or document a lightweight dataset preparation path that can be reproduced from a clean clone.
* Add JSONL validation for train and test data and ensure every line is a complete JSON object with the required fields.
* Replace local-only model paths with a portable model ID or clearly documented override mechanism.
* Add CUDA, MPS, and CPU fallback so the example does not assume CUDA-only execution.
* Add the example-specific dependency file and validate the documented dependency installation flow.
* Update metric handling so empty or invalid prediction-answer pairs return a safe result instead of crashing.
* Add the Example-owned Mock fixture and validate the unchanged Hugging Face inference flow through the shared Runtime.
* Reuse existing related PR work from Issue/PR #452 when it already solves part of the restoration.
* Confirm the restored example through CI smoke testing or clearly document any resource-dependent step that cannot run in normal PR CI.

---

## Functional Requirements

### FR-1 Example Inventory

The system shall maintain a list of Ianvs examples and their validation status.

Deliverable:

* Example Classification Matrix

---

### FR-2 Static Validation

The system shall detect common static configuration problems, including documentation consistency and LLM-related portability issues that can be verified without full runtime execution.

Deliverable:

* Static validation script and CI job

---

### FR-3 Dependency Validation

The system shall verify dependency installation and Python compatibility.

Deliverable:

* Dependency validation workflow

---

### FR-4 Dataset and JSONL Validation

The system shall validate dataset path consistency and lightweight dataset structure where practical.

Deliverable:

* Dataset and JSONL validation script for selected examples

---

### FR-5 Example Smoke Testing

The system shall execute selected examples using GitHub Actions.

Deliverable:

* Smoke test workflow for representative examples

---

### FR-6 Mock LLM Runtime Injection

The system shall inject a shared Mock Runtime and Example-owned responses through `IANVS_LLM_MOCK=1` and the inventory-defined `PYTHONPATH`, without changing Example inference code or `prepare_env`. It shall run the existing `llm_simple_qa` Hugging Face flow offline and support offline OpenAI Chat Completions calls through synchronous, asynchronous, and streaming APIs. All substituted-response runs shall be reported as `mocked_llm`.

Deliverable:

* Shared Runtime, common adapter interface, Hugging Face adapter, implemented OpenAI Chat Completions adapter, Example-owned fixtures, and adapter tests

---

### FR-7 Local Contributor Validation

The system shall provide local commands for contributors. It shall also document how contributors can locally execute the relevant GitHub Actions workflows before pushing changes, for example by using `nektos/act` directly or through a VS Code integration such as `github-local-actions`.

The local validation flow shall also detect changed files relative to the current `upstream/main` merge-base and support validation from a temporary branch rebased onto `upstream/main`, so contributors can reproduce the same effective comparison model used by pull request validation.

Deliverable:

* Local validation CLI and workflow-local verification guide

---

### FR-8 Example Health Reporting

The system shall generate reports for maintainers.

Deliverable:

* Example health report or dashboard

---

### FR-9 Tiered CI Validation

The system shall use tiered validation to avoid running every example on every pull request.

Deliverable:

* PR and scheduled validation workflows

---

### FR-10 Failure Classification

The system shall classify validation failures by cause and PR impact.

Deliverable:

* Failure classification logic and CI report output

---

## Non-Functional Requirements

### Reliability

Validation results should be reproducible across local and CI environments.

### Maintainability

New examples should be easy to add to validation workflows.

### Scalability

The framework should support more examples as the repository grows without requiring every example to run on every PR.

### Usability

CI failure messages should help contributors and maintainers understand the classification result and next action.

### Non-intrusiveness

The first version should avoid unnecessary Ianvs core code changes.

### Efficiency

CI runtime should remain reasonable by separating lightweight checks from expensive smoke tests and full validation.

### Separation of Responsibility

The project should clearly separate validation and classification from broad example repair.

Repairing broken examples other than `examples/llm_simple_qa` should be handled by separate proposals, issues, or PRs. `examples/llm_simple_qa` is restored in this proposal as the first concrete reference case, not as a signal that all examples must be restored within this proposal.

---

## Feedback Mechanism

The project will collect feedback from four sources.

### 1. Maintainer Feedback

* PR review comments
* CI result review
* Validation rule discussion
* Classification policy discussion
* Review of `llm_simple_qa` validation targets

### 2. Contributor Feedback

* Whether CI errors are understandable
* Whether local validation commands are easy to use
* Whether failure classifications are actionable
* Whether unrelated failures are handled fairly
* Whether LLM example validation helps reproduce failures locally

### 3. User Feedback

* GitHub Issues from failed example execution
* Reports from new contributors
* Feedback about whether example status reports are useful
* Feedback about whether `llm_simple_qa` can be run from a clean clone

### 4. CI Feedback

* Recurring failure patterns
* Flaky test reports
* Failed workflow logs
* Example health history
* Scheduled validation results

Feedback loop:

```text
CI failure
        ↓
Failure classification
        ↓
Maintainer / contributor review
        ↓
Update validation rule, classification, or inventory
        ↓
Create separate follow-up issue if repair is needed
```

---

## Expected Impact

### Maintainers Impact

* Reduced review burden
* Earlier regression detection
* Better example health visibility
* Better distinction between PR regressions and maintenance failures
* Clearer separation between classification work and restoration work
* Clear validation evidence for `examples/llm_simple_qa`

### Contributors Impact

* Faster feedback before review
* Clearer contribution expectations
* Easier local reproduction of CI failures
* Less risk of being blocked by unrelated historical failures
* Clearer understanding of whether they are responsible for a failure
* More reliable guidance when updating LLM examples

### New Developers Impact

* Better onboarding
* Less time spent guessing which examples are runnable
* More confidence when selecting examples to study or reuse
* A more reliable entry-level LLM QA example if `llm_simple_qa` passes validation

### Enterprise Users Impact

* More confidence when evaluating Ianvs examples
* Clearer understanding of dependency, dataset, model, and hardware requirements
* Better visibility into example stability before internal adoption

### Community Impact

* Higher example transparency
* Better maintenance visibility
* More sustainable example classification and validation process
* A reusable pattern for validating future LLM and non-LLM examples

---

## Roadmap

### Early Phase — Jun 15–Jul 19, 2026

Focus:

* Documentation, example classification, validation rules, and implementation planning.

Outcome:

* Example classification matrix, CI architecture proposal, and a clear implementation plan for the validation framework and `llm_simple_qa` restoration.

---

### Middle Phase — Jul 20–Aug 16, 2026

Focus:

* Main implementation of the CI framework, focusing first on online CI and restoration of `examples/llm_simple_qa`.

Outcome:

* A working online CI prototype, classification reporting, and an initial portable `llm_simple_qa` restoration result.

---

### Late Phase — Aug 17–Sep 14, 2026

Focus:

* Local validation, wrap-up, maintenance guidance, and future work after the main implementation phase.

Outcome:

* Local validation support, final report, example health report, maintenance handover notes, and a future work plan covering broader example onboarding, stronger scheduled validation, and optional workflow-approval automation.

---

## Acceptance Criteria

The project will be considered successful if:

1. An example inventory exists.
2. Selected examples are classified by validation status.
3. CI can detect hardcoded paths and missing files.
4. CI can verify dependency installation for selected examples.
5. CI can validate dataset and JSONL structure for `examples/llm_simple_qa` or classify the dataset requirement clearly.
6. CI can detect local model paths and CUDA-only assumptions for `examples/llm_simple_qa`.
7. CI can run smoke tests for selected examples.
8. CI produces readable classification reports.
9. Contributors can run validation scripts.
10. Maintainers can use CI results during PR review.
11. CI uses tiered validation instead of running every example on every PR.
12. CI can distinguish PR regressions from known or time-based failures.
13. Scheduled validation can detect dependency drift, dataset unavailability, model unavailability, or environment changes.
14. The project clearly documents that broad repair of broken examples is out of scope, except for the explicit `examples/llm_simple_qa` restoration target.
15. Failures requiring repair outside `examples/llm_simple_qa` are recorded for separate follow-up issues or proposals.
16. `examples/llm_simple_qa` is repaired or has mentor-approved remaining blockers documented for clean-environment execution, portable paths, dataset setup, model configuration, hardware fallback, dependency documentation, and metric robustness.
17. CI can run the unchanged `llm_simple_qa` Hugging Face flow offline through the Smoke Test-only Mock Runtime without model download or GPU.
18. CI verifies that the OpenAI adapter supports synchronous, asynchronous, and streaming Chat Completions response shapes without API credentials or external network access.

---

## Risk Analysis

### Risk 1: Some examples require large datasets

Mitigation:

* Mark dataset-dependent examples in the inventory.
* Separate full benchmark runs from smoke tests.
* Classify dataset/resource failures instead of attempting to repair them in this project.

### Risk 2: Some examples require GPU or special hardware

Mitigation:

* Mark hardware requirements in the example inventory.
* Use CPU-compatible smoke tests only when already practical.
* Skip or classify hardware-specific examples unless runners support them.
* For `examples/llm_simple_qa`, validate CUDA/MPS/CPU fallback as a target.

### Risk 3: CI runtime becomes too long

Mitigation:

* Use tiered validation:
  * Static checks on every PR
  * Dependency and smoke tests for changed examples
  * Affected-example validation for shared code changes
  * Full or broader checks on scheduled workflows

### Risk 4: Dependency conflicts are difficult to resolve

Mitigation:

* Record dependency profiles.
* Validate Python version compatibility.
* Classify dependency conflicts clearly.
* Move dependency repair work to separate issues or restoration proposals.

### Risk 5: Core Ianvs changes may be required

Mitigation:

* Avoid core changes in the initial implementation.
* Only propose core changes when repeated failures show a framework-level issue.
* Handle such changes in separate discussions or PRs when necessary.
* Treat missing `preprocess()` as a historical blocker for `llm_simple_qa`, not as an active target, because PR #407 addressed the core-side behavior.

### Risk 6: Contributors may be blocked by unrelated failures

Mitigation:

* Compare base branch and PR branch validation results.
* Use failure classification.
* Do not block unrelated PRs for known or pre-existing failures.
* Allow maintainers to quarantine time-based failures.

### Risk 7: Classification may be mistaken for full restoration

Mitigation:

* Clearly document that this project primarily classifies, validates, and reports example status.
* Add follow-up issue links for examples requiring repair.
* Keep broad restoration work in separate proposals.
* Restore `examples/llm_simple_qa` as the initial reference case, while avoiding any implication that every example will be restored in this proposal.

### Risk 8: LLM model downloads and external provider access may be unstable in CI

Mitigation:

* Allow model-dependent examples to be classified separately.
* Use the Validator-owned Hugging Face adapter with the Example-owned `llm_simple_qa` fixture for its Python Smoke Test.
* Use the Validator-owned OpenAI adapter with Example-owned fixtures for Chat Completions Smoke Tests without API credentials or network access.
* Run each Example in a separate subprocess so fixture modules and response data do not leak between Smoke Tests.
* Record model download failures as model/resource drift when caused by external availability.
* Keep full model execution in scheduled validation if it is too expensive for every PR.

---

## Future Work

After the initial project, the validation framework can be extended with:

* Full example health dashboard
* Scheduled nightly example validation
* Performance regression detection
* Documentation synchronization checks
* Automatic issue creation for broken examples
* Broader coverage across all Ianvs examples
* Release validation reports
* More advanced affected-example detection
* Additional Hugging Face Mock Adapters for API patterns beyond those currently used by `llm_simple_qa`
* Runtime adapters for other specialized SDKs and non-Python Examples
* Additional Example-owned LLM response fixtures, including provider-error and malformed-response cases
* Scheduled real-provider validation alongside low-cost substituted-response CI, with results reported as separate validation tiers
* Dependency lockfile support for reproducible CI
* Historical example health tracking
* Separate restoration proposals based on classification results
* Broader LLM example validation patterns based on `llm_simple_qa`
* Dependency license scanning if maintainers later decide it is needed
* Automatic review bot for workflow approval gating, if maintainers want to reduce repeated manual approval for low-risk pull requests

---

## Summary

This proposal introduces a CI-based validation and classification framework for Ianvs examples. The framework will help maintainers detect regressions before merge, classify example health, and understand whether failures are caused by PR changes, known issues, dependency drift, dataset/resource problems, model/resource problems, hardware assumptions, metric edge cases, or CI environment changes.

The project does not aim to redesign Ianvs core architecture or repair every broken example. Instead, it builds a sustainable validation and classification layer around existing examples while repairing `examples/llm_simple_qa` as the first concrete reference case.

The framework will use tiered validation so that every pull request can receive useful feedback without requiring every example to run every time. Lightweight static checks will run broadly, while dependency checks, dataset checks, LLM-specific checks, and smoke tests will focus on changed or affected examples. Scheduled workflows will provide broader validation to detect time-based failures.

`examples/llm_simple_qa` will serve as the initial repair and validation target because it captures many real maintenance problems in one compact LLM benchmark: hardcoded paths, unclear dataset setup, JSONL validation, local model assumptions, hardware fallback, dependency documentation, and metric robustness. The expected outcome is a more transparent, maintainable, and scalable Ianvs example validation process, with a restored `llm_simple_qa` providing a reusable pattern for validating future examples.

Example repair and restoration work for other examples can then be planned separately based on the classification results.
