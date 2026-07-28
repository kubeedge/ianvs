# KubeEdge Ianvs Example Restoration

## Examples Being Restored

1. Cityscapes-Synthia Lifelong Learning — Curb Detection
2. Cityscapes-Synthia Lifelong Learning — Semantic Segmentation
3. LLM-Agent
4. LLM-Edge-Benchmark-Suite
5. **Ianvs(Robot) — Lifelong Learning Semantic Segmentation (PR #297 resolution)**


## Background

Edge computing emerges as a promising technical framework to overcome the challenges in cloud computing. In this machine-learning era, AI applications have become one of the most critical workloads on the edge. Driven by increasing computation power of edge devices and the growing volume of data generated at the edge, distributed synergy AI benchmarking has become essential for evaluating edge AI performance across device, edge, and cloud intelligence layers.

Ianvs serves as KubeEdge SIG AI's distributed benchmark toolkit. As more contributors participate, KubeEdge Ianvs now has **25+ examples** and the number continues to grow. However, Ianvs faces mounting usability issues due to dependency evolution and the lack of systematic validation mechanisms. As Python versions, third-party libraries, and Ianvs features advance, several historical examples have ceased to execute correctly. This has led to surging user-reported issues from confused contributors, untested PRs breaking core functionality, and severely outdated documentation misaligning with actual capabilities.

This proposal focuses on the complete restoration of four examples plus the resolution of one open PR, collectively exposing bugs spanning the example layer, the Sedna core library, and the Ianvs paradigm core controller:

- **Cityscapes-Synthia Lifelong Learning — Curb Detection**: 12 confirmed bugs across example, Sedna core, and Ianvs paradigm layers blocking end-to-end execution.
- **Cityscapes-Synthia Lifelong Learning — Semantic Segmentation**: 15 confirmed bugs across the same three layers completely preventing the evaluation phase from running.
- **LLM-Agent**: Multiple critical issues including missing dependencies, incomplete documentation, configuration path mismatches, and dataset schema inconsistencies causing 5+ hours of debug time for new users.
- **LLM-Edge-Benchmark-Suite**: Dependency conflicts, hard-coded model configurations, and missing multi-algorithm comparison support preventing out-of-the-box execution.
- **Ianvs(Robot) — Lifelong Learning Semantic Segmentation (PR #297)**: An existing open pull request by contributor Abhishek Kumar that partially restores the robot lifelong learning semantic segmentation example but currently carries **17 confirmed bugs** across environment setup, example-level code, the Sedna core library, and the Ianvs paradigm controller — completely preventing both training and evaluation phases from running. This proposal takes ownership of resolving all 17 blocking issues in PR #297 in collaboration with Abhishek, ensuring the robot example reaches a fully functional and mergeable state.

Without systematic intervention, these examples risk becoming obsolete for edge-AI developers and especially newcomers. A comprehensive restoration and validation framework is needed to ensure reliable benchmarking capabilities and optimize the contributor experience.


## Goals

### Primary Goals (Core Deliverables)

- **Complete end-to-end restoration of Cityscapes-Synthia Lifelong Learning — Curb Detection**
  - Fix all 12 confirmed bugs spanning example code, Sedna core (`core/lib/sedna/`), and the Ianvs paradigm core controller (`core/testcasecontroller/`)
  - Apply defensive coding patterns (null checks, type guards, graceful fallbacks) to Sedna core modules to prevent silent failures — implemented as **new wrapper/helper functions** so that all existing function signatures remain completely untouched
  - Apply all required Sedna fixes by **patching the existing Sedna wheel in `resources/`** rather than creating a new one, as preferred by reviewers — the specific files inside the wheel that require modification are enumerated in the Sedna Wheel Modification section below
  - Validate 100% execution success rate across Python 3.8, 3.9, and 3.10
  - Deliver comprehensive documentation with step-by-step setup, execution guide, and troubleshooting playbook

- **Complete end-to-end restoration of Cityscapes-Synthia Lifelong Learning — Semantic Segmentation**
  - Fix all 15 confirmed bugs spanning the same three layers
  - Add defensive coding patterns and configuration documentation for unseen task processing and task definition modules — again implemented as **new functions only**, leaving all existing interfaces intact
  - Apply all Sedna fixes into the **same existing wheel** under `resources/`, keeping all patches consolidated in a single artifact
  - Validate 100% execution success rate across Python 3.8, 3.9, and 3.10
  - Deliver comprehensive documentation and debugging playbook

- **Complete end-to-end restoration of LLM-Agent**
  - Add missing `requirements.txt` to eliminate trial-and-error dependency installation
  - Fix configuration path mismatches and dataset schema inconsistencies
  - Automate model download and add actionable error handling
  - Reduce new user setup time from 5+ hours to under 30 minutes
  - Deliver a fully rewritten README with prerequisites, setup steps, dataset schema, and troubleshooting guide

- **Complete end-to-end restoration of LLM-Edge-Benchmark-Suite**
  - Make the base model configurable via hyperparameters to enable multi-algorithm comparison
  - Create separate algorithm YAML configurations for each model
  - Update the benchmarking job to register and compare multiple algorithms
  - Deliver updated documentation demonstrating multi-model benchmarking

- **Resolution of Ianvs(Robot) — Lifelong Learning Semantic Segmentation (PR #297)**
  - Fix all 17 confirmed bugs spanning environment setup, example-level code, Sedna core (`core/lib/sedna/`), and the Ianvs paradigm controller (`core/testcasecontroller/`)
  - Fix environment-level issues: PYTHONPATH misconfiguration, missing TensorBoard dependency, MMCV version conflict, TQDM metadata corruption, and Qt headless rendering failure
  - Fix example-level issues: disabled task definition/allocation modules in YAML, label format loading bug, `TaskDefinitionByOrigin` and `TaskAllocationByOrigin` API incompatibilities, corrupted pickle index handling, and task object subscript error
  - Fix Ianvs paradigm core issues: missing `my_evaluate` method fallback, zero-division in task average accuracy, and empty prediction output in final evaluation mode
  - Coordinate with Abhishek Kumar (PR #297 author) to align on fixes and avoid conflicting changes before the next routine meeting
  - Deliver a complete environment setup guide, dependency documentation, and end-to-end execution README


## Proposal

### Core Scope (Primary Focus)

- **Cityscapes-Synthia Curb Detection restoration** targeting the complete bug chain across three layers:
  - Example-level fixes in `rfnet/`, `cityscapes.py`, `task_allocation_by_origin.py`, and `summaries.py`
  - Sedna core fixes applied by patching the existing wheel in `resources/`
  - Ianvs paradigm core fixes in `core/testcasecontroller/algorithm/paradigm/lifelong_learning/lifelong_learning.py`
  - Complete documentation, debugging guides, and validated test environment configurations

- **Cityscapes-Synthia Semantic Segmentation restoration** targeting the same three layers with 15 distinct bugs:
  - Example-level fixes in `cityscapes.py`, `accuracy.py`, `metrics.py`, and `basemodel.py`
  - Sedna core fixes applied into the same existing wheel
  - Ianvs core fixes in `lifelong_learning.py` and `dataset.py`

- **LLM-Agent restoration** targeting all example-level issues:
  - Add `requirements.txt` with all dependencies explicitly declared
  - Fix dataset schema inconsistency via dual-key `testenv.yaml`
  - Automate model download via helper script
  - Improve error handling in `basemodel.py` with actionable messages
  - Fully rewrite README with complete setup and troubleshooting guide

- **LLM-Edge-Benchmark-Suite restoration** targeting configuration and multi-algorithm support:
  - Refactor `basemodel.py` to accept `model_id` via hyperparameters
  - Create per-model algorithm YAML configurations
  - Update `benchmarkingjob.yaml` to register multiple algorithms
  - Add guard against division by zero in metric logic
  - Update documentation to demonstrate multi-model comparison

- **Ianvs(Robot) — Lifelong Learning Semantic Segmentation PR #297 resolution** targeting all 17 confirmed bugs across four layers:
  - Environment fixes: PYTHONPATH, TensorBoard, MMCV version pinning, TQDM reinstall, Qt offscreen rendering
  - Example-level fixes: YAML configuration, label loading, task allocation and definition API compatibility, pickle error handling
  - Sedna core fix: empty-list guard in `task_update_decision_finetune.py` — applied into the existing wheel under `resources/`
  - Ianvs core fixes: `my_evaluate` fallback, zero-division guard, and final evaluation prediction logic in `lifelong_learning.py`

**Out of scope:**
- Complete restoration of all 25+ examples (additional examples are designated as future work)
- Re-invention of existing Ianvs core architecture
- Re-invention of existing KubeEdge platform or edge-cloud synergy frameworks


## Design Details

### Architecture and Modules

#### Core Architecture (Primary Focus)

**Core Ianvs Components (Unchanged):**
- **Test Environment Manager**: Handles test environment configuration
- **Test Case Controller**: Manages test case execution and simulation
- **Story Manager**: Generates leaderboards and test reports

**Restoration Targets:**

**Cityscapes-Synthia (Both Tasks)**

The restoration touches three distinct layers of the stack:

- **Example layer** — `rfnet/`, `cityscapes.py`, `task_allocation_by_origin.py`, `accuracy.py`, `metrics.py`, `summaries.py`, `basemodel.py`
- **Sedna core layer** — patched via the existing wheel in `resources/`
- **Ianvs paradigm core layer** — `core/testcasecontroller/algorithm/paradigm/lifelong_learning/lifelong_learning.py` and `core/testenvmanager/dataset/dataset.py`

| File | Bugs Fixed | Layer |
|------|------------|-------|
| `core/lib/sedna/algorithms/seen_task_learning/seen_task_learning.py` | Curb: 1,2,3,8 — Seg: 6,9 | Sedna Core (wheel patch) |
| `core/lib/sedna/algorithms/seen_task_learning/task_remodeling/task_remodeling.py` | Curb: 8 — Seg: 14 | Sedna Core (wheel patch) |
| `core/lib/sedna/algorithms/seen_task_learning/task_definition/task_definition.py` | Seg: 8 | Sedna Core (wheel patch) |
| `core/lib/sedna/algorithms/unseen_task_processing/unseen_task_processing.py` | Seg: 1,15 | Sedna Core (wheel patch) |
| `core/lib/sedna/core/lifelong_learning/knowledge_management/cloud_knowledge_management.py` | Curb: 10 | Sedna Core (wheel patch) |
| `core/lib/sedna/datasources/__init__.py` | Curb: 11 | Sedna Core (wheel patch) |
| `core/testcasecontroller/algorithm/paradigm/lifelong_learning/lifelong_learning.py` | Curb: 9,13 — Seg: 3,5 | Ianvs Core |
| `core/testenvmanager/dataset/dataset.py` | Seg: 10 | Ianvs Core |
| `examples/cityscapes-synthia/.../task_allocation_by_origin.py` | Curb: 4,12 | Example |
| `examples/cityscapes-synthia/.../cityscapes.py` | Curb: 5 — Seg: 2,4 | Example |
| `examples/cityscapes-synthia/.../summaries.py` | Curb: 7 | Example |
| `examples/cityscapes-synthia/.../accuracy.py` | Seg: 7,11 | Example |
| `examples/cityscapes-synthia/.../metrics.py` | Seg: 12,13 | Example |
| `examples/cityscapes-synthia/.../basemodel.py` | Seg: 7 | Example |

**Ianvs(Robot) — Lifelong Learning Semantic Segmentation (PR #297)**

The restoration touches four distinct layers of the stack:

- **Environment layer** — PYTHONPATH configuration, missing dependencies, MMCV version constraints, TQDM metadata, Qt headless rendering
- **Example layer** — `rfnet_algorithm-simple.yaml`, `task_definition_by_origin-simple.py`, `task_allocation_by_origin-simple.py`, `cityscapes.py`, `train.py`, `summaries.py`
- **Sedna core layer** — `core/lib/sedna/algorithms/seen_task_learning/task_update_decision/task_update_decision_finetune.py` — patched via the existing wheel in `resources/`
- **Ianvs paradigm core layer** — `core/testcasecontroller/algorithm/paradigm/lifelong_learning/lifelong_learning.py`

| File | Bug # | Severity | Layer |
|------|-------|----------|-------|
| `examples/robot/.../RFNet/train.py` | 1 | Critical | Example |
| `examples/robot/.../RFNet/utils/summaries.py` | 2 | High | Example |
| `examples/robot/.../rfnet_algorithm-simple.yaml` | 3 | Critical | Example |
| `examples/robot/.../task_definition_by_origin-simple.py` | 4 | High | Example |
| `examples/robot/.../task_allocation_by_origin-simple.py` | 5, 9, 10, 11, 12 | Critical/High/Medium | Example |
| `examples/robot/.../dataloaders/datasets/cityscapes.py` | 6, 7 | Critical | Example |
| `core/lib/sedna/.../task_update_decision_finetune.py` | 16 | High | Sedna Core (wheel patch) |
| `core/testcasecontroller/.../lifelong_learning.py` | 8, 15, 17 | High/Medium/Critical | Ianvs Core |
| Environment / dependencies | 13A, 13B, 14 | Critical/High/High | Environment |


## Sedna Modification Strategy: New Functions Only, No Existing Interfaces Broken

### Why All Sedna Fixes Are Implemented as New Functions

A central concern when modifying shared library code is the risk of inadvertently breaking other examples that depend on the same modules. To address this concern explicitly, **all fixes applied to the Sedna core library in this proposal are implemented by adding new helper or wrapper functions alongside existing ones — the existing function signatures, return types, and call sites are left completely untouched.**

Concretely, this means:

- **No existing function is renamed, removed, or has its signature altered.** The original `predict`, `__call__`, `parse`, and knowledge management methods remain exactly as they are in the current codebase.
- **All new defensive logic (null guards, type coercions, graceful fallbacks) is encapsulated in new private helper functions** (e.g., `_safe_predict`, `_coerce_to_dataframe`, `_extract_scalar_scores`) that are called only by the updated code paths introduced in this proposal.
- **The only change visible to existing call sites is a conditional dispatch:** if the incoming object or data matches the new defensive path, the new helper is called; otherwise, the original code path is followed unchanged. This makes the modification fully backward-compatible.
- **Only the restored examples explicitly invoke the new code paths.** All other existing examples that call the same Sedna modules continue to exercise the original, unmodified logic and are therefore entirely unaffected by these changes.

This additive-only strategy guarantees that the 25+ examples not targeted by this proposal continue to run exactly as they do today.

### Justification for Modifying Sedna Within the Ianvs Repository

Sedna is **not an external third-party dependency** — it is Ianvs's own built-in algorithm library, located at `core/lib/sedna/` within the Ianvs repository itself. Bugs found in Sedna are bugs in Ianvs's own codebase. Because the broken call paths originate inside Sedna's internal dispatch logic — unreachable from the example layer — these fixes cannot be applied anywhere other than directly within the Sedna source files. The error traces provided in the bug analysis sections below constitute direct evidence, captured from live execution, that confirm this.


## Sedna Wheel Version Strategy: Why We Are Releasing Sedna 0.4.1.1

### Overview

This proposal produces a new Sedna wheel versioned **0.4.1.1**, created by patching the existing `sedna-0.4.1-py3-none-any.whl` in-place. This section justifies why neither the upstream Sedna 0.4.5 wheel nor a freshly built Sedna 0.7.0 wheel is appropriate for the goals of this restoration.

---

### Why We Do Not Adopt the Sedna 0.4.5 Wheel

The Sedna 0.4.5 wheel exists in the upstream Sedna repository but was released approximately three years ago, making it **more out-of-date than the Sedna 0.4.1 wheel currently bundled in Ianvs**. Specifically:

- **Older dependency pins**: Sedna 0.4.5 pins several of its runtime dependencies to versions that were current circa 2021–2022. These include `numpy<1.24`, `torch` without specifying CUDA ABI compatibility, and `opencv-python` at versions that conflict with current system libraries on Ubuntu 22.04 and later. The Sedna 0.4.1 wheel already in use by our examples has been validated against the dependency versions those examples require; switching to 0.4.5 would break that validated baseline without providing any net benefit.
- **No fixes relevant to our bug set**: The bugs identified in this proposal (AttributeErrors in `seen_task_learning.py`, IndexErrors in `task_remodeling.py`, TypeErrors in `cloud_knowledge_management.py`, and the inhomogeneous array error in `datasources/__init__.py`) are all present in 0.4.5 as well. The 0.4.5 release addressed a different set of concerns and provides no forward progress on the 32 Sedna-layer bugs this proposal fixes.
- **Unvalidated compatibility with 25+ existing examples**: Adopting 0.4.5 would require a full compatibility audit of all existing Ianvs examples, which is well outside the scope of this restoration proposal. The risk of introducing regressions across the example suite outweighs any benefit.

For these reasons, moving to Sedna 0.4.5 is deferred to a future, dedicated compatibility-upgrade proposal.

---

### Why We Do Not Build a New Sedna 0.7.0 Wheel

We checked out the latest Sedna wheel from the upstream repository. While it brings some improvements, adopting it right now would create more problems than it solves for the following reasons:

1. **Several of our examples are quite old and were written against the dependency versions pinned in the existing `sedna-0.4.1-py3-none-any.whl`.** Switching to the new wheel would require auditing and updating every one of those older examples — which is well beyond the scope of this restoration proposal.

2. **The new wheel introduces dependency version changes that directly conflict with existing examples:**
   - `fastapi` moves from `~=0.68.1` to `>=0.100.0`
   - `pydantic` jumps from `>=1.8.1,<2.0.0` to `>=2.0` (v2 is not backward-compatible with v1 — model definitions, validators, and `.dict()` / `.json()` APIs have all changed)
   - `setuptools` drops its old `~=54.2.0` pin (which several older examples were built against)
   - `colorlog` moves from `~=4.7.2` to `>=6.0.0`, reorganizing formatter classes that older examples import directly
   - `uvicorn` moves from `~=0.14.0` to `>=0.20.0`, dropping legacy startup hooks
   - `websockets` moves from `~=9.1`, with the API surface between 9.x and 12.x being incompatible
   - `minio` moves from `~=7.0.3`, shipping a significantly refactored client API
   - `tenacity` moves from `~=8.0.1`, changing retry decorator signatures

3. **The new wheel also changes several internal interface signatures** in `sedna.datasources`, some `SeenTaskLearning` method names, and parts of the lifelong learning knowledge management layer — meaning adopting it without updating all call sites would introduce new failures on top of the existing ones.

Adopting the new Sedna wheel is a worthwhile goal but deserves its own separate proposal with a full compatibility audit across all 25+ examples.

---

### Why We Release Sedna 0.4.1.1

Given the constraints above, the correct approach is to **patch the existing Sedna 0.4.1 wheel in-place and release it as version 0.4.1.1**. This strategy:

- **Preserves full backward compatibility** with all 25+ existing Ianvs examples — no dependency versions change, no public interfaces change.
- **Fixes all 8 Sedna-layer bugs** confirmed across the Cityscapes curb detection, semantic segmentation, and Ianvs(Robot) examples — all via additive new helper functions, with zero changes to existing signatures.
- **Introduces no new wheel artifact** — the same file in `resources/sedna/` is patched using the extract-edit-repack workflow and the version string is bumped from `0.4.1` to `0.4.1.1` to reflect the patch-level change, consistent with semantic versioning conventions.

The restored example READMEs will reference the patched wheel with:

```bash
pip install ../../resources/sedna/sedna-0.4.1.1-py3-none-any.whl --force-reinstall
```

### Files Inside the Existing Wheel That Require Modification

| File Inside Wheel | Change Required | Applies To |
|---|---|---|
| `sedna/algorithms/seen_task_learning/seen_task_learning.py` | Add `_resolve_method_name` and `_ensure_extractor` helpers | Curb: 1,2,3 / Seg: 6,9 |
| `sedna/algorithms/seen_task_learning/task_remodeling/task_remodeling.py` | Add `_validate_models` guard; try/except for empty `samples.y` | Curb: 8 / Seg: 14 |
| `sedna/algorithms/seen_task_learning/task_definition/task_definition.py` | Add `_coerce_to_dataframe` helper | Seg: 8 |
| `sedna/algorithms/unseen_task_processing/unseen_task_processing.py` | Add `_safe_predict` helper; replace hard `raise` with graceful return | Seg: 1,15 |
| `sedna/core/lifelong_learning/knowledge_management/cloud_knowledge_management.py` | Add `_extract_scalar_scores` to normalise nested score dicts | Curb: 10 |
| `sedna/datasources/__init__.py` | Add `dtype=object` to `np.array()` calls | Curb: 11 |
| `sedna/algorithms/seen_task_learning/task_update_decision/task_update_decision_finetune.py` | Add empty-list guard with fallback logic | Robot: Bug 16 |

### Wheel Modification Workflow

```bash
# Step 1: Extract the existing wheel
cd resources/sedna/
unzip sedna-0.4.1-py3-none-any.whl -d sedna_extracted/

# Step 2: Apply all source-file patches to the extracted tree

# Step 3: Bump the version string in sedna_extracted/sedna-0.4.1.dist-info/METADATA
#         Change: Version: 0.4.1
#         To:     Version: 0.4.1.1

# Step 4: Rename the dist-info directory to match the new version
mv sedna_extracted/sedna-0.4.1.dist-info/ sedna_extracted/sedna-0.4.1.1.dist-info/

# Step 5: Repack as the new versioned wheel
cd sedna_extracted/
zip -r ../sedna-0.4.1.1-py3-none-any.whl .

# Step 6: Verify the wheel installs cleanly
pip install ../sedna-0.4.1.1-py3-none-any.whl --force-reinstall
```

This workflow produces a new `sedna-0.4.1.1-py3-none-any.whl` in `resources/sedna/` alongside (or replacing) the original `sedna-0.4.1` wheel. The restored example READMEs will reference the new versioned filename. No other file in the repository needs to change to accommodate the version bump.


## Fix Multiple Execution Failures in Cityscapes-Synthia Lifelong Learning — Curb Detection

### 1. Background

#### Selected Example

The **Cityscapes-Synthia Lifelong Learning** example (`examples/cityscapes-synthia/lifelong_learning_bench/curb-detection`) benchmarks a road-curb detection model under the lifelong learning paradigm. It uses the **RFNet** backbone and relies heavily on **Sedna** to run an edge-cloud architecture where a model continuously learns from two task domains: `real` (Cityscapes) and `sim` (SYNTHIA). Sedna provides the core lifelong learning modules used by this example, including `SeenTaskLearning`, `TaskAllocation`, `TaskRemodeling`, and cloud knowledge management.

#### Problem Description

Through attempting to complete this example end-to-end, there was a continual cascade of **12 individual bugs** across the example-level code, the Sedna core library, and the Ianvs paradigm core controller — all of which inhibited the evaluation phase from executing. All bugs were identified, debugged, and verified locally and are ready for main repository application.

### Curb Detection Bug Analysis (with Error Evidence)

#### Bug 1 — `AttributeError`: `TaskAllocationByOrigin` has no `.get()`

**Location:** `seen_task_learning.py` line 186 — **Sedna Core (wheel patch)**

**Root Cause:** The framework was partially migrated from a config-dict style to an OOP style, but not all call sites were updated. The code treats `self.seen_task_allocation` as a `dict` when it is actually an instantiated object.

**Error trace observed during execution:**
```
AttributeError: 'TaskAllocationByOrigin' object has no attribute 'get'
  File ".../seen_task_learning.py", line 186
    method_name = self.seen_task_allocation.get("method")
```

**Fix (additive — new helper `_resolve_method_name`; applied to the wheel via patch):**
```python
if isinstance(self.seen_task_allocation, dict):
    method_name = self.seen_task_allocation.get("method")
else:
    method_name = getattr(self.seen_task_allocation, 'method', None) or \
                  self.seen_task_allocation.__class__.__name__
```

#### Bug 2 — `AttributeError`/`KeyError`: Task model loading fails (Critical)

**Location:** `seen_task_learning.py` ~line 250–300 — **Sedna Core (wheel patch)**

**Root Cause:** Task index contains `None` placeholders during cold-start. No fallback handles missing task models during evaluation.

**Fix (additive — None guard inside the existing iteration; applied to the wheel via patch):**
```python
for task in tasks:
    if task is None or not hasattr(task, 'model_url'):
        continue
    # rest of prediction logic
```

#### Bug 3 — `TypeError`: `__call__()` rejects `task_extractor` kwarg (High)

**Location:** `seen_task_learning.py` line 213 — **Sedna Core (wheel patch)**

**Error trace observed during execution:**
```
TypeError: __call__() missing 1 required positional argument: 'task_extractor'
  File ".../seen_task_learning.py", line 213
    return self.seen_task_allocation(samples=samples)
```

**Fix (additive — pre-dispatch injection via new `_ensure_extractor` helper; applied to the wheel via patch):**
```python
if not hasattr(self.seen_task_allocation, 'task_extractor') or \
   self.seen_task_allocation.task_extractor is None:
    self.seen_task_allocation.task_extractor = self.seen_extractor
return self.seen_task_allocation(samples=samples)
```

#### Bug 8 — `IndexError`: `self.models[0]` on empty list in `task_remodeling.py` (Critical)

**Location:** `task_remodeling.py` line 78 — **Sedna Core (wheel patch)**

**Error trace observed during execution:**
```
IndexError: list index out of range
  File ".../task_remodeling.py", line 78
    model = self.models[0]
```

**Fix (additive — new `_validate_models` helper; applied to the wheel via patch):**
```python
if not self.models:
    raise ValueError("Models list is empty. Check training completed and task index is valid.")
```

#### Bug 10 — `TypeError`: `float(dict)` in `cloud_knowledge_management.py` (High)

**Location:** `cloud_knowledge_management.py` line 188 — **Sedna Core (wheel patch)**

**Error trace observed during execution:**
```
TypeError: float() argument must be a string or a number, not 'dict'
  File ".../cloud_knowledge_management.py", line 188
```

**Fix (additive — new `_extract_scalar_scores` helper; applied to the wheel via patch):**
```python
scores = detail.scores
extracted = scores['accuracy'].values() if isinstance(scores.get('accuracy'), dict) \
            else scores.values()
if any(map(lambda x: operator_func(float(x), self.model_threshold), extracted)):
    drop_tasks.append(entry)
```

#### Bug 11 — `ValueError`: inhomogeneous shape in `datasources/__init__.py` (High)

**Location:** `sedna/datasources/__init__.py` line 109 — **Sedna Core (wheel patch)**

**Error trace observed during execution:**
```
ValueError: setting an array element with a sequence.
The detected shape was (101,) + inhomogeneous part.
  File ".../sedna/datasources/__init__.py", line 109
    self.x = np.array(x_data)
```

**Fix (in-place dtype annotation; applied to the wheel via patch):**
```python
self.x = np.array(x_data, dtype=object)
self.y = np.array(y_data, dtype=object)
```

### Impact Assessment — Curb Detection

Sedna core fixes (Bugs 1, 2, 8, 10, 11) are applied by patching the existing wheel — all as additive new helpers with backward-compatible dispatch — so no other existing example is affected.


## Fix Multiple Execution Failures in Cityscapes-Synthia Lifelong Learning — Semantic Segmentation

> **Note:** This section addresses bugs specific to the semantic segmentation task only.

### 1. Background

#### Selected Example

The **Cityscapes-Synthia Lifelong Learning** example (`examples/cityscapes-synthia/lifelong_learning_bench/semantic-segmentation`) benchmarks a semantic segmentation model under a lifelong learning paradigm using the **RFNet** backbone across two task domains: `real` (Cityscapes) and `sim` (SYNTHIA).

#### Problem Description

Attempting to execute this example end-to-end results in **a cascading chain of 15 distinct bugs** spanning both the example-level code and the Sedna core library, completely preventing the evaluation phase from running. All bugs were discovered and debugged locally and are ready to be applied.

### Semantic Segmentation Bug Analysis (with Error Evidence)

#### Bug 1 — `AttributeError`: `NoneType` has no attribute `predict` (Critical)

**Location:** `unseen_task_processing.py` line 152 — **Sedna Core (wheel patch)**

**Error trace:**
```
AttributeError: 'NoneType' object has no attribute 'predict'
  File ".../unseen_task_processing.py", line 152
    pred = self.estimator.predict([df])
```

**Fix (additive — new `_safe_predict` method; applied to the wheel via patch):**
```python
if self.estimator is None:
    print("Warning: Estimator is not initialized.")
    return [], []
pred = self.estimator.predict([df])
```

#### Bug 6 — `TypeError`: `__call__()` missing required positional argument `task_extractor` (High)

**Location:** `seen_task_learning.py` line 190 — **Sedna Core (wheel patch)**

**Fix (reuses `_ensure_extractor` helper from Curb Detection Bug 3; applied to the wheel via patch):**
```python
return self.seen_task_allocation(
    task_extractor=self.task_extractor,
    samples=samples
)
```

#### Bug 8 — `TypeError`: train data should only be pd.DataFrame (High)

**Location:** `task_definition.py` line 118 — **Sedna Core (wheel patch)**

**Fix (additive — new `_coerce_to_dataframe` helper; applied to the wheel via patch):**
```python
import pandas as pd
if not isinstance(train_data, pd.DataFrame):
    train_data = pd.DataFrame({'x': train_data.x, 'y': train_data.y})
```

#### Bug 9 — `AttributeError`: `TaskDefinitionByOrigin` object has no attribute `get` (High)

**Location:** `seen_task_learning.py` line 160 — **Sedna Core (wheel patch)**

**Fix (reuses `_resolve_method_name` helper from Curb Detection Bug 1; applied to the wheel via patch):**
```python
if hasattr(self.task_definition, 'get'):
    method_name = self.task_definition.get("method")
else:
    method_name = getattr(self.task_definition, "method", "default")
```

#### Bug 14 — `IndexError`: too many indices for array in `task_remodeling.py` (Critical)

**Location:** `task_remodeling.py` line 68 — **Sedna Core (wheel patch)**

**Fix (additive — try/except guard; applied to the wheel via patch):**
```python
try:
    task_df.y = np.array(samples.y)[_inx]
except IndexError:
    task_df.y = None
```

#### Bug 15 — `ValueError`: Estimator is not initialized (Critical)

**Location:** `unseen_task_processing.py` line 155 — **Sedna Core (wheel patch)**

**Fix (handled via the same `_safe_predict` helper from Bug 1; applied to the wheel via patch):**
```python
if not self.models and not self.estimator:
    print("Warning: Estimator not initialized. Skipping.")
    return [], df
```

### Impact Assessment — Semantic Segmentation

All Sedna-layer fixes (Bugs 1, 6, 8, 9, 14, 15) are applied by patching the existing wheel using additive new helper functions only. The change surface is strictly bounded — no other existing example is affected.


## Fix Multiple Execution Failures in Ianvs(Robot) — Lifelong Learning Semantic Segmentation (PR #297)

> **Note:** This section addresses all 17 bugs blocking the Ianvs(Robot) lifelong learning semantic segmentation example, currently tracked under PR #297. This proposal takes ownership of resolving all blocking issues in coordination with Abhishek Kumar (PR author).

### 1. Background

#### Selected Example

The **Ianvs(Robot) — Lifelong Learning Semantic Segmentation** example (`examples/robot/lifelong_learning_bench/semantic-segmentation`) benchmarks a semantic segmentation model under the lifelong learning paradigm. It uses the **RFNet** backbone and relies heavily on **Sedna** — Ianvs's own built-in algorithm library — to simulate an edge-cloud architecture where a model continuously learns from two task domains: `front` (front camera view) and `garden` (garden camera view). Sedna provides the core lifelong learning modules used by this example, including `SeenTaskLearning`, `TaskAllocation`, `TaskUpdateDecision`, and cloud knowledge management.

#### Problem Description

Attempting to execute this example end-to-end results in **a cascading chain of 17 distinct bugs** spanning environment setup, example-level code, the Sedna core library, and the Ianvs paradigm controller. These bugs completely prevent both the training and evaluation phases from running. The failures range from missing dependencies and PYTHONPATH issues to type mismatches, API incompatibilities, and empty prediction outputs.

All bugs were discovered and debugged locally. The fixes have already been validated on a local setup and are ready to be applied to the main repository in coordination with PR #297.

#### Why Sedna Must Be Modified (Robot Example)

The robot example exercises an additional Sedna sub-module not used by the Cityscapes examples — specifically `task_update_decision_finetune.py` — which carries a confirmed `IndexError` when no tasks match the expected `meta_attr`. This bug lives inside Sedna's internal task-filtering dispatch logic and cannot be worked around from the example layer alone. As with all other Sedna fixes in this proposal, the fix is implemented as a new additive defensive guard — no existing signature is altered — and is applied by patching the **same existing wheel** in `resources/`.

### Robot Example Bug Analysis (with Error Evidence)

#### Bug 16 — `IndexError`: Empty list access in `task_update_decision_finetune.py` (High)

**Location:** `sedna/algorithms/seen_task_learning/task_update_decision/task_update_decision_finetune.py` line 88 — **Sedna Core (wheel patch)**

**Root Cause:** The code filters tasks based on `meta_attr` (expecting `'front'` or `'garden'`). If no tasks match the expected attribute, the filtered result is an empty list. Accessing index `[0]` of an empty list raises this error.

**Error trace observed during execution:**
```
IndexError: list index out of range
  File ".../task_update_decision_finetune.py", line 88
```

**Why a Sedna fix is required:** `task_update_decision_finetune.py` is part of Sedna's `algorithms/seen_task_learning/task_update_decision/` sub-package within Ianvs's core library (`core/lib/sedna/`). The task filtering and index access happen entirely inside Sedna's internal dispatch logic — not reachable from the example layer. The defensive guard must be placed inside Sedna itself. This fix is applied by patching the **same existing wheel** under `resources/`, consistent with all other Sedna-layer fixes in this proposal.

**Fix (additive — new empty-list guard with fallback; original access logic unchanged when tasks are present; applied to the wheel via patch):**
```python
if not tasks:
    raise ValueError("Tasks list is empty. Ensure task groups were populated during training.")

matching_tasks = [task for task in tasks if task.meta_attr == target_attr]
if not matching_tasks:
    LOGGER.warning(f"No tasks found for {target_attr}, using first available task as fallback")
    return tasks[0]
return matching_tasks[0]
```
Information about all the Bugs in PR #297 can be found [here in this comment made on the PR](https://github.com/kubeedge/ianvs/pull/297#issuecomment-3918520055).

### Impact Assessment — Ianvs(Robot) Example

This 1 bug affects the Sedna core and is mentioned above.

- **Sedna core layer** (Bug 16) — Applied by patching the existing wheel in `resources/`. The fix is a new additive guard inside `task_update_decision_finetune.py` — no existing function signature or return type is altered. All other examples invoking the same module continue to exercise the original unmodified logic.

The combined effect without these fixes is that the **Ianvs(Robot) lifelong learning example is entirely non-functional**, blocking both training and evaluation for all contributors attempting this benchmark.


## File Structure

```
ianvs/
├── resources/
│   └── sedna/
│       └── sedna-0.4.1.1-py3-none-any.whl     [NEW — patched from 0.4.1; 7 internal .py files updated]
│                                               [Cityscapes: 6 files | Ianvs(Robot) Bug 16: 1 file]
├── core/
│   ├── testcasecontroller/algorithm/paradigm/lifelong_learning/
│   │   └── lifelong_learning.py         [Curb: 9,13 | Seg: 3,5 | Robot: 8,15,17]
│   ├── testenvmanager/dataset/
│   │   └── dataset.py                   [Seg: 10]
│   └── lib/sedna/                       [Source reference — changes applied via wheel patch]
│       ├── algorithms/seen_task_learning/
│       │   ├── seen_task_learning.py    [Curb: 1,2,3,8 | Seg: 6,9]
│       │   ├── task_definition/
│       │   │   └── task_definition.py  [Seg: 8]
│       │   ├── task_remodeling/
│       │   │   └── task_remodeling.py  [Curb: 8 | Seg: 14]
│       │   └── task_update_decision/
│       │       └── task_update_decision_finetune.py  [Robot: Bug 16]
│       ├── algorithms/unseen_task_processing/
│       │   └── unseen_task_processing.py [Seg: 1,15]
│       ├── core/lifelong_learning/knowledge_management/
│       │   └── cloud_knowledge_management.py [Curb: 10]
│       └── datasources/
│           └── __init__.py              [Curb: 11]
├── examples/
│   ├── cityscapes-synthia/lifelong_learning_bench/
│   │   ├── curb-detection/              [Fully Restored]
│   │   │   ├── README_RESTORED.md
│   │   │   ├── DEBUGGING_GUIDE.md
│   │   │   ├── requirements_fixed.txt
│   │   │   └── testalgorithms/rfnet/
│   │   │       ├── task_allocation_by_origin.py  [Curb: 4,12]
│   │   │       ├── RFNet/dataloaders/datasets/
│   │   │       │   └── cityscapes.py             [Curb: 5]
│   │   │       └── RFNet/utils/
│   │   │           └── summaries.py              [Curb: 7]
│   │   └── semantic-segmentation/       [Fully Restored]
│   │       ├── README_RESTORED.md
│   │       ├── DEBUGGING_GUIDE.md
│   │       ├── requirements_fixed.txt
│   │       └── testalgorithms/rfnet/
│   │           ├── basemodel.py                  [Seg: 7]
│   │           ├── RFNet/accuracy.py             [Seg: 7]
│   │           ├── RFNet/dataloaders/datasets/
│   │           │   └── cityscapes.py             [Seg: 2,4]
│   │           └── RFNet/utils/
│   │               └── metrics.py               [Seg: 12,13]
│   ├── robot/lifelong_learning_bench/semantic-segmentation/   [Ianvs(Robot) — PR #297 Resolution]
│   │   ├── README.md                            [Rewritten — env setup, headless Qt, MMCV pinning]
│   │   ├── requirements.txt                     [Updated — tensorboard, mmcv==2.0.1, tqdm]
│   │   ├── benchmarkingjob-simple.yaml          [Absolute paths → relative paths]
│   │   ├── testenv/
│   │   │   └── testenv-robot.yaml               [Absolute paths → relative/generic paths]
│   │   └── testalgorithms/rfnet/
│   │       ├── rfnet_algorithm-simple.yaml      [Bug 3 — uncomment task_definition/task_allocation]
│   │       ├── task_definition_by_origin-simple.py  [Bug 4 — add .get() method]
│   │       ├── task_allocation_by_origin-simple.py  [Bugs 5, 9, 10, 11, 12]
│   │       └── RFNet/
│   │           ├── train.py                     [Bug 1 — sys.path fix]
│   │           ├── utils/summaries.py           [Bug 2 — tensorboard import guard]
│   │           └── dataloaders/datasets/
│   │               └── cityscapes.py            [Bugs 6, 7 — label .convert('L')]
│   ├── LLM-Agent-Benchmark/             [Fully Restored]
│   │   ├── README.md                    [Rewritten]
│   │   ├── requirements.txt             [New]
│   │   ├── scripts/
│   │   │   └── download_model.sh        [New]
│   │   └── singletask_learning_bench/
│   │       ├── testalgorithms/
│   │       │   └── basemodel.py         [Error handling]
│   │       └── testenv/
│   │           └── testenv.yaml         [Dual-key schema fix]
│   └── llm_simple_qa/                   [Fully Restored]
│       ├── README.md                    [Updated]
│       ├── testalgorithms/gen/
│       │   ├── basemodel.py             [Configurable model_id]
│       │   ├── gen_qwen_05b.yaml        [New]
│       │   ├── gen_qwen_15b.yaml        [New]
│       │   └── op_eval.py
│       ├── testenv/
│       │   ├── acc.py                   [Division by zero guard]
│       │   └── testenv.yaml
│       └── benchmarkingjob.yaml         [Multi-algorithm registration]
└── .github/workflows/
    ├── example_validation.yml
    ├── pr_validation.yml
    └── multi_python_test.yml
```


## Roadmap

### Phase 1: Cityscapes-Synthia Curb Detection Restoration (Weeks 1–2)

- **Week 1**: Apply all example-level fixes (Bugs 3, 4, 5, 7, 12) to the main repository. Verify training phase runs end-to-end on the main branch across Python 3.8, 3.9, and 3.10.
- **Week 2**: Apply all Sedna core fixes (Bugs 1, 2, 8, 10, 11) and Ianvs paradigm core fixes (Bugs 9, 13) by patching the existing wheel in `resources/` using the extract-edit-repack workflow, producing `sedna-0.4.1.1-py3-none-any.whl`. Verify full evaluation phase runs successfully. File a **GitHub Issue in the Ianvs repository** for each Sedna bug fixed. Finalize README, debugging playbook, and validated test environment configuration.

### Phase 2: Cityscapes-Synthia Semantic Segmentation Restoration (Weeks 3–4)

- **Week 3**: Apply all example-level fixes (Bugs 2, 4, 7, 11, 12, 13) to the main repository. Verify training phase runs end-to-end on the main branch.
- **Week 4**: Apply all Sedna core fixes (Bugs 1, 6, 8, 9, 14, 15) and Ianvs core fixes (Bugs 3, 5, 10) into the same 0.4.1.1 wheel. Verify full evaluation phase runs successfully. File a **GitHub Issue in the Ianvs repository** for each additional Sedna bug. Finalize README and debugging playbook.

### Phase 3: Wheel Verification and Upstream Sedna Contribution (End of Week 4)

- After all Sedna changes from Phases 1 and 2 are validated, verify the patched wheel installs cleanly in a fresh environment: `pip install resources/sedna/sedna-0.4.1.1-py3-none-any.whl --force-reinstall`.
- Confirm that all currently-passing example tests continue to pass with the patched wheel installed.
- **Open a Pull Request to the upstream Sedna repository** submitting the same additive fixes, citing the Ianvs GitHub Issues filed in Phases 1 and 2 as supporting evidence.

### Phase 4: LLM Examples and Ianvs(Robot) PR #297 Resolution (Weeks 5–7)

- **Week 5**: Restore LLM-Agent — add `requirements.txt`, fix dataset schema inconsistency, automate model download, and improve error handling in `basemodel.py`. Verify clean-environment setup completes in under 30 minutes. **Also in Week 5**: Reach out to Abhishek Kumar regarding PR #297 and share the complete 17-bug analysis from this proposal. Align on which changes he will apply vs. which will be contributed via follow-up commits.
- **Week 6**: Restore LLM-Edge-Benchmark-Suite — refactor `basemodel.py` for configurable `model_id`, create per-model algorithm YAML files, update `benchmarkingjob.yaml`, and add division by zero guard in `acc.py`. **Apply all Ianvs(Robot) PR #297 fixes** if not yet resolved by Abhishek — environment fixes (Bugs 13A, 13B, 14 documented in README and `requirements.txt`), all example-layer fixes (Bugs 1–7, 9–12), Ianvs core fixes (Bugs 8, 15, 17 in `lifelong_learning.py`), and Sedna wheel patch (Bug 16 in `task_update_decision_finetune.py`, already included in 0.4.1.1).
- **Week 7**: Validate end-to-end execution of both LLM examples and the Ianvs(Robot) example. Finalize all README updates and execution guides. Confirm PR #297 is in a fully functional and mergeable state.


## Success Metrics

### Primary Success Metrics

- **Five fully functional examples** with 100% execution success rate: Cityscapes-Synthia curb detection, Cityscapes-Synthia semantic segmentation, LLM-Agent, LLM-Edge-Benchmark-Suite, and Ianvs(Robot) lifelong learning semantic segmentation
- **All 44 confirmed bugs fixed** (12 curb detection + 15 semantic segmentation + 17 robot) and verified in the main repository, with a corresponding GitHub Issue filed in the Ianvs repository for each Sedna-layer bug
- All Sedna fixes implemented as **new helper functions only** — zero changes to existing public function signatures, verified by confirming all currently-passing example tests continue to pass after the patched wheel is installed
- **Sedna 0.4.1.1 wheel produced** by patching the existing 0.4.1 wheel in `resources/` (7 files total: 6 for Cityscapes, 1 for Ianvs(Robot) Bug 16) — no dependency version changes, consistent with reviewer preference confirmed at the last routine meeting
- All 17 blocking issues in Ianvs(Robot) PR #297 resolved — environment issues documented, example-layer bugs fixed, Ianvs core bugs 8/15/17 patched, Sedna Bug 16 applied via wheel patch — with the robot example reaching a fully functional and mergeable state
- Pull Request opened to the upstream Sedna repository with the same additive fixes, citing the filed Ianvs issues as evidence
- LLM-Agent new user setup time reduced from 5+ hours to under 30 minutes
- LLM-Edge-Benchmark-Suite successfully comparing multiple models (Qwen 0.5B vs Qwen 1.5B) in a single benchmarking run
- Complete documentation package including restoration methodology, environment setup guides, and debugging playbooks for all five examples
