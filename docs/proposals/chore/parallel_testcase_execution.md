# Parallel Test Case Execution

## Table of Contents
- [Motivation](#motivation)
  - [Background](#background)
  - [Goals](#goals)
    - [Basic Goals](#basic-goals)
    - [Advanced Goals](#advanced-goals)
- [Proposal](#proposal)
- [Details](#details)
  - [Phase 1: ThreadPoolExecutor-Based Parallelism](#phase-1-threadpoolexecutor-based-parallelism)
    - [Design](#design)
    - [Race Condition Fix](#race-condition-fix)
    - [Configuration](#configuration)
    - [Files Changed](#files-changed)
  - [Phase 2: Sandbox Engine (Future)](#phase-2-sandbox-engine-future)
  - [Comparison of Approaches](#comparison-of-approaches)
- [Roadmap](#roadmap)

---

## Motivation

### Background

Ianvs supports testing multiple groups of hyperparameters across
algorithm modules. Each test case runs a full training or inference
pipeline, which can take minutes to hours depending on the model
and dataset. When a user wants to compare several parameter
combinations (e.g., different learning rates, batch sizes, or
model architectures), these test cases execute sequentially in
`TestCaseController.run_testcases()`.

This serial execution causes unbearable time overhead as documented
in [Issue #8](https://github.com/kubeedge/ianvs/issues/8). For
example, testing 5 hyperparameter groups on a model that takes
30 minutes each requires 2.5 hours total — even if the machine
has sufficient CPU and memory to run multiple test cases
simultaneously.

The core sequential loop in `testcasecontroller.py`:

```python
for testcase in self.test_cases:
    res, time = (testcase.run(workspace), utils.get_local_time())
    succeed_results[testcase.id] = (res, time)
```

has no parallelism mechanism and no configuration option to
enable concurrent execution.

### Goals

#### Basic Goals

1. Enable opt-in parallel execution of test cases to reduce
   total benchmarking time when multiple parameter groups
   are tested simultaneously.

2. Maintain full backward compatibility — all existing
   `benchmarkingjob.yaml` files must continue to work
   without modification.

3. Fix an existing race condition in `_get_output_dir()`
   that would cause failures under any parallel execution.

4. Provide a simple, dependency-free solution that works
   on all platforms (Linux, macOS, Windows).

#### Advanced Goals

1. Provide a clear migration path to the more comprehensive
   Sandbox Engine proposed in PR [#526](https://github.com/kubeedge/ianvs/pull/526)
   for full process isolation and dependency management.

2. Allow users to configure the maximum number of parallel
   workers to control resource consumption.

---

## Proposal

We propose a **phased approach** to parallel test case execution:

**Phase 1 (This PR):** Add opt-in thread-based parallelism using
Python's `concurrent.futures.ThreadPoolExecutor`. This requires
no new dependencies, works on all platforms, and can be merged
immediately to address the core time overhead problem.

**Phase 2 (Future):** Migrate to the Sandbox Engine (PR #526)
for full process isolation, dependency conflict resolution, and
OS-level resource limits. Phase 1 serves as a stepping stone
that immediately benefits users while Phase 2 is developed.

The parallel flag defaults to `false` — zero impact on existing
users until they explicitly opt in.

---

## Details

### Phase 1: ThreadPoolExecutor-Based Parallelism

#### Design

Python's `ThreadPoolExecutor` is chosen over `ProcessPoolExecutor`
for a critical reason: ML model objects (PyTorch modules, TensorFlow
graphs, Sedna paradigm instances) loaded during `testcase.run()`
are often not picklable — a requirement for process-based
parallelism. Thread-based parallelism shares the same memory
space, eliminating serialization entirely.

The implementation adds two private methods to `TestCaseController`:

- `_run_testcases_sequential()` — preserves the exact original
  behavior, called when `parallel=False` (default)
- `_run_testcases_parallel()` — runs test cases concurrently
  using `ThreadPoolExecutor`, called when `parallel=True`

The public `run_testcases()` method routes to the appropriate
implementation based on the `parallel` parameter:

```python
def run_testcases(self, workspace, parallel=False, max_workers=None):
    if parallel:
        return self._run_testcases_parallel(workspace, max_workers)
    return self._run_testcases_sequential(workspace)
```

Failed test cases in parallel mode are collected and reported
together rather than stopping execution immediately — giving
users a complete picture of which parameter groups failed.

#### Race Condition Fix

The original `_get_output_dir()` implementation contains a
race condition that would cause silent failures under parallel
execution:

```python
# ORIGINAL — race condition
while flag:
    output_dir = os.path.join(workspace, self.algorithm.name, str(self.id))
    if not os.path.exists(output_dir):  # two threads can pass simultaneously
        flag = False
```

Two threads executing simultaneously can both pass the
`os.path.exists()` check before either creates the directory,
resulting in a collision. The fix uses `uuid1()` (already unique
per test case) with `os.makedirs(exist_ok=True)`:

```python
# FIXED — no race condition
def _get_output_dir(self, workspace):
    output_dir = os.path.join(workspace, self.algorithm.name, str(self.id))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
```

This fix is beneficial even for sequential execution — it
removes an unnecessary while loop and makes directory creation
atomic.

#### Configuration

Users opt in by adding two optional fields to
`benchmarkingjob.yaml`:

```yaml
benchmarkingjob:
  name: "benchmarkingjob"
  workspace: "./workspace"
  parallel: true      # optional, default: false
  max_workers: 4      # optional, default: number of test cases
  testenv: "..."
  test_object: ...
```

Both fields are optional and backward compatible. When
`parallel` is omitted or set to `false`, execution is
identical to the current behavior.

#### Files Changed

| File | Change |
|---|---|
| `core/testcasecontroller/testcasecontroller.py` | Add `_run_testcases_sequential()`, `_run_testcases_parallel()`, update `run_testcases()` signature |
| `core/testcasecontroller/testcase/testcase.py` | Fix race condition in `_get_output_dir()` |
| `core/cmd/obj/benchmarkingjob.py` | Add `parallel` and `max_workers` fields, pass to `run_testcases()` |

### Phase 2: Sandbox Engine (Future)

PR [#526](https://github.com/kubeedge/ianvs/pull/526) proposes
a comprehensive Sandbox Engine that addresses deeper problems
beyond basic parallelism:

- **Dependency isolation**: each test case runs in its own
  virtual environment, preventing conflicts between algorithms
  requiring different package versions
- **OOM protection**: OS-level resource limits via `prlimit`
  and `cgroups` (Linux) prevent a single test case from
  crashing the entire benchmarking job
- **Repository protection**: enforced relative path boundaries
  prevent algorithm code from polluting the Ianvs core

Phase 1 (this proposal) and Phase 2 (Sandbox Engine) are
complementary, not competing. Phase 1 solves the immediate
time overhead problem for the common case (same environment,
multiple hyperparameter groups). Phase 2 addresses the harder
problem of conflicting dependencies and resource isolation for
production edge deployments.

### Comparison of Approaches

| Aspect | Phase 1 (ThreadPoolExecutor) | Phase 2 (Sandbox Engine) |
|---|---|---|
| Implementation time | Hours | Months (Fall LFX term) |
| Platform support | All (Linux, macOS, Windows) | Linux primary |
| New dependencies | None | `uv`, `psutil` |
| ML model support | All models (no pickling needed) | Subprocess serialization required |
| Dependency isolation | No | Yes (separate venv per test case) |
| OOM protection | No | Yes (prlimit/cgroups) |
| Backward compatibility | Full (parallel: false default) | New config schema required |
| Merge readiness | Ready now | Design phase (PR #526) |
| Best for | Multiple hyperparameter groups, same environment | Algorithms with conflicting dependencies |

---

## Roadmap

### Phase 1 (This PR — Immediate)
- Fix race condition in `_get_output_dir()`
- Add `ThreadPoolExecutor`-based parallel execution
- Add `parallel` and `max_workers` config fields
- Full backward compatibility verified
- Works on Linux, macOS, Windows

### Phase 2 (Future — Fall LFX Term)
- Sandbox Engine implementation per PR #526
- Full process isolation with `uv`/`venv`
- OS-level resource limits (Linux)
- Cross-platform graceful degradation
- Migration guide from Phase 1 to Phase 2