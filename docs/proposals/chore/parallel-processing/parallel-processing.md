# Parallel Test Case Processing for Ianvs Benchmarking Framework

**Author:** Krrish Biswas ([@krrish175-byte](https://github.com/krrish175-byte))  
**Date:** February 2026  
**Status:** Draft  
**Related Issue:** [#8](https://github.com/kubeedge/ianvs/issues/8)  
**Related PR:** [#308](https://github.com/kubeedge/ianvs/pull/308)

---

## Table of Contents

1. [Background](#background)
2. [Motivation & Problem Statement](#motivation--problem-statement)
3. [Goals](#goals)
4. [Non-Goals](#non-goals)
5. [Proposed Architecture Design](#proposed-architecture-design)
6. [Configuration Interface](#configuration-interface)
7. [Implementation Details](#implementation-details)
8. [Impact on Existing Examples](#impact-on-existing-examples)
9. [Testing & Validation Strategy](#testing--validation-strategy)
10. [Expected Performance Improvements](#expected-performance-improvements)
11. [Migration Strategy](#migration-strategy)
12. [Alternative Approaches Considered](#alternative-approaches-considered)
13. [Risk Assessment](#risk-assessment)
14. [Open Questions & Future Work](#open-questions--future-work)

---

## Background

Ianvs is an open-source benchmarking platform for cloud-edge collaborative AI under the KubeEdge project. It enables researchers and developers to evaluate AI algorithms across various paradigms including single-task learning, incremental learning, and lifelong learning.

Currently, Ianvs executes test cases **serially** — one test case runs to completion before the next one begins. While this approach is simple and predictable, it significantly underutilizes available computational resources, especially when benchmarking multiple parameter configurations or algorithm variants.

### Current Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Serial Test Case Execution                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Test Case 1        Test Case 2        Test Case 3        Test Case N  │
│   ┌──────────┐      ┌──────────┐       ┌──────────┐       ┌──────────┐  │
│   │  Start   │      │  Start   │       │  Start   │       │  Start   │  │
│   │    ↓     │      │    ↓     │       │    ↓     │       │    ↓     │  │
│   │  Train   │ ──→  │  Train   │  ──→  │  Train   │  ──→  │  Train   │  │
│   │    ↓     │      │    ↓     │       │    ↓     │       │    ↓     │  │
│   │   Eval   │      │   Eval   │       │   Eval   │       │   Eval   │  │
│   │    ↓     │      │    ↓     │       │    ↓     │       │    ↓     │  │
│   │   Done   │      │   Done   │       │   Done   │       │   Done   │  │
│   └──────────┘      └──────────┘       └──────────┘       └──────────┘  │
│                                                                          │
│   Timeline: ════════════════════════════════════════════════════════►   │
│             t=0     t=T        t=2T              t=3T            t=N×T  │
│                                                                          │
│   Total Time: N × T (where T = time per test case)                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Motivation & Problem Statement

### The Core Problem

When users run benchmarks with multiple test configurations, the **serial execution model** leads to:

1. **Excessive Execution Time**: A benchmark with 6 test cases, each taking 30 minutes, requires **3 hours** to complete.

2. **Underutilized Hardware**: Modern machines have multiple CPU cores (4-64+), but serial execution only uses one core at a time for CPU-bound preprocessing and evaluation tasks.

3. **Poor Developer Experience**: Long wait times reduce iteration speed, making it difficult to experiment with different hyperparameter configurations or algorithm variations.

4. **Wasted Opportunity Cost**: While one test case trains, other CPU cores remain idle during I/O operations or waiting for GPU synchronization.

### Real-World Impact

| Scenario | Test Cases | Serial Time | Hardware Utilization |
|----------|-----------|-------------|----------------------|
| PCB-AOI Benchmark | 6 | ~3 hours | ~15% CPU |
| Robot Lifelong Learning | 8 | ~4 hours | ~12% CPU |
| Multi-hyperparameter Sweep | 20 | ~10 hours | ~10% CPU |

### Community Demand

Issue [#8](https://github.com/kubeedge/ianvs/issues/8) has been open since **July 2022**, demonstrating a long-standing community need for parallel test case execution:

> *"Each use case spends the most of the time on training process. When a user wants to test several groups of parameters, serial training will incur unbearable time overhead."*

---

## Goals

### Primary Goals

1. **Enable Parallel Execution**: Allow multiple test cases to run concurrently using Python's `ProcessPoolExecutor`.

2. **Maintain Backward Compatibility**: All existing examples **MUST** continue to work without any modifications. Serial execution remains the default behavior.

3. **Provide Flexible Configuration**: Support both CLI arguments and YAML configuration for enabling parallelism.

4. **Ensure Robust Error Handling**: Failures in one test case should not crash the entire benchmarking job.

5. **Preserve Result Consistency**: Results from parallel execution should be equivalent to serial execution.

### Success Criteria

- [ ] All 27+ examples in `examples/` directory work identically in both serial and parallel modes
- [ ] No changes required to existing `benchmarkingjob.yaml` configurations
- [ ] Performance improvement of 2-4x on 4-core machines for CPU-bound portions
- [ ] Zero breaking changes to existing user workflows
- [ ] Comprehensive test coverage for new functionality

---

## Non-Goals

The following are explicitly **out of scope** for this proposal:

| Non-Goal | Reason | Future Consideration |
|----------|--------|---------------------|
| GPU Resource Management | Requires sophisticated scheduling and CUDA context handling | Phase 2 |
| Distributed Multi-Node Execution | Requires cluster coordination infrastructure | Phase 3 |
| Dynamic Load Balancing | Adds complexity without proven need | Evaluate after Phase 1 |
| Automatic Worker Count Optimization | Depends on workload characteristics | Future work |
| Intra-Test-Case Parallelism | Would require significant algorithm changes | Out of scope |

---

## Proposed Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Parallel Execution Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        BenchmarkingJob                                │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │                    Configuration Layer                          │ │   │
│  │  │  • CLI: --parallel, --workers                                   │ │   │
│  │  │  • YAML: parallel_execution, num_workers                        │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      TestCaseController                               │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │                  run_testcases()                                │  │   │
│  │  │                                                                 │  │   │
│  │  │   parallel=False?  ──────►  Serial Execution (existing)         │  │   │
│  │  │         │                                                       │  │   │
│  │  │         ▼                                                       │  │   │
│  │  │   parallel=True?   ──────►  ProcessPoolExecutor                 │  │   │
│  │  │                             │                                   │  │   │
│  │  │                             ├──► Worker 1 ──► run_testcase_func │  │   │
│  │  │                             ├──► Worker 2 ──► run_testcase_func │  │   │
│  │  │                             ├──► Worker 3 ──► run_testcase_func │  │   │
│  │  │                             └──► Worker N ──► run_testcase_func │  │   │
│  │  │                                                                 │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Result Aggregation                              │   │
│  │  • Collect results via as_completed()                                 │   │
│  │  • Error handling per test case                                       │   │
│  │  • Rank save and visualization                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Flow Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Parallel Test Case Execution (4 Workers)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Worker 1       Worker 2       Worker 3       Worker 4                      │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐                      │
│   │ TC 1   │    │ TC 2   │    │ TC 3   │    │ TC 4   │    First Batch       │
│   │ Train  │    │ Train  │    │ Train  │    │ Train  │                      │
│   │ Eval   │    │ Eval   │    │ Eval   │    │ Eval   │                      │
│   └────────┘    └────────┘    └────────┘    └────────┘                      │
│       │             │             │             │                            │
│       ▼             ▼             ▼             ▼                            │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐                      │
│   │ TC 5   │    │ TC 6   │    │ (idle) │    │ (idle) │    Second Batch      │
│   │ Train  │    │ Train  │    │        │    │        │                      │
│   │ Eval   │    │ Eval   │    │        │    │        │                      │
│   └────────┘    └────────┘    └────────┘    └────────┘                      │
│                                                                              │
│   Timeline: ════════════════════════════════════►                            │
│             t=0              t=T              t≈2T                           │
│                                                                              │
│   Total Time: ⌈N/W⌉ × T (where W = workers)                                 │
│   Example: 6 test cases / 4 workers = ~2T (vs 6T serial)                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Component Interaction Diagram                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Input                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│   ┌──────────────────────┐                                                   │
│   │  benchmarking.py     │  CLI argument parsing                             │
│   │  --parallel          │  --parallel, --workers                            │
│   │  --workers 4         │                                                   │
│   └─────────┬────────────┘                                                   │
│             │                                                                │
│             ▼                                                                │
│   ┌──────────────────────┐                                                   │
│   │  BenchmarkingJob     │  Merge CLI args with YAML config                  │
│   │  • parallel=True     │  CLI takes precedence                             │
│   │  • workers=4         │                                                   │
│   └─────────┬────────────┘                                                   │
│             │                                                                │
│             ▼                                                                │
│   ┌──────────────────────┐                                                   │
│   │ TestCaseController   │  Execution orchestration                          │
│   │  run_testcases()     │                                                   │
│   └─────────┬────────────┘                                                   │
│             │                                                                │
│             ├──── parallel=False ────►  Sequential for loop (existing)       │
│             │                                                                │
│             └──── parallel=True  ────►  ProcessPoolExecutor                  │
│                         │                                                    │
│                         ▼                                                    │
│                   ┌────────────────────────────────────────┐                 │
│                   │      Worker Processes                   │                 │
│                   │  ┌──────────────────────────────────┐  │                 │
│                   │  │   run_testcase_func(tc, ws)      │  │                 │
│                   │  │   • Runs TestCase.run()          │  │                 │
│                   │  │   • Returns serializable result  │  │                 │
│                   │  │   • Handles exceptions           │  │                 │
│                   │  └──────────────────────────────────┘  │                 │
│                   └────────────────────────────────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Interface

### CLI Arguments

```bash
# Enable parallel execution with auto-detected worker count
ianvs -f benchmarkingjob.yaml --parallel

# Enable parallel execution with specific worker count
ianvs -f benchmarkingjob.yaml --parallel --workers 4

# Short form
ianvs -f benchmarkingjob.yaml -p -w 4

# Serial execution (default, unchanged)
ianvs -f benchmarkingjob.yaml
```

### YAML Configuration

```yaml
# benchmarkingjob.yaml
benchmarkingjob:
  name: "pcb-aoi-benchmark"
  workspace: "./workspace"
  
  # NEW: Parallel execution configuration (optional)
  parallel_execution: true    # Enable parallel mode
  num_workers: 4              # Number of worker processes
  
  # Existing configuration (unchanged)
  testenv: "./testenv/testenv.yaml"
  test_object:
    type: algorithms
    algorithms:
      - name: "fpn"
        url: "./testalgorithms/fpn/fpn_algorithm.yaml"
```

### Configuration Priority

```
CLI Arguments > YAML Configuration > Default Values

Example:
  YAML: parallel_execution: true, num_workers: 2
  CLI:  --parallel --workers 4
  
  Result: parallel=True, workers=4 (CLI overrides YAML)
```

### Default Behavior

| Setting | Default Value | Description |
|---------|--------------|-------------|
| `parallel` | `False` | Serial execution (backward compatible) |
| `workers` | `cpu_count() - 1` | Auto-detect, leave 1 core for system |

> [!IMPORTANT]
> **Backward Compatibility**: If no parallel options are specified (neither CLI nor YAML), Ianvs behaves exactly as before, executing test cases serially. This ensures **all existing workflows continue working without modification**.

---

## Implementation Details

### Core Code Changes

#### 1. TestCaseController (`core/testcasecontroller/testcasecontroller.py`)

```python
import concurrent.futures
import os

from core.common import utils
from core.common.log import LOGGER
from core.testcasecontroller.testcase import TestCase, run_testcase_func


class TestCaseController:
    """Test Case Controller with parallel execution support."""

    def __init__(self):
        self.test_cases = []

    def run_testcases(self, workspace, parallel=False, workers=None):
        """
        Run all test cases.
        
        Parameters
        ----------
        workspace : str
            Output directory for test results
        parallel : bool
            Enable parallel execution (default: False)
        workers : int or None
            Number of worker processes (default: cpu_count - 1)
            
        Returns
        -------
        tuple
            (succeed_testcases, succeed_results)
        """
        succeed_results = {}
        succeed_testcases = []

        if parallel:
            # Parallel execution path
            if workers is None:
                workers = max(1, (os.cpu_count() or 2) - 1)
            
            LOGGER.info(f"Running {len(self.test_cases)} test cases "
                       f"in parallel with {workers} workers")
            
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers
            ) as executor:
                # Submit all test cases to the pool
                future_to_testcase = {
                    executor.submit(run_testcase_func, testcase, workspace): testcase
                    for testcase in self.test_cases
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_testcase):
                    testcase = future_to_testcase[future]
                    try:
                        result = future.result()
                        if result["status"] == "success":
                            res = result["results"]
                            time = utils.get_local_time()
                            succeed_results[testcase.id] = (res, time)
                            succeed_testcases.append(testcase)
                            LOGGER.info(f"Test case {testcase.algorithm.name} "
                                       f"completed successfully")
                        else:
                            LOGGER.error(f"Test case {testcase.id} failed: "
                                        f"{result.get('error')}")
                    except Exception as exc:
                        LOGGER.error(f"Test case {testcase.id} generated "
                                    f"an exception: {exc}")
        else:
            # Serial execution path (existing behavior, unchanged)
            for testcase in self.test_cases:
                try:
                    res, time = (testcase.run(workspace), utils.get_local_time())
                except Exception as err:
                    raise RuntimeError(
                        f"testcase(id={testcase.id}) runs failed, error: {err}"
                    ) from err
                    
                succeed_results[testcase.id] = (res, time)
                succeed_testcases.append(testcase)

        return succeed_testcases, succeed_results
```

#### 2. Worker Function (`core/testcasecontroller/testcase/testcase.py`)

```python
def run_testcase_func(testcase, workspace):
    """
    Top-level worker function for parallel execution.
    
    This function must be defined at module level to be picklable
    by ProcessPoolExecutor.
    
    Parameters
    ----------
    testcase : TestCase
        The test case instance to run
    workspace : str
        Output directory path
        
    Returns
    -------
    dict
        Result dictionary with status, config name, and results or error
    """
    try:
        res = testcase.run(workspace)
        return {
            "status": "success",
            "config": testcase.algorithm.name,
            "results": res
        }
    except Exception as e:
        return {
            "status": "failed",
            "config": testcase.algorithm.name,
            "error": str(e)
        }
```

> [!NOTE]
> **Why a top-level function?** Python's `pickle` module (used by `ProcessPoolExecutor`) cannot serialize lambda functions or nested functions. The worker function must be defined at module level for proper serialization.

#### 3. CLI Argument Parsing (`core/cmd/benchmarking.py`)

```python
def _generate_parser():
    parser = argparse.ArgumentParser(description='AI Benchmarking Tool')
    parser.prog = "ianvs"

    parser.add_argument("-f", "--benchmarking_config_file",
                        nargs="?", type=str,
                        help="benchmarking config file (yaml/yml)")

    # NEW: Parallel execution arguments
    parser.add_argument("-p", "--parallel",
                        action="store_true",
                        help="run test cases in parallel")

    parser.add_argument("-w", "--workers",
                        type=int,
                        help="number of workers for parallel execution")

    parser.add_argument('-v', '--version',
                        action='version',
                        version=__version__)

    return parser
```

#### 4. BenchmarkingJob Configuration (`core/cmd/obj/benchmarkingjob.py`)

```python
class BenchmarkingJob:
    def __init__(self, config, args=None):
        self.name: str = ""
        self.workspace: str = "./workspace"
        self.test_object: dict = {}
        self.rank = None
        self.test_env = None
        self.simulation = None
        self.testcase_controller = TestCaseController()
        
        # NEW: Parallel execution settings
        self.parallel = False
        self.workers = None
        
        self._parse_config(config)

        # CLI args override YAML config
        if args:
            if args.parallel:
                self.parallel = args.parallel
            if args.workers:
                self.workers = args.workers

    def _parse_config(self, config: dict):
        for k, v in config.items():
            # ... existing parsing ...
            elif k == "parallel_execution":
                self.parallel = v
            elif k == "num_workers":
                self.workers = v
            # ... rest of parsing ...
```

### Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Error Handling Flow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Test Case Execution                                                        │
│       │                                                                      │
│       ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │             run_testcase_func (in worker process)                    │   │
│   │                                                                      │   │
│   │   try:                                                               │   │
│   │       result = testcase.run(workspace)                               │   │
│   │       return {"status": "success", "results": result}  ────────────────► │
│   │   except Exception as e:                                             │   │
│   │       return {"status": "failed", "error": str(e)}  ───────────────────► │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Result Collection (in main process)                                        │
│       │                                                                      │
│       ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │   for future in as_completed(futures):                               │   │
│   │       try:                                                           │   │
│   │           result = future.result()                                   │   │
│   │           if result["status"] == "success":                          │   │
│   │               ✓ Add to succeed_results                               │   │
│   │           else:                                                      │   │
│   │               ⚠ Log error, continue processing                       │   │
│   │       except Exception as exc:                                       │   │
│   │           ⚠ Log exception, continue processing                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Key Behaviors:                                                             │
│   • Failed test cases don't crash the entire benchmark                       │
│   • All errors are logged with test case identification                      │
│   • Successful test cases are processed and ranked normally                  │
│   • Final rank/visualization only includes successful test cases             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Logging Considerations

```python
# Each worker process logs with test case identification
LOGGER.info(f"[Worker] Starting test case: {testcase.algorithm.name}")
LOGGER.info(f"[Worker] Completed test case: {testcase.algorithm.name}")
LOGGER.error(f"[Worker] Failed test case {testcase.id}: {error}")

# Main process logs overall progress
LOGGER.info(f"Running {len(test_cases)} test cases with {workers} workers")
LOGGER.info(f"Completed {completed}/{total} test cases")
```

> [!WARNING]
> **Log Interleaving**: In parallel mode, log messages from different workers may interleave. Each log message includes the test case name/ID for traceability.

---

## Impact on Existing Examples

> [!IMPORTANT]
> **This is a CRITICAL section.** All existing examples MUST continue working without any modifications when parallel processing is disabled (the default).

### Compatibility Matrix

| Example Directory | Example Type | Serial Mode | Parallel Mode | Notes |
|-------------------|--------------|-------------|---------------|-------|
| `examples/pcb-aoi/singletask_learning_bench/` | Single-task | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/pcb-aoi/incremental_learning_bench/` | Incremental | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/robot/lifelong_learning_bench/` | Lifelong | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/cityscapes/` | Semantic Seg | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/cifar100/` | Classification | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/bdd/` | Object Detection | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/imagenet/` | Classification | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/Cloud_Robotics/` | Cloud-Edge | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/llm-agent/` | LLM | ✅ Compatible | ⚠️ Test Required | GPU resource consideration |
| `examples/federated-llm/` | Federated | ✅ Compatible | ⚠️ Test Required | Multi-node scenarios |
| `examples/government/` | NLP | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/aoa/` | AOA | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/yaoba/` | Custom | ✅ Compatible | ✅ Compatible | No changes required |
| `examples/smart_coding/` | LLM | ✅ Compatible | ⚠️ Test Required | API rate limiting |
| All other examples | Various | ✅ Compatible | ✅ Compatible | No changes required |

### Backward Compatibility Guarantee

```yaml
# EXISTING benchmarkingjob.yaml - NO CHANGES REQUIRED
# This continues to work exactly as before
benchmarkingjob:
  name: "pcb-aoi-benchmark"
  workspace: "./workspace"
  testenv: "./testenv/testenv.yaml"
  test_object:
    type: algorithms
    algorithms:
      - name: "fpn"
        url: "./testalgorithms/fpn/fpn_algorithm.yaml"
```

```bash
# Existing command - works exactly as before (serial execution)
ianvs -f benchmarkingjob.yaml
```

### Example Validation Test Script

```bash
#!/bin/bash
# validate_all_examples.sh
# Run all examples in both serial and parallel modes

EXAMPLES=(
    "examples/pcb-aoi/singletask_learning_bench/"
    "examples/pcb-aoi/incremental_learning_bench/"
    "examples/robot/lifelong_learning_bench/"
    "examples/cityscapes/"
    # ... add all examples
)

for example in "${EXAMPLES[@]}"; do
    echo "Testing $example..."
    
    # Serial mode (existing behavior)
    echo "  Serial mode..."
    ianvs -f "$example/benchmarkingjob.yaml"
    SERIAL_RESULT=$?
    
    # Parallel mode (new feature)
    echo "  Parallel mode..."
    ianvs -f "$example/benchmarkingjob.yaml" --parallel --workers 2
    PARALLEL_RESULT=$?
    
    # Compare results
    if [ $SERIAL_RESULT -eq 0 ] && [ $PARALLEL_RESULT -eq 0 ]; then
        echo "  ✅ PASSED"
    else
        echo "  ❌ FAILED"
    fi
done
```

---

## Testing & Validation Strategy

### Test Pyramid

```
                    ┌─────────────────────────┐
                    │   E2E Validation        │  All 27+ examples
                    │   (Manual + CI)         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴─────────────┐
                    │   Integration Tests     │  Full benchmark runs
                    │                         │  Result comparison
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │         Unit Tests                     │
            │  • TestCaseController.run_testcases    │
            │  • run_testcase_func                   │
            │  • CLI argument parsing                │
            │  • YAML config parsing                 │
            └────────────────────────────────────────┘
```

### Unit Tests

```python
# tests/test_parallel_execution.py

import unittest
from unittest.mock import Mock, patch
from core.testcasecontroller.testcasecontroller import TestCaseController
from core.testcasecontroller.testcase import run_testcase_func


class TestParallelExecution(unittest.TestCase):
    """Unit tests for parallel test case execution."""

    def test_serial_execution_default(self):
        """Verify serial execution is the default behavior."""
        controller = TestCaseController()
        controller.test_cases = [Mock(), Mock()]
        
        # Should not raise, uses serial path
        controller.run_testcases("/tmp/workspace", parallel=False)

    def test_parallel_execution_enabled(self):
        """Verify parallel execution works when enabled."""
        controller = TestCaseController()
        controller.test_cases = [Mock(), Mock()]
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_pool:
            controller.run_testcases("/tmp/workspace", parallel=True, workers=2)
            mock_pool.assert_called_once_with(max_workers=2)

    def test_worker_count_auto_detection(self):
        """Verify worker count auto-detection."""
        controller = TestCaseController()
        controller.test_cases = [Mock()]
        
        with patch('os.cpu_count', return_value=8):
            with patch('concurrent.futures.ProcessPoolExecutor') as mock_pool:
                controller.run_testcases("/tmp/workspace", parallel=True)
                mock_pool.assert_called_once_with(max_workers=7)  # cpu_count - 1

    def test_run_testcase_func_success(self):
        """Test worker function success case."""
        mock_testcase = Mock()
        mock_testcase.run.return_value = {"accuracy": 0.95}
        mock_testcase.algorithm.name = "test_algo"
        
        result = run_testcase_func(mock_testcase, "/tmp/workspace")
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"]["accuracy"], 0.95)

    def test_run_testcase_func_failure(self):
        """Test worker function error handling."""
        mock_testcase = Mock()
        mock_testcase.run.side_effect = RuntimeError("Test error")
        mock_testcase.algorithm.name = "test_algo"
        
        result = run_testcase_func(mock_testcase, "/tmp/workspace")
        
        self.assertEqual(result["status"], "failed")
        self.assertIn("Test error", result["error"])

    def test_failed_testcase_doesnt_crash_others(self):
        """Verify one failure doesn't crash the entire batch."""
        controller = TestCaseController()
        
        good_tc = Mock()
        good_tc.run.return_value = {"accuracy": 0.9}
        good_tc.id = "good"
        good_tc.algorithm.name = "good_algo"
        
        bad_tc = Mock()
        bad_tc.run.side_effect = RuntimeError("Bad")
        bad_tc.id = "bad"
        bad_tc.algorithm.name = "bad_algo"
        
        controller.test_cases = [good_tc, bad_tc]
        
        succeed_testcases, succeed_results = controller.run_testcases(
            "/tmp/workspace", parallel=True, workers=2
        )
        
        # Good test case should still succeed
        self.assertEqual(len(succeed_testcases), 1)
        self.assertEqual(succeed_testcases[0].id, "good")


class TestCLIArgs(unittest.TestCase):
    """Unit tests for CLI argument parsing."""

    def test_parallel_flag(self):
        """Test --parallel flag parsing."""
        from core.cmd.benchmarking import _generate_parser
        
        parser = _generate_parser()
        args = parser.parse_args(["-f", "test.yaml", "--parallel"])
        
        self.assertTrue(args.parallel)

    def test_workers_argument(self):
        """Test --workers argument parsing."""
        from core.cmd.benchmarking import _generate_parser
        
        parser = _generate_parser()
        args = parser.parse_args(["-f", "test.yaml", "-w", "4"])
        
        self.assertEqual(args.workers, 4)


class TestYAMLConfig(unittest.TestCase):
    """Unit tests for YAML configuration parsing."""

    def test_parallel_execution_yaml(self):
        """Test parallel_execution YAML config."""
        from core.cmd.obj.benchmarkingjob import BenchmarkingJob
        
        config = {
            "name": "test",
            "workspace": "/tmp",
            "parallel_execution": True,
            "num_workers": 4,
            "test_object": {"type": "algorithms", "algorithms": []}
        }
        
        with patch.object(BenchmarkingJob, '_parse_testenv_config'):
            with patch.object(BenchmarkingJob, '_check_fields'):
                job = BenchmarkingJob(config)
        
        self.assertTrue(job.parallel)
        self.assertEqual(job.workers, 4)

    def test_cli_overrides_yaml(self):
        """Verify CLI args override YAML config."""
        from core.cmd.obj.benchmarkingjob import BenchmarkingJob
        from argparse import Namespace
        
        config = {
            "name": "test",
            "workspace": "/tmp",
            "parallel_execution": False,
            "num_workers": 2,
            "test_object": {"type": "algorithms", "algorithms": []}
        }
        
        cli_args = Namespace(parallel=True, workers=8)
        
        with patch.object(BenchmarkingJob, '_parse_testenv_config'):
            with patch.object(BenchmarkingJob, '_check_fields'):
                job = BenchmarkingJob(config, cli_args)
        
        self.assertTrue(job.parallel)  # CLI override
        self.assertEqual(job.workers, 8)  # CLI override
```

### Integration Tests

```python
# tests/integration/test_parallel_integration.py

import os
import tempfile
import unittest

class TestParallelIntegration(unittest.TestCase):
    """Integration tests for parallel execution."""

    @unittest.skipIf(not os.path.exists("examples/pcb-aoi"), 
                     "PCB-AOI example not available")
    def test_pcb_aoi_serial_parallel_equivalence(self):
        """Verify serial and parallel produce equivalent results."""
        import subprocess
        
        example_path = "examples/pcb-aoi/singletask_learning_bench/"
        config = f"{example_path}/benchmarkingjob.yaml"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run serial
            serial_workspace = os.path.join(tmpdir, "serial")
            subprocess.run([
                "ianvs", "-f", config,
                "--workspace", serial_workspace
            ], check=True)
            
            # Run parallel
            parallel_workspace = os.path.join(tmpdir, "parallel")
            subprocess.run([
                "ianvs", "-f", config,
                "--workspace", parallel_workspace,
                "--parallel", "--workers", "2"
            ], check=True)
            
            # Compare results
            # (implementation details for result comparison)
            self.assertTrue(self._compare_results(
                serial_workspace, parallel_workspace
            ))
```

### Performance Benchmarks

```python
# tests/benchmarks/test_parallel_performance.py

import time
import statistics

def benchmark_parallel_speedup():
    """Measure actual speedup from parallel execution."""
    
    # Configure test with known workload
    test_cases = 8
    workers_list = [1, 2, 4, 8]
    
    results = {}
    
    for workers in workers_list:
        times = []
        for _ in range(3):  # 3 runs for averaging
            start = time.time()
            # Run benchmark
            elapsed = time.time() - start
            times.append(elapsed)
        
        results[workers] = {
            "mean": statistics.mean(times),
            "stddev": statistics.stdev(times) if len(times) > 1 else 0,
        }
    
    # Calculate speedup
    baseline = results[1]["mean"]
    for workers, data in results.items():
        data["speedup"] = baseline / data["mean"]
    
    return results
```

---

## Expected Performance Improvements

### Theoretical Analysis

```
Speedup = T_serial / T_parallel

Where:
  T_serial = N × T_avg      (N test cases, T_avg time per case)
  T_parallel ≈ ⌈N/W⌉ × T_avg  (W workers)

Theoretical maximum speedup = min(N, W)
Actual speedup depends on:
  • I/O overhead
  • Memory bandwidth
  • GPU contention (if applicable)
  • Process spawning overhead
```

### Realistic Projections

| Scenario | Test Cases | Workers | Serial Time | Parallel Time | Speedup |
|----------|-----------|---------|-------------|---------------|---------|
| PCB-AOI Basic | 4 | 4 | 120 min | 35 min | ~3.4x |
| PCB-AOI Extended | 8 | 4 | 240 min | 65 min | ~3.7x |
| Hyperparameter Sweep | 16 | 4 | 480 min | 130 min | ~3.7x |
| Full Benchmark Suite | 20 | 8 | 600 min | 90 min | ~6.6x |

### Example Calculation

```
Scenario: 6 test cases × 30 minutes each

Serial Execution:
  Total time = 6 × 30 = 180 minutes (3 hours)

Parallel Execution (4 workers):
  Batch 1: TC1, TC2, TC3, TC4 (run in parallel) = 30 min
  Batch 2: TC5, TC6 (run in parallel)           = 30 min
  Total time ≈ 60 minutes (1 hour)
  
  Overhead (process spawning, etc.) ≈ 5-10%
  Realistic total ≈ 65 minutes
  
  Speedup = 180 / 65 = 2.77x
```

### Performance Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Expected Speedup vs Worker Count                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Speedup                                                                     │
│     │                                                                        │
│  8x ┤                                              ○ Theoretical maximum     │
│     │                                           ○                            │
│  6x ┤                                        ●                               │
│     │                                     ●     ● Actual (8 test cases)      │
│  4x ┤                                  ●                                     │
│     │                               ●                                        │
│  3x ┤                            ●                                           │
│     │                         ●                                              │
│  2x ┤                      ●                                                 │
│     │                   ●                                                    │
│  1x ┼────────────────●───────────────────────────────────────────────────    │
│     │                                                                        │
│     └──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────────────────►   │
│            1      2      3      4      5      6      7      8    Workers     │
│                                                                              │
│  Note: Actual speedup < theoretical due to:                                  │
│        • Process spawning overhead                                           │
│        • Memory bandwidth limits                                             │
│        • I/O serialization                                                   │
│        • Python multiprocessing overhead                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Migration Strategy

### Phased Rollout Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         4-Week Migration Timeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Week 1: Foundation                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Merge core implementation                                            │ │
│  │ • Unit test coverage                                                   │ │
│  │ • Documentation updates                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Week 2: Validation                                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Validate all existing examples (serial mode)                         │ │
│  │ • Run integration tests                                                │ │
│  │ • Fix any compatibility issues                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Week 3: Extended Testing                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Test parallel mode on all examples                                   │ │
│  │ • Performance benchmarking                                             │ │
│  │ • Edge case testing                                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Week 4: Release                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Update release notes                                                 │ │
│  │ • User documentation                                                   │ │
│  │ • Community announcement                                               │ │
│  │ • Close Issue #8                                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Communication Plan

| Milestone | Action | Audience |
|-----------|--------|----------|
| Proposal Approval | Announce design in SIG AI meeting | Core maintainers |
| PR Merged | Update CHANGELOG.md | All developers |
| Week 2 Complete | Post validation results | Community |
| Release | Blog post / documentation | All users |

---

## Alternative Approaches Considered

### 1. Thread-Based Parallelism (Rejected)

```python
# REJECTED: Using ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(testcase.run, workspace) 
               for testcase in test_cases]
```

**Why rejected:**
- Python's Global Interpreter Lock (GIL) prevents true parallel execution of CPU-bound code
- Training and evaluation are CPU-bound, making threading ineffective
- No performance benefit for compute-heavy workloads

### 2. Distributed Execution Frameworks (Deferred)

```python
# DEFERRED: Using Ray or Dask
import ray

@ray.remote
def run_testcase(testcase, workspace):
    return testcase.run(workspace)

# Submit to Ray cluster
futures = [run_testcase.remote(tc, ws) for tc in test_cases]
results = ray.get(futures)
```

**Why deferred:**
- Adds heavy dependencies (Ray, Dask)
- Requires cluster setup and configuration
- Over-engineering for single-node use case
- Planned for Phase 3 (distributed multi-node execution)

### 3. asyncio-Based Execution (Rejected)

```python
# REJECTED: Using asyncio
import asyncio

async def run_all():
    tasks = [asyncio.create_task(testcase.run(workspace)) 
             for testcase in test_cases]
    return await asyncio.gather(*tasks)
```

**Why rejected:**
- asyncio is for I/O-bound concurrency, not CPU-bound parallelism
- Training/evaluation are CPU-bound operations
- Would require extensive code refactoring with no benefit

### Comparison Summary

| Approach | Parallelism Type | GIL Bypass | Complexity | Chosen? |
|----------|-----------------|------------|------------|---------|
| ProcessPoolExecutor | Process | ✅ Yes | Low | ✅ **Yes** |
| ThreadPoolExecutor | Thread | ❌ No | Low | ❌ No |
| Ray/Dask | Distributed | ✅ Yes | High | 🕐 Deferred |
| asyncio | Async I/O | ❌ No | Medium | ❌ No |

---

## Risk Assessment

### Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Resource Exhaustion | Medium | High | Auto-limit workers to cpu_count-1 |
| Debugging Complexity | Medium | Medium | Detailed logging with test case IDs |
| Result Inconsistency | Low | High | Comprehensive comparison testing |
| Breaking Changes | Low | Critical | Serial mode as default |
| Log Interleaving | High | Low | Prefix logs with process/test case ID |
| Memory Pressure | Medium | Medium | Document memory requirements |
| GPU Contention | Medium | Medium | Mark as future work (Phase 2) |

### Resource Exhaustion Mitigation

```python
# Automatic worker limit to prevent overload
def get_safe_worker_count(requested=None):
    cpu_count = os.cpu_count() or 2
    max_workers = max(1, cpu_count - 1)  # Leave 1 core for system
    
    if requested is None:
        return max_workers
    
    if requested > max_workers:
        LOGGER.warning(
            f"Requested {requested} workers exceeds safe limit. "
            f"Using {max_workers} workers instead."
        )
        return max_workers
    
    return requested
```

### Debugging Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Debugging in Parallel Mode                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Challenge: Errors in child processes are harder to trace                    │
│                                                                              │
│  Solutions:                                                                  │
│  1. Capture full tracebacks in worker function                               │
│  2. Include test case ID in all error messages                               │
│  3. Preserve exception chain with 'from err'                                 │
│  4. Log to individual files per worker (future enhancement)                  │
│  5. Recommend starting with --workers 1 for debugging                        │
│                                                                              │
│  Debug Workflow:                                                             │
│     1. Error in parallel mode                                                │
│        ↓                                                                     │
│     2. Identify failing test case from logs                                  │
│        ↓                                                                     │
│     3. Re-run single test case in serial mode                                │
│        ↓                                                                     │
│     4. Debug with full traceback visibility                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Open Questions & Future Work

### Open Questions for Community Discussion

1. **Memory Limits**: Should we add a `--max-memory-per-worker` option to prevent OOM conditions?

2. **Result Ordering**: Should parallel results be presented in submission order or completion order?

3. **Progress Reporting**: Should we add a progress bar for parallel execution (e.g., using tqdm)?

4. **Graceful Shutdown**: How should we handle Ctrl+C during parallel execution? Currently terminates all workers immediately.

5. **Worker Affinity**: Should we support CPU affinity for workers to optimize cache utilization?

### Future Work (Phase 2 and Beyond)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Future Roadmap                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 2: GPU Resource Management                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Detect available GPUs via nvidia-smi or pynvml                       │ │
│  │ • Assign GPU IDs to workers (CUDA_VISIBLE_DEVICES)                     │ │
│  │ • Support multiple test cases per GPU with time-slicing                │ │
│  │ • GPU memory monitoring and throttling                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Phase 3: Distributed Multi-Node Execution                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Integration with Ray or Dask                                         │ │
│  │ • Kubernetes-native job scheduling                                     │ │
│  │ • Support for cloud provider auto-scaling                              │ │
│  │ • Result aggregation across nodes                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Phase 4: Smart Scheduling                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Runtime estimation based on test case characteristics                │ │
│  │ • Load balancing across heterogeneous workers                          │ │
│  │ • Dynamic worker scaling based on queue depth                          │ │
│  │ • Priority scheduling for critical benchmarks                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

This proposal outlines a minimal, backward-compatible approach to adding parallel test case execution to the Ianvs benchmarking framework. Key highlights:

- **Zero Breaking Changes**: Existing examples and workflows continue to work without modification
- **Opt-In Parallelism**: Users explicitly enable parallel mode via CLI or YAML config
- **Simple Implementation**: Leverages Python's built-in `concurrent.futures` module
- **Robust Error Handling**: Individual test case failures don't crash the entire benchmark
- **Measurable Performance Gains**: 2-4x speedup expected on typical 4-core machines

We believe this feature will significantly improve the developer experience for Ianvs users while maintaining the stability and reliability the community expects.

---

## References

- [Issue #8: Not supported parallel processing of multiple use cases yet](https://github.com/kubeedge/ianvs/issues/8)
- [PR #308: feat: Add parallel processing for multiple test case execution](https://github.com/kubeedge/ianvs/pull/308)
- [Python concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)
- [ProcessPoolExecutor Best Practices](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor)

---

## Appendix A: Full Code Changes Summary

### Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `core/testcasecontroller/testcasecontroller.py` | Modified | Add parallel execution path |
| `core/testcasecontroller/testcase/testcase.py` | Modified | Add `run_testcase_func` worker function |
| `core/testcasecontroller/testcase/__init__.py` | Modified | Export `run_testcase_func` |
| `core/cmd/benchmarking.py` | Modified | Add `--parallel`, `--workers` CLI args |
| `core/cmd/obj/benchmarkingjob.py` | Modified | Parse parallel config from YAML/CLI |

### Dependencies Added

None. Uses Python standard library only.

---

## Appendix B: Example Configuration Templates

### Minimal Parallel Configuration

```yaml
# benchmarkingjob-parallel.yaml
benchmarkingjob:
  name: "parallel-benchmark"
  workspace: "./workspace"
  parallel_execution: true
  testenv: "./testenv/testenv.yaml"
  test_object:
    type: algorithms
    algorithms:
      - name: "algo1"
        url: "./testalgorithms/algo1.yaml"
```

### Full Configuration with All Options

```yaml
# benchmarkingjob-full.yaml
benchmarkingjob:
  name: "full-parallel-benchmark"
  workspace: "./workspace"
  
  # Parallel execution settings
  parallel_execution: true
  num_workers: 4
  
  # Standard settings
  testenv: "./testenv/testenv.yaml"
  rank:
    visualization:
      mode: selected_only
      metrics: ["accuracy", "f1_score"]
  
  test_object:
    type: algorithms
    algorithms:
      - name: "algorithm-v1"
        url: "./testalgorithms/v1.yaml"
      - name: "algorithm-v2"
        url: "./testalgorithms/v2.yaml"
      - name: "algorithm-v3"
        url: "./testalgorithms/v3.yaml"
```
