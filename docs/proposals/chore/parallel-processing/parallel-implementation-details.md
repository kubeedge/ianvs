# Parallel Processing Implementation Details

This document supplements the main [Parallel Processing Proposal](parallel-processing.md) with in-depth technical details, code walkthroughs, and research findings.

## Table of Contents

1. [Line-by-Line Code Explanation](#line-by-line-code-explanation)
2. [Worker Memory Management & OOM Prevention](#worker-memory-management--oom-prevention)
3. [Parallelization Support by AI Learning Paradigm](#parallelization-support-by-ai-learning-paradigm)
4. [Future Work Roadmap](#future-work-roadmap)

---

## Line-by-Line Code Explanation

This section provides an exhaustive, line-by-line walkthrough of every code change introduced in Phase 1. The goal is to ensure that every contributor and reviewer understands not just *what* was changed, but *why* each line exists.

### Overview of All Changes

| File Modified | Lines Added | Description |
|---------------|-------------|-------------|
| `core/testcasecontroller/testcasecontroller.py` | ~45 | Added parallel execution logic to `run_testcases` |
| `core/testcasecontroller/testcase/testcase.py` | ~15 | Added top-level worker function `run_testcase_func` |
| `core/testcasecontroller/testcase/__init__.py` | ~1 | Exported the new worker function |
| `core/cmd/benchmarking.py` | ~10 | Added `--parallel` and `--workers` CLI arguments |
| `core/cmd/obj/benchmarkingjob.py` | ~10 | Added config parsing for parallel fields |

### File 1: `core/testcasecontroller/testcasecontroller.py`

This is the core file where the parallel execution logic lives. Below is the explanation of the `run_testcases` modifications.

#### New Imports Added

```python
import concurrent.futures   # Python's built-in module for process/thread pools
import os                    # Used for os.cpu_count() to detect available cores
```

**Why `concurrent.futures`?** It is part of the Python standard library (no pip install needed), provides a high-level API for process-based parallelism, and handles process lifecycle management automatically.

**Why `os`?** We use `os.cpu_count()` to determine how many CPU cores the machine has, so we can set a safe default worker count.

#### New Import from Our Codebase

```python
from core.testcasecontroller.testcase import TestCase, run_testcase_func
```

**Why import `run_testcase_func`?** This is the new top-level function we will create in `testcase.py`. We need it here because `ProcessPoolExecutor.submit()` requires a callable that can be pickled (serialized). Class methods and lambdas cannot be pickled, so we use a module-level function instead.

#### Updated Method Signature

```python
# BEFORE (existing):
def run_testcases(self, workspace):

# AFTER (updated):
def run_testcases(self, workspace, parallel=False, workers=None):
```

**Line-by-line explanation:**
- `parallel=False`: A boolean flag. Defaults to `False` so that all existing callers of this method continue to work without any changes. When the user passes `--parallel` on the CLI, this becomes `True`.
- `workers=None`: An optional integer. When `None`, we auto-detect a safe value. When the user passes `--workers 4`, this becomes `4`.

**Why default to `False`?** This is the backward compatibility guarantee. Any code that currently calls `controller.run_testcases(workspace)` will continue to use the serial path because `parallel` defaults to `False`.

#### The Parallel Branch (`if parallel:`)

```python
if parallel:
    # Step 1: Determine worker count
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
```

**Line-by-line:**
- `if workers is None`: The user did not specify `--workers`, so we auto-detect.
- `os.cpu_count()`: Returns the number of CPU cores (e.g., 8 on a typical machine). Can return `None` on some exotic platforms.
- `or 2`: If `os.cpu_count()` returns `None`, fall back to 2 as a safe assumption.
- `- 1`: Reserve one core for the operating system and other background tasks. This prevents the machine from becoming unresponsive during benchmarking.
- `max(1, ...)`: Ensure we always have at least 1 worker, even on a single-core machine where `cpu_count() - 1 = 0`.

**Example:** On an 8-core machine: `max(1, 8 - 1) = 7` workers.

```python
    LOGGER.info(f"Running {len(self.test_cases)} test cases "
               f"in parallel with {workers} workers")
```

**Why log this?** So that when users read the log output, they can immediately see how many workers were used. This is critical for debugging performance issues.

```python
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
```

**Line-by-line:**
- `ProcessPoolExecutor`: Creates a pool of separate Python processes (not threads). Each process has its own memory space and its own Python interpreter, bypassing the GIL (Global Interpreter Lock).
- `max_workers=workers`: Limits the number of simultaneous processes.
- `with ... as executor`: The context manager ensures that when the block exits (either normally or due to an exception), all worker processes are properly shut down and their resources are released.

**Why `ProcessPoolExecutor` and not `ThreadPoolExecutor`?** Python's GIL prevents threads from executing CPU-bound Python code in parallel. Since ML training and evaluation are CPU-bound, threads would provide zero speedup. Processes bypass the GIL entirely.

```python
        future_to_testcase = {
            executor.submit(run_testcase_func, testcase, workspace): testcase
            for testcase in self.test_cases
        }
```

**Line-by-line:**
- `executor.submit(run_testcase_func, testcase, workspace)`: Submits a single task to the process pool (non-blocking).
- `run_testcase_func`: The top-level function we defined in `testcase.py`.
- `testcase, workspace`: Arguments passed to the worker function. Both must be picklable.
- `{ future: testcase for ... }`: Dictionary mapping futures back to test cases for lookups.

```python
        for future in concurrent.futures.as_completed(future_to_testcase):
```

**Why `as_completed()`?** It yields futures as soon as they finish, regardless of submission order. This allows us to process results immediately (e.g., stream logs).

```python
            testcase = future_to_testcase[future]
            try:
                result = future.result()
```

**`future.result()`**: Blocks until the future is complete. If the worker crashed, this raises an exception.

```python
                if result["status"] == "success":
                    # ... Store success ...
                else:
                    LOGGER.error(f"Test case {testcase.id} failed: "
                                f"{result.get('error')}")
```

**Why check `result["status"]`?** `run_testcase_func` catches Python exceptions internally and returns them as a failure status dict. This prevents one test case from crashing the main process.


### File 2: `core/testcasecontroller/testcase/testcase.py`

#### The Worker Function

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
```

**Why is this a top-level function and not a method of `TestCase`?** Python's `multiprocessing` module uses `pickle` to serialize objects and send them to worker processes. `pickle` cannot serialize bound methods (like `testcase.run`) or lambda functions. Only module-level functions can be pickled. By defining `run_testcase_func` at the top level of the module (outside any class), we ensure it can be sent to worker processes.

```python
    try:
        res = testcase.run(workspace)
```

**This line calls the existing `TestCase.run()` method.** We are not reimplementing any test case logic. The worker function is a thin wrapper that delegates to the existing implementation.

```python
        return {
            "status": "success",
            "config": testcase.algorithm.name,
            "results": res
        }
```

**Why return a dictionary instead of just `res`?** We need to communicate both the result AND the status back to the main process. By using a standardized dictionary format, the controller can distinguish between success and failure without relying on exceptions (which behave differently across process boundaries).

- `"status"`: Either `"success"` or `"failed"`. The controller checks this field.
- `"config"`: The algorithm name, included for logging and debugging.
- `"results"`: The actual test case results (metrics, predictions, etc.).

```python
    except Exception as e:
        return {
            "status": "failed",
            "config": testcase.algorithm.name,
            "error": str(e)
        }
```

**Why catch all exceptions?** If we let the exception propagate, it would be raised in the main process when calling `future.result()`. While we handle that case too, catching here gives us cleaner error messages because we can include the algorithm name and convert the exception to a string before it crosses the process boundary (some exception types are not picklable).

### File 3: `core/testcasecontroller/testcase/__init__.py`

```python
# BEFORE:
from .testcase import TestCase

# AFTER:
from .testcase import TestCase, run_testcase_func
```

**One line changed.** We add `run_testcase_func` to the package's public API so that `testcasecontroller.py` can import it with `from core.testcasecontroller.testcase import TestCase, run_testcase_func`.

### File 4: `core/cmd/benchmarking.py`

```python
# NEW: Parallel execution arguments
parser.add_argument("-p", "--parallel",
                    action="store_true",
                    help="run test cases in parallel")
```

**Line-by-line:**
- `"-p", "--parallel"`: The short form `-p` and long form `--parallel`. Users can use either.
- `action="store_true"`: This means `--parallel` is a flag (no value needed). If present, `args.parallel = True`. If absent, `args.parallel = False`.
- `help="run test cases in parallel"`: Shown when the user runs `ianvs --help`.

```python
parser.add_argument("-w", "--workers",
                    type=int,
                    help="number of workers for parallel execution")
```

**Line-by-line:**
- `"-w", "--workers"`: Short form `-w` and long form `--workers`.
- `type=int`: Argparse will automatically convert the string argument to an integer and raise an error if the user passes a non-integer value (e.g., `--workers abc`).
- If the user does not pass `--workers`, `args.workers` will be `None`, which triggers auto-detection in the controller.

### File 5: `core/cmd/obj/benchmarkingjob.py`

```python
# NEW: Parallel execution settings
self.parallel = False
self.workers = None
```

**Why initialize to `False` and `None`?** These are the safe defaults. `False` means serial execution (backward compatible). `None` means auto-detect worker count.

```python
self._parse_config(config)
```

This existing line parses the YAML configuration. We add two new cases to the parsing loop:

```python
elif k == "parallel_execution":
    self.parallel = v
elif k == "num_workers":
    self.workers = v
```

**These lines read from the YAML file.** If the user has `parallel_execution: true` in their `benchmarkingjob.yaml`, then `self.parallel` becomes `True`.

```python
# CLI args override YAML config
if args:
    if args.parallel:
        self.parallel = args.parallel
    if args.workers:
        self.workers = args.workers
```

**Why does CLI override YAML?** This is a common pattern in CLI tools. The YAML file represents the default/persistent configuration, while CLI arguments represent the user's immediate intent. Example: A user might have `parallel_execution: false` in their YAML but want to run one specific benchmark in parallel: `ianvs -f job.yaml --parallel`.

### Why These Changes Are Safe

1. **Code Addition, Not Modification**: We are adding new code paths, not changing existing ones. The `else` branch in `run_testcases` is identical to the current implementation.
2. **Safe Defaults**: `parallel` defaults to `False`, ensuring the code enters the `else` block (serial path) by default.
3. **Clear Separation**: The parallel logic is entirely contained within the `if parallel:` block. It cannot accidentally affect the serial path.
4. **Easy Verification**: Running `ianvs -f benchmarkingjob.yaml` (without `--parallel`) executes the exact same code path as the current codebase.
5. **No New Dependencies**: All imports (`concurrent.futures`, `os`) are from the Python standard library.

### Testing Strategy for Code Changes

1. **Verify Serial Path Unchanged**:
   ```bash
   # Should pass exactly as before
   ianvs -f benchmarkingjob.yaml
   ```

2. **Verify Parallel Equivalence**:
   ```bash
   # Run serial
   ianvs -f job.yaml
   mv ./workspace ./serial_results
   # Run parallel
   ianvs -f job.yaml --parallel --workers 4
   mv ./workspace ./parallel_results
   # Diff results
   diff -r ./serial_results ./parallel_results
   ```

3. **Verify Config Priority**:
   - Set `parallel_execution: false` in YAML.
   - Run `ianvs -f job.yaml --parallel`.
   - Confirm parallel mode is activated (CLI overrides YAML).

---

## Worker Memory Management & OOM Prevention

### Problem Statement

Parallel execution increases memory usage linearly with the number of workers. While CPU cores are often abundant, **System RAM** is usually the bottleneck.

**Typical memory usage per test case:**
- **PCB-AOI (Object Detection)**: ~2-4GB RAM per test case
- **Robot Lifelong Learning**: ~4-8GB RAM per test case
- **LLM Fine-tuning**: ~14GB+ RAM per test case

**Risk**: Naively setting `--workers 32` on a 64GB RAM machine running LLM tests will cause Out-Of-Memory (OOM) errors, potentially freezing the system.

### Runtime Memory Monitoring (Phase 1.5)

In Phase 1.5, we plan to introduce helper utilities for memory-aware scheduling:

```python
import psutil

def estimate_safe_workers(test_case_sample, workspace):
    """
    Profile a single test case to measure peak memory usage
    and calculate how many can safely fit in available RAM.
    
    Algorithm:
    1. Run one test case serially while monitoring RSS (Resident Set Size)
    2. Record peak memory usage
    3. Get available system RAM
    4. Apply safety factor (use only 80% of available RAM)
    5. Return min(cpu_count, floor(safe_ram / memory_per_test))
    """
    import tracemalloc
    tracemalloc.start()
    test_case_sample.run(workspace)
    current, peak = tracemalloc.get_traced_memory()
    peak_gb = peak / (1024 ** 3)
    tracemalloc.stop()
    
    # Calculate safe workers
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    safe_ram = available_gb * 0.8  # 80% safety factor
    cpu_count = os.cpu_count() or 2
    
    safe_workers = min(cpu_count - 1, int(safe_ram // peak_gb))
    return max(1, safe_workers)
```

### Empirical Worker Optimization Research (Phase 1.5)

**Timeline**: Weeks 5-7 after initial Phase 1 release.

**Experiment 1: Worker Count vs. Performance**
- **Setup**: Test on 4/8/16/32-core machines
- **Workloads**: PCB-AOI, Robot LL, LLM fine-tuning
- **Workers tested**: [1, 2, 4, 6, 8, 12, 16, 24, 32]
- **Metrics**: Total time, speedup factor, CPU utilization %, RAM usage, Swap usage

**Experiment 2: Memory Profiling by Workload Type**
- **Expected Findings**:
  - PCB-AOI: ~3.2GB/worker -- Safe: 3 workers (16GB RAM), 7 workers (32GB RAM)
  - Robot LL: ~4.8GB/worker -- Safe: 2 workers (16GB RAM), 4 workers (32GB RAM)
  - LLM (7B): ~14GB/worker -- Safe: 1 worker (16GB RAM), 2 workers (32GB RAM)

### User Guidance: Choosing Worker Count

#### Quick Reference Table

| Workload Type | Laptop (16GB RAM) | Workstation (64GB RAM) | Server (256GB RAM) |
|---------------|-------------------|------------------------|--------------------|
| **Light** (Tabular/Small CV) | `cpu_count - 1` | `cpu_count - 1` | `cpu_count - 1` |
| **Medium** (Object Det/Seg) | 2 workers | 8 workers | 24 workers |
| **Heavy** (LLM/GenAI) | 1 worker (Serial) | 2-3 workers | 12 workers |

---

## Parallelization Support by AI Learning Paradigm

### Overview

It is crucial to distinguish between two types of parallelism:

1. **Inter-Test-Case Parallelism**: Running multiple independent test cases at the same time. Each test case runs a complete training + evaluation cycle. **This is what Phase 1 implements.**

2. **Intra-Model Parallelism**: Distributing a single model's training across multiple GPUs/processes (e.g., PyTorch DDP). **This is deferred to Phase 2+.**

### Support Matrix

| Learning Paradigm | Support Level | Implementation Strategy |
|-------------------|--------------|-------------------------|
| **Joint Inference** | Full Support | Divide dataset into $N$ partitions. Each worker handles Partition $i$. Results are merged. |
| **Lifelong Learning** | Full Support | Each test case is an independent task sequence. Run sequence A and sequence B in parallel. |
| **Federated Learning** | Full Support | Each simulation (FedAvg, FedProx) is independent. Run multiple algorithm comparisons in parallel. |
| **Incremental Learning** | Partial Support | Can run different experimental configs in parallel. Cannot split one incremental training run across workers (requires DDP). |

### Detailed Analysis by Paradigm

#### Joint Inference (Fully Supported)
- **What**: Testing a model on a dataset partition.
- **Why it works**: Each partition is completely independent. There is no shared state between workers.

#### Lifelong Learning (Fully Supported)
- **What**: Learning continuously from a stream of tasks.
- **Why it works**: Each algorithm instance (EWC vs LwF) is independent and maintains its own model weights.

#### Incremental Learning (Partially Supported)
- **Supported**: Config-Level Parallelism (Running ResNet18 vs ResNet50 experiments in parallel).
- **Not Supported**: Intra-Model Parallelism (Using 4 GPUs to train one ResNet model faster).
- **Workaround**: Focus on benchmarking throughput (experiments per hour) rather than latency (time per experiment).

---

## Future Work Roadmap

### Phase 2: Enhanced Resource Management (3-4 months)
- **Goals**: GPU-aware scheduling, memory-aware worker limiting
- **Features**: `resources: { gpu: 1, ram: "4GB" }` in YAML config
- **Deliverables**: Resource scheduler module, Updated YAML schema

### Phase 3: Distributed Multi-Node Execution (6-12 months)
- **Goals**: Scale beyond a single machine using Ray or Kubernetes Job Dispatcher
- **Conceptual Code**:
```python
import ray

@ray.remote(num_gpus=1)
def remote_testcase_runner(testcase, workspace):
    return testcase.run(workspace)

# In Controller:
futures = [remote_testcase_runner.remote(tc, ws) for tc in test_cases]
results = ray.get(futures)
```
- **Deliverables**: Ianvs Distributed Controller, Cluster deployment documentation

### Phase 4: Intelligent Optimization (12+ months)
- **Goals**: AI-powered benchmarking optimization
- **Features**: ML model predicts optimal worker count based on dataset size and model type
- **Deliverables**: Auto-tuning module
