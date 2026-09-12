# Proposal: KubeEdge-Ianvs Simulation Sandbox

### Environment-Isolated Execution and System Metrics Profiling

| | |
|---|---|
| **Program** | LFX Mentorship 2026 Term 3 (Sep–Nov) |
| **Project** | CNCF – KubeEdge: KubeEdge-Ianvs Simulation Sandbox |
| **Tracking issue** | [#348 — Simulator for Edge-cloud Collaborative AI: a Lifelong Learning example](https://github.com/kubeedge/ianvs/issues/348) |
| **Prior art** | Proposal [PR #35](https://github.com/kubeedge/ianvs/pull/35), implementation [PR #39](https://github.com/kubeedge/ianvs/pull/39) — Zhang Yang ([@iszhyang](https://github.com/iszhyang)), OSPP 2022 |
| **Related** | [#307](https://github.com/kubeedge/ianvs/issues/307) feature ledger, [#8](https://github.com/kubeedge/ianvs/issues/8) parallel test cases, [#430](https://github.com/kubeedge/ianvs/issues/430) / [#460](https://github.com/kubeedge/ianvs/issues/460) example failures |
| **Status** | RFC — design and reference implementation submitted together |

---

## 0. Summary

The Ianvs architecture has always specified a **Simulation Controller**. Zhang Yang built roughly half of it in 2022. The other half was never written, and the half that exists has since stopped working.

This proposal restores it, completes it, and adds the system-metrics layer the project description asks for. Three things distinguish this submission:

1. **Every claim about the existing code is reproducible.** A single script, `verify_legacy_simulation.py`, runs against a pristine `main` checkout and confirms **13 defects**. It is included, it exits with the defect count, and it is designed to become a CI regression guard once the fixes land.

2. **Two of those 13 defects are, to my knowledge, previously unreported** — and one of them means the feature's headline capability does not work at all. See §2.3.

3. **The design is two-tier, and the lightweight tier is the default.** The failure modes named in the project description — dependency conflicts, path contamination, OOM crashes — are *single-process* problems. They are caused by every test case sharing one interpreter and one address space. A Kubernetes cluster does fix them, but it requires a privileged Linux host, which excludes macOS contributors and most CI runners. Making the cluster the only path would leave the most common form of the problem unsolved for the people who hit it most. Reasoning in §4.

A working reference implementation accompanies this document: 10 new modules, 4 patched core files, the now-superseded `simulation_system_admin` package removed, 60 passing tests, `pylint core` at 10.00/10, and a live-run OOM-containment demo with a real, enforced memory ceiling — not just a measured one. It is offered as evidence of feasibility, not as a finished merge candidate.

---

## 1. Background

KubeEdge-Ianvs benchmarks distributed synergy AI. Today it runs **every test case inside one shared Python process**:

```python
# core/testcasecontroller/testcasecontroller.py
for testcase in self.test_cases:
    try:
        res, time = (testcase.run(workspace), utils.get_local_time())
    except Exception as err:
        raise RuntimeError(f"testcase(id={testcase.id}) runs failed, error: {err}") from err
```

That loop has three consequences, and all three are visible in the issue tracker.

**Dependencies collide.** One `sys.path`, one set of installed versions. Contributors resolve conflicts by editing core files, which produces regressions across the 30+ maintained examples. Issue [#460](https://github.com/kubeedge/ianvs/issues/460) records `multiedge_inference.py` importing `onnx` unconditionally at module load, so an unrelated example cannot start without a heavy dependency it never uses.

> I hit exactly this while preparing this proposal. Standing up a clean Ianvs environment produced, in sequence: `ModuleNotFoundError: onnx` from the eager import chain; `sedna 0.4.1 requires six~=1.15.0` against an incompatible installed `six`; `scikit-learn 1.8.0 requires joblib>=1.3.0, but you have joblib 1.1.1`; and `ImportError: cannot import name 'JsonlDataParse' from 'sedna.datasources'` — Ianvs `main` depending on a Sedna API that the released library does not export. I confirmed that last one reproduces on a pristine clone with no patches applied, so it is pre-existing framework/library drift, not something I introduced. Four dependency failures before a single line of algorithm code ran. This is the problem, encountered live.

**State leaks.** Absolute paths, environment variables and `ClassFactory` registrations set by one test case persist into the next. Failures depend on execution order, which makes them very hard to reproduce.

**One crash destroys everything.** The `raise` above exits the loop. Results already computed for earlier test cases are never handed to `StoryManager`. A crash in the tenth of ten LLM benchmarks discards the nine that succeeded. Issue [#430](https://github.com/kubeedge/ianvs/issues/430) describes precisely this: *partial benchmark failures currently result in the total loss of hours of compute time*.

And separately: **Ianvs cannot measure what a system costs to run.** `SystemMetricType` enumerates `samples_transfer_ratio`, `FWT`, `BWT`, `task_avg_acc`, `MATRIX`, `forget_rate` — every one of them algorithm-wise. There is no peak memory, no CPU utilisation, no wall-clock. For an *edge* AI benchmark, where the binding constraint is almost always the device envelope rather than accuracy, that is a structural gap. A leaderboard that ranks an algorithm first without recording that it needed 14 GiB of RAM is not describing something deployable to an edge node.

---

## 2. What exists today, and what is broken

### 2.1 The 2022 implementation

Zhang Yang's code was, until this restoration, still in the tree at `core/testcasecontroller/simulation/` and `core/testcasecontroller/simulation_system_admin/`. It provided:

- `Simulation` — parses `cloud_number`, `edge_number`, `cluster_name`, `kubeedge_version`, `sedna_version` from `benchmarkingjob.yaml`
- `simulation_system_admin` — host checks, plus `build_simulation_enviroment()` invoking the [Sedna all-in-one script](https://github.com/kubeedge/sedna/blob/main/scripts/installation/all-in-one.sh)
- A hook in `BenchmarkingJob.run()`

The design was sound and the community reviewed it. What follows is not a criticism of that work; four years of drift in Ianvs, KubeEdge, Sedna, kind and Python is simply a lot of drift.

### 2.2 The Simulation Job Administrator was never built

PR #35 specified four functions. None existed in the repository before this proposal's reference implementation:

1. Build the images of algorithms to be tested
2. Generate the YAML file of the simulation job
3. Deploy and delete the simulation job with workers
4. List-watch the results of the simulation job with workers

So even with a cluster running, there was no mechanism to get an algorithm onto it or a result back. `job_admin.py` in the reference implementation builds this component for the first time (§5); it has not been exercised against a live cluster — see "Honest limitations."

### 2.3 Verified defects

Run `python3 verify_legacy_simulation.py` from a pristine checkout. Output on `main`, commit `36ec008` (August 2026): **13 confirmed, 0 skipped**.

| ID | Defect | Severity | How it was verified |
|---|---|---|---|
| **B12** | **Node count silently exceeds the backend's hard ceiling** | **Critical** | Fetched the Sedna script; found `MAX_CLOUD_WORKER_NODES=2`, `MAX_EDGE_WORKER_NODES=3` |
| **B7** | **Cluster is built but never destroyed** | **High** | Walked all of `core/`: the destroy function has definitions and exports but **zero call sites** |
| B1 | Docker/kind auto-install branches are unreachable dead code | High | `subprocess.run("exit 3", check=True)` raises `CalledProcessError` before the `returncode` test runs |
| B14 | One failing test case discards all completed results | High | Regex-matched the `raise RuntimeError` that exits the loop |
| B13 | No system-level metric types exist | High | Enumerated `SystemMetricType`; no memory/CPU/time member |
| B11 | ARM64 hosts unsupported | Medium | `arch()` maps only `x86_64`; other machine strings pass through verbatim |
| B3 | Unknown config keys silently dropped | Medium | Live: `edge_nodes: 5` → `edge_number = 0`, no error |
| B2 | Booleans accepted as node counts | Medium | `isinstance(True, int)` is `True` |
| B4 | Empty required fields accepted | Medium | Type checks only; emits a bare `CLUSTER_NAME=` |
| B5 | `lscpu \| grep CPU:` parsing is fragile | Medium | Returned empty on a slim container → `IndexError` |
| B8 | kind pinned to v0.17.0 (2022), amd64 hardcoded | Medium | Regex on the installer URL |
| B9 | No shipped example uses the feature | Medium | Zero of the shipped configs contain a `simulation:` block |
| B6 | Build uses `/master/`, destroy uses `/main/` | **Low** | Both return HTTP 200 via redirect — see note below |

**On B12, the critical one.** The Sedna all-in-one backend caps topology at **2 cloud workers and 3 edge nodes** and aborts above that. The Ianvs `Simulation` class accepted any integer. So `edge_number: 10` — large-scale multi-node simulation, the entire stated purpose of the feature — passed every Ianvs-side validation, passed the host checks, and then died inside a `curl | bash` pipeline with an error the user could not connect to their config file. **The feature's headline capability was capped at three edge nodes by its own backend, and nothing told the user.** Fixing this properly means either validating against the ceiling (so the failure is immediate and legible) or providing a path not bound by it. This proposal does both: the cluster tier now validates at parse time, and the process tier is not bound by the script at all.

**On B7.** `destory_simulation_enviroment()` was defined and re-exported from `__init__.py`, and called from nowhere. `BenchmarkingJob.run()` built the cluster and never tore it down. Every run left a kind cluster and its containers resident until the user removed them by hand. The fix is structural, not a one-line addition: teardown is now driven from a `finally` block (`SimulationEnvironmentAdministrator.destroy()`, called by every caller of `SimulationController.run_testcases`) so it survives exceptions and `Ctrl-C`.

**On B6, a correction.** Both `/master/` and `/main/` return HTTP 200 — GitHub redirects between them. This is an internal inconsistency worth tidying, not a breakage, and I have graded it Low accordingly. I flag it explicitly because I have seen it described elsewhere as a hard failure, and a proposal that overstates its evidence is worth less than one that does not.

---

## 3. Goals and non-goals

### Goals

- **G1 — Isolation.** A test case's dependencies, paths and process state cannot reach the Ianvs core or another test case.
- **G2 — Resource bounding.** A test case can be given an edge-like CPU/memory ceiling, enforced by the kernel, not merely reported after the fact.
- **G3 — Fault containment.** A crash, hang or OOM is recorded as a failure; the run continues and prior results survive.
- **G4 — System metrics.** Peak memory, CPU utilisation and wall-clock are captured per test case and surfaced through the existing `StoryManager` leaderboard next to accuracy.
- **G5 — KubeEdge-native validation.** `kind` + `edgecore` + Sedna, with the Job Administrator completed, so results are measured on genuine KubeEdge topology.
- **G6 — Zero regression.** With no `sandbox` block, execution is byte-identical to today.

### Non-goals

- **Parallel test-case execution** ([#8](https://github.com/kubeedge/ianvs/issues/8)). Isolation is its prerequisite, and prior attempts were paused by reviewers for exactly that reason. Deferred deliberately; see §8.
- **Modifying paradigm internals.** The sandbox wraps a whole test case. No paradigm changes.
- **Migrating existing examples.** One or two proof-of-concept examples only.
- **Adversarial isolation.** Ianvs runs code the user chose to benchmark. The threat model is accident, not attacker.

---

## 4. Design: two tiers, lightweight by default

### 4.1 The reasoning

The project description names three failure modes: *dependency conflicts, path contamination, and fatal OOM crashes*. Each is caused by a shared interpreter. Both a cluster and a bounded subprocess fix them — but they differ sharply in who can actually run them:

| | Process tier | Cluster tier |
|---|---|---|
| Root required | No | Yes |
| Docker / Kubernetes | No | Yes |
| macOS, Windows | Yes | No |
| Typical CI runner | Yes | Rarely |
| Startup cost | ~1–30 s | 15–40 min cold |
| Memory floor | ~50 MiB | ~4 GiB |
| Fixes dependency conflicts | Yes | Yes |
| Fixes OOM cascades | Yes | Yes |
| Real edge-cloud topology | No | **Yes** |
| Network condition emulation | No | **Yes** |
| Bounded by the 3-node ceiling (B12) | **No** | Yes |

The cluster tier is the only way to measure genuine KubeEdge topology, and it is a required deliverable. But if it is the *only* tier, then a contributor on a MacBook, or a GitHub Actions job, cannot get isolation at all — and those are the people filing the dependency-conflict issues. Worse, the cluster tier inherits B12's three-edge-node ceiling.

So: **process tier by default, cluster tier for topology fidelity.** Both behind one config contract and one metrics schema, so a user moves between them by changing a single word.

### 4.2 Architecture

```
BenchmarkingJob.run()
  │
  ├─ TestEnvManager.prepare()
  ├─ TestCaseController.build_testcases()
  │
  └─ TestCaseController.run_testcases(workspace, sandbox, simulation)
       │
       ├── sandbox absent or disabled ──► _run_testcases_inline()   ← today's path, untouched
       │
       └── sandbox.enabled ────────────► SimulationController
                                            │
                                            ├─ SimulationEnvironmentAdministrator
                                            │    1. parse system config
                                            │    2. check host environment
                                            │    3. build environment
                                            │    4. deploy required modules
                                            │    5. close and delete            ← finally block
                                            │
                                            ├─ ProcessSandbox  ──► worker process
                                            │                       venv + RLIMIT_AS/affinity
                                            │                       TreeSampler (PSS/RSS)
                                            │
                                            └─ ClusterSandbox ──► SimulationJobAdministrator
                                                                    1. build algorithm image
                                                                    2. generate job YAML
                                                                    3. deploy / delete job
                                                                    4. list-watch results
```

The five Environment Administrator responsibilities and the four Job Administrator functions map one-to-one onto the issue text. Responsibility 5 is the one that was missing (B7), and it is now owned by a `finally` block rather than a hopeful call site.

### 4.3 Process tier — what isolation actually means

| Dimension | Mechanism | Status in this reference build |
|---|---|---|
| Python dependencies | per-test-case `venv`, seeded from parent site-packages so heavy shared deps are not re-downloaded; an example's own `requirements.txt` installs on top | Implemented |
| `sys.path` / `PYTHONPATH` | rebuilt from scratch in the child | Implemented |
| Working directory | private temp dir, removed on teardown (kept on request via `keep_workdir`) | Implemented |
| Environment variables | **allowlist** — the parent's env does not leak | Implemented |
| Memory | `RLIMIT_AS`, applied via `preexec_fn` before the worker execs | Implemented and verified live (§5) |
| Memory (preferred mechanism) | cgroup v2 `memory.max` + `memory.swap.max=0` where a writable delegation exists | **Not implemented in this build** — `hostcheck.has_cgroup_v2_write_access()` exists to detect it, but enforcement still falls through to `RLIMIT_AS` unconditionally. Roadmapped, see §6. |
| CPU | affinity mask, applied the same way as the memory limit | Implemented |
| Wall clock | timeout → `SIGTERM` to the process group → `SIGKILL` after grace | Implemented |
| Crash blast radius | separate process; the parent survives | Implemented and verified live |

Two choices worth defending:

**Allowlist, not denylist.** A denylist requires knowing every variable an algorithm might set. An allowlist means a leak is a bug in one visible list. Tested: `test_environment_is_allowlisted`.

**`RLIMIT_AS` as the current, honestly-scoped mechanism.** `RLIMIT_AS` bounds *virtual* address space, not resident memory. ML runtimes routinely reserve far more virtual memory than they touch, so a tight `RLIMIT_AS` ceiling can reject a workload that would have fit comfortably — and, observed live in §5, a rejected allocation surfaces as a catchable Python `MemoryError` inside the worker rather than a kernel `SIGKILL`. `detect_oom_kill()` treats both the same way (a kernel signal *or* a marker in the captured error text), so the sandbox still reports it correctly as an OOM failure either way. cgroup v2 `memory.max` is the more accurate mechanism, bounding resident memory and delivering a real kernel kill, and is the documented preferred path — but it needs a writable cgroup delegation that this reference build does not yet create. This is disclosed, not glossed over: see the roadmap and risk table.

### 4.4 Metrics: measuring memory honestly

Peak memory is reported as **PSS** (proportional set size) where the kernel exposes `/proc/<pid>/smaps_rollup` via `psutil.Process.memory_full_info()`, and summed RSS otherwise, with the source recorded in the output as `memory_source`.

This matters more than it sounds. Summing RSS across a process tree double-counts every shared page — the CUDA runtime, the interpreter's own text, shared model weights after a fork. For the LLM and VLA workloads this project targets, forked data loaders are routine and the inflation is large. Reporting a number that is wrong in a direction users cannot detect would make the leaderboard actively misleading. PSS divides shared pages by the number of mappers, so the sum is meaningful.

Sampling covers **the whole process tree**, not the direct child, and runs on the parent — so a worker that is OOM-killed still yields the measurements taken up to the moment it died. That is exactly the case where the number matters most.

### 4.5 Configuration contract

Entirely optional. Absent ⇒ today's behaviour.

```yaml
benchmarkingjob:
  name: "sandbox_bench"
  workspace: "./workspace/sandbox_bench"
  testenv: "./examples/pcb-aoi/singletask_learning_bench/fault_detection/testenv/testenv.yaml"
  test_object: { ... }

  sandbox:
    enabled: true
    mode: "process"        # process | cluster | auto
    isolation: "venv"      # venv | none
    fail_fast: false
    resources:
      memory: "2Gi"
      cpus: 2
      timeout: 3600
    metrics: ["peak_memory", "cpu_utilization", "wall_time"]

  simulation:              # only read by the cluster tier; 2022 schema unchanged
    cloud_number: 1
    edge_number: 2
    cluster_name: "ianvs-sim"

  rank:
    sort_by: [ { "f1_score": "descend" }, { "peak_memory_mb": "ascend" } ]
    selected_dataitem:
      metrics: [ "f1_score", "peak_memory_mb", "cpu_utilization_pct", "wall_time_s" ]
```

`auto` degrades to the process tier when the host cannot support a cluster. Explicit modes never degrade silently: if you asked for `cluster` and the host cannot provide one, that is an error — quietly downgrading would mean publishing cluster-topology results that were never measured on a cluster.

### 4.6 Leaderboard integration

`SimulationController` merges the sandbox's system metrics into the same flat dict the paradigms already return (per the `sandbox.metrics` selection list, or all of them via `["all"]`), so `StoryManager` needs **no changes**:

| algorithm | paradigm | f1_score | peak_memory_mb | cpu_utilization_pct | wall_time_s |
|---|---|---|---|---|---|
| fpn_singletask | singletasklearning | 0.8987 | 1842.31 | 187.4 | 421.7 |
| fpn_incremental | incrementallearning | 0.9013 | 3721.88 | 194.1 | 1204.3 |

A user sorting by `f1_score` descending and `peak_memory_mb` ascending now gets the question edge deployment actually asks: *which algorithm is most accurate within my device budget?* The `memory_headroom_pct` metric expresses it directly, and is allowed to go negative when an algorithm exceeded the budget it was benchmarked against — clamping that at zero would hide the finding.

---

## 5. Reference implementation

Submitted with this proposal as evidence of feasibility, and built and verified against this actual `kubeedge/ianvs` checkout, not a mock.

```
core/testcasecontroller/simulation/
├── __init__.py            re-exports Simulation, SandboxConfig, SimulationController
├── simulation.py          hardened config — fixes B2, B3, B4, B12
├── config.py               SandboxConfig, ResourceQuota, k8s-style quantities
├── hostcheck.py            host probing — fixes B1, B5; tier-aware
├── profiler.py             PSS/RSS tree sampling (TreeSampler), OOM detection
├── worker.py                the inner "worker-in-worker" process
├── env_admin.py            Environment Administrator (5 responsibilities)
├── job_admin.py             Job Administrator (4 functions) — NEW, first implementation
├── controller.py            SimulationController, fault containment, metric merge
├── kubeutil.py              kubectl wrappers used by the cluster tier
└── sandbox/
    ├── base.py              backend interface
    ├── process.py           process tier — RLIMIT_AS/affinity enforcement, venv isolation
    └── cluster.py           cluster tier — fixes B6, B7, B8, B11

core/common/constant.py                        + SandboxMode, IsolationLevel, 6 metric types
core/testcasecontroller/metrics/metrics.py     + 6 metric functions
core/testcasecontroller/testcasecontroller.py  + sandbox dispatch, inline path preserved
core/cmd/obj/benchmarkingjob.py                + sandbox parsing, teardown moved
requirements.txt                               + psutil
core/testcasecontroller/simulation_system_admin/  REMOVED — superseded, zero remaining callers

tests/simulation/test_sandbox.py               60 tests
verify_legacy_simulation.py                    13-defect evidence script
examples/simulation/sandbox_bench/             annotated example config
```

### Verified behaviour

```
$ python3 -m pytest tests/simulation -q
60 passed in 0.11s

$ pylint core --max-positional-arguments=10
Your code has been rated at 10.00/10
```

Fault containment and **real quota enforcement**, run live against the actual `ProcessSandbox` (synthetic test cases — a paradigm was not wired in for this run; see "Honest limitations"), a 256 MiB quota, one test case deliberately allocating 2 GiB without bound:

```
--- memory-hog ---
  succeeded : False
  error     : test case was OOM-killed inside the sandbox. The declared memory
              quota was 256MiB; peak observed was 9.21MiB. The Ianvs process
              itself was unaffected. (worker detail: MemoryError, raised when
              the RLIMIT_AS ceiling rejected a 2GiB allocation)
  peak mem  : 9.21 MiB (quota 256.0 MiB)
  exit code : 1   oom_killed: True

--- well-behaved ---
  succeeded : True
  metrics   : {'accuracy': 0.913, 'f1_score': 0.887}
  peak mem  : 27.48 MiB (quota 256.0 MiB)

PARENT PROCESS ALIVE — survived the OOM child.
Results preserved for 1/2 test cases.
```

Note the shape of this result: the 2 GiB allocation was actually *rejected by the kernel-enforced ceiling* — this is not a measurement of an OOM that happened to occur, it is a demonstration that the quota is real. Under today's inline path, the first case aborts the job and the second never runs; here it does.

### Backward compatibility

- `Simulation.__name__` and all five attribute names unchanged — `benchmarkingjob.py` dispatches on `str.lower(Simulation.__name__)`
- The 2022 YAML schema parses identically (`test_legacy_config_still_parses`)
- With no `sandbox` block, `_run_testcases_inline()` runs — the original loop, verbatim, including B14's discard-on-first-failure behaviour. Fixing that unconditionally would be a behaviour change with no opt-in; fixing it only when the sandbox is enabled keeps G6 (zero regression) literal rather than aspirational.
- New config keys are additive; no existing key changes meaning

---

## 6. Roadmap (12 weeks)

| Weeks | Milestone | Deliverable |
|---|---|---|
| **1–2** | **Research document** | Sandbox techniques comparison: `venv`/`uv`, subprocess + cgroups v2, container, `kind`. Analysis of why the 2022 container-in-container approach stalled. Interaction analysis across all 5 paradigms confirming the boundary wraps a whole test case and needs no paradigm change. **Submitted to SIG AI before implementation.** |
| **3** | **Restoration PR** | Fix all 13 verified defects. `verify_legacy_simulation.py` merged as a CI regression guard. |
| **4–5** | **Process tier hardening** | cgroup v2 `memory.max`/`cpu.max` as the preferred enforcement path, with `RLIMIT_AS` + affinity as the already-implemented fallback. Timeout escalation and teardown already implemented; extend unit tests to cover the cgroup path on hosts where it's writable. |
| **6** | **Profiler** | PSS/RSS tree sampling and OOM detection are implemented; cross-platform degradation (no `psutil`, no `/proc`) verified on Linux + macOS. |
| **7** | **Leaderboard** | `SystemMetricType` extension and metric functions are implemented; `StoryManager` integration verified against a real multi-algorithm run. Mid-term report. |
| **8–9** | **Cluster tier** | `kind` + `edgecore` + Sedna provisioning code exists (`sandbox/cluster.py`, `job_admin.py`, `kubeutil.py`); validated end-to-end on Ubuntu LTS with current KubeEdge/Sedna for the first time. |
| **10** | **Job Administrator hardening** | The four functions (image build, YAML generation, deploy/delete, ConfigMap list-watch) have a first implementation; harden against real cluster failure modes (image pull errors, pod eviction, partial ConfigMap writes). |
| **11** | **PoC + validation** | 1–2 examples (a lifelong-learning example and one heavy example) run in both tiers through the real paradigm code, not the synthetic harness used for the reference build. Cross-tier metric comparison. |
| **12** | **Docs + handover** | User guide, migration notes, `docs/proposals/` update, final report. |

Weeks 1–3 are deliberately front-loaded onto research and restoration. A reviewer asked for comprehensive research before code, and the restoration PR is small, independently valuable, and mergeable on its own — so there is something in `main` by week 3 regardless of how the rest lands.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| cgroup v2 unavailable (macOS, rootless, some CI) — and not yet implemented at all in this reference build | `RLIMIT_AS` + affinity is already implemented and is the mechanism actually exercised in §5; the host check (`hostcheck.has_cgroup_v2_write_access`) reports honestly which would be in force once cgroup v2 support lands, so a user is never told a number came from a mechanism that didn't run |
| `venv` creation slow for heavy examples | `system_site_packages=True` inherits heavy deps; `isolation: none` skips venv entirely; per-run cache is future work |
| Cluster provisioning flaky in CI | Cluster tier is opt-in and never on the default CI path; process tier covers CI |
| PSS unavailable (non-Linux, restricted `/proc`) | Fall back to summed RSS and record `memory_source` in the output, so the number is never silently of unknown provenance |
| Scope too large for 12 weeks | Tiers are independent. Process tier + profiler + leaderboard (weeks 3–7) is a complete, shippable feature on its own; cluster tier builds on it |
| Sedna's 3-node ceiling limits "large-scale" claims | Documented explicitly (B12); process tier is not bound by it; raising the ceiling is an upstream Sedna conversation this project can open |

---

## 8. Future work

**Parallel test cases ([#8](https://github.com/kubeedge/ianvs/issues/8)).** This proposal is its prerequisite. Once each test case owns its runtime and resource envelope, a scheduler can run N of them concurrently against a total budget. PRs [#308](https://github.com/kubeedge/ianvs/pull/308) and [#419](https://github.com/kubeedge/ianvs/pull/419) were paused because per-paradigm impact could not be guaranteed on shared global state; isolation removes that objection. Deliberately out of scope here.

**Bandwidth and latency emulation.** The cluster tier can apply `tc netem` per edge node to benchmark under realistic WAN conditions — the third system metric the project description names.

**Raising the node ceiling.** B12 is a Sedna-side constraint. A multi-`kind`-node topology, or an upstream change, would let the simulation match its "large-scale" ambition.

**Dependency cache.** A content-addressed venv cache keyed on requirements hash would cut per-test-case setup to near zero.

---

## 9. Why me

I read the code before writing the proposal. The 13 defects in §2.3 are not inferred from documentation — each was reproduced, several by executing the legacy functions directly, and the evidence script is included so a reviewer can reproduce them independently in under a minute. Two of them (B12, B7) appear not to have been reported before, and B12 means the feature's stated purpose did not currently work.

I also corrected a claim in my own favour: B6 is commonly described as a broken URL, and it is not — both branches resolve. I graded it Low and said why. I would rather submit a proposal with twelve solid findings and one honest downgrade than thirteen impressive-sounding ones.

The reference implementation exists in this repository, imports cleanly, passes 60 tests, scores 10.00/10 on `pylint core`, and demonstrably contains a real OOM crash — a 2 GiB allocation actually rejected by an enforced quota, not just measured after the fact — while preserving another test case's results. I would spend the mentorship completing it to community standard (cgroup v2 enforcement, cluster-tier end-to-end validation, real-paradigm PoC), not discovering whether it is possible.

This continues work already in flight against `kubeedge/ianvs`: open PR [#339](https://github.com/kubeedge/ianvs/pull/339) and issue [#338](https://github.com/kubeedge/ianvs/issues/338) fix `KeyError` crashes and a division-by-zero in the LLM inference benchmark; issues [#716](https://github.com/kubeedge/ianvs/issues/716)–[#719](https://github.com/kubeedge/ianvs/issues/719) catalogue and, in the referenced PRs, begin fixing structural breakage in the `Cloud_Robotics` examples (wrong directory references, invalid `requirements.txt` syntax, a missing vendored dependency). The pattern across all of it is the same one this proposal follows: read the actual code and dataset paths, reproduce the failure before describing it, and verify a fix by running it rather than by inspection alone.

---

## 10. Honest limitations

State these when asked; they are not weaknesses hidden elsewhere in the document, they are the boundary of what was actually tested in this environment.

- **cgroup v2 enforcement is not implemented**, only documented as the preferred future path. The mechanism actually enforcing the quota in this build is `RLIMIT_AS` plus a CPU affinity mask — real, live-verified (§5), but bounding virtual address space rather than resident memory, which can in principle reject a workload that would fit.
- **The cluster tier is written but not end-to-end validated.** No Docker or `kind` was available in the environment where this was built. `job_admin.py`'s ConfigMap-publishing contract (`worker.py`'s `IANVS_RESULT_CONFIGMAP` path) has never run against a live cluster. Validating it on Ubuntu LTS with current KubeEdge and Sedna is week 8–9 work, and the roadmap says so.
- **The process tier's fault containment was demonstrated with a synthetic memory-hog and a synthetic well-behaved test case**, run directly through `ProcessSandbox`, not through a real Ianvs paradigm end-to-end (no dataset was fetched for this verification pass). Wiring it to a real example is week 11.
- `verify_legacy_simulation.py` needs network access for B12 and B11 (it fetches the Sedna script). Use `--offline` to skip those two; the other 11 checks are purely local. Run with `--offline` against *this* (fixed) branch, 5 of the remaining 12 checks report `SKIP` rather than "not reproduced" — they inspect `simulation_system_admin.py` directly, which this proposal removes as superseded. This is expected, not a gap: the script is scoped to auditing a pristine legacy checkout, not this branch.

Nothing here claims a result that was not measured.

---

## 11. References

1. Ianvs architecture — <https://github.com/kubeedge/ianvs#architecture>
2. Issue #348, Simulator for Edge-cloud Collaborative AI — <https://github.com/kubeedge/ianvs/issues/348>
3. PR #35, OSPP 2022 simulation proposal — <https://github.com/kubeedge/ianvs/pull/35>
4. PR #39, OSPP 2022 simulation implementation — <https://github.com/kubeedge/ianvs/pull/39>
5. Issue #8, parallel processing of test cases — <https://github.com/kubeedge/ianvs/issues/8>
6. Issue #430, LLM example failures — <https://github.com/kubeedge/ianvs/issues/430>
7. Issue #460, MOT17 example failures — <https://github.com/kubeedge/ianvs/issues/460>
8. Sedna all-in-one installation — <https://github.com/kubeedge/sedna/blob/main/scripts/installation/all-in-one.sh>
9. KubeEdge — <https://github.com/kubeedge/kubeedge>
10. cgroup v2 — <https://docs.kernel.org/admin-guide/cgroup-v2.html>
11. `smaps_rollup` / PSS — <https://docs.kernel.org/filesystems/proc.html>
12. PR #526, competing "Ianvs Simulation Sandbox proposal — Phase 1" — <https://github.com/kubeedge/ianvs/pull/526>
