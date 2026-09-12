# Running a benchmark inside the Ianvs Simulation Sandbox

> **Audit notice (17 August 2026):** This document describes a proposed user
> experience, not functionality available in the supplied bundle or Ianvs
> `main`. The implementation modules are missing, and the supplied test file
> does not collect against the checkout. Do not run this quick start or quote
> its output as a measured result. See `SUBMISSION_GUIDE.md` for the verified
> state and the focused issue/PR prepared from the audit.

This example shows how to run an existing Ianvs benchmark with **environment
isolation** and **system-level metrics**, on a single machine, without root,
Docker or Kubernetes — and how to move the same job onto a real KubeEdge
cluster by changing one word.

## Why

Today every Ianvs test case runs in one shared Python process. That causes
three things:

- **Dependencies collide.** One `sys.path`, one set of versions.
- **State leaks.** Paths and environment variables set by one test case
  survive into the next, so failures depend on execution order.
- **One crash loses everything.** A single failing test case raises out of the
  run loop and discards the results of every test case that already finished.

And separately: Ianvs reports accuracy but never reports what an algorithm
*cost* to run. For edge AI, the device envelope is usually the binding
constraint.

The sandbox addresses both.

## Quick start

From the root of your Ianvs checkout:

```bash
pip install psutil
ianvs -f examples/simulation/sandbox_bench/benchmarkingjob.yaml
```

That is the whole change. The `sandbox` block in the config does the rest.

## What you get

```
+---------------------+--------------------+----------+----------------+---------------------+-------------+
| algorithm           | paradigm           | f1_score | peak_memory_mb | cpu_utilization_pct | wall_time_s |
+---------------------+--------------------+----------+----------------+---------------------+-------------+
| fpn_singletask      | singletasklearning |   0.8987 |        1842.31 |               187.4 |       421.7 |
+---------------------+--------------------+----------+----------------+---------------------+-------------+
```

`peak_memory_mb`, `cpu_utilization_pct` and `wall_time_s` sit in the same table
as `f1_score`, sortable the same way. Sorting by f1 descending and peak memory
ascending answers the question edge deployment actually asks: *which algorithm
is most accurate within my device budget?*

## The configuration

The entire `sandbox` block is optional. **Delete it, or set `enabled: false`,
and Ianvs runs exactly as it does today** — same code path, same results.

```yaml
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
```

### `mode`

| Value | What it does | Needs |
|---|---|---|
| `process` | Each test case gets its own worker process, transient Python runtime and CPU/memory envelope. | Nothing beyond Python |
| `cluster` | `kind` + KubeEdge `edgecore` + Sedna. Real edge-cloud topology. | Docker, kind, kubectl, Linux, 4 GiB free |
| `auto` | Cluster when the host supports it, process otherwise. | — |

`auto` degrades quietly. `cluster` does **not**: if you asked for a cluster and
the host cannot provide one, that is an error, because silently downgrading
would report cluster-topology results that were never measured on a cluster.

### `isolation`

- `venv` — a throwaway virtualenv per test case, seeded from your existing
  site-packages so heavy shared dependencies are not re-downloaded. If the
  example ships a `requirements.txt` next to its algorithm YAML, it is
  installed on top. This is how two examples with conflicting pins run in the
  same job.
- `none` — reuse the parent interpreter. Process and resource isolation only.
  Faster; use when you know dependencies do not conflict.

### `resources`

Kubernetes-style quantities: `2Gi`, `512Mi`, `1GB`, or a plain byte count.
Enforced by cgroup v2 where writable, otherwise `RLIMIT_AS` plus an affinity
mask. The host check at startup tells you which is in force.

`timeout` sends `SIGTERM` to the worker's process group, then `SIGKILL` after a
grace period.

### `fail_fast`

Default `false`, and that default is the point. A test case that crashes, hangs
or is OOM-killed is recorded as a failure with its metrics up to the moment it
died, and **the run continues**. Set `true` for the old abort-on-first-failure
behaviour.

## Fault containment, demonstrated

Two test cases, a 512 MiB quota, one deliberately allocating without bound:

```
--- memory-hog ---
  succeeded : False
  error     : test case was OOM-killed inside the sandbox. The declared memory
              quota was 512MiB; peak observed was 445MiB. The Ianvs process
              itself was unaffected.
  peak mem  : 444.57 MiB (quota 512.0 MiB)
  exit code : 137   oom_killed: True

--- well-behaved ---
  succeeded : True
  metrics   : {'accuracy': 0.913, 'f1_score': 0.887}
  peak mem  : 86.78 MiB (quota 512.0 MiB)
  cpu       : 39.77%   wall: 0.704s
```

The parent survived. One test case's results were preserved. On today's path,
the first case aborts the job and the second never runs.

## Moving to a real cluster

Change one word and supply a topology:

```yaml
sandbox:
  mode: "cluster"

simulation:
  cloud_number: 1
  edge_number: 2
  cluster_name: "ianvs-sim"
  kubeedge_version: ""   # empty => newest release
  sedna_version: ""
```

Ianvs then provisions `kind` + `edgecore` + Sedna, builds an image for each
algorithm, deploys it as a Job on a simulated edge node, list-watches the
result ConfigMap, and **tears the cluster down afterwards** — including when
the run fails or you press `Ctrl-C`.

> **Node ceiling.** The Sedna all-in-one backend supports at most **2 cloud
> worker nodes and 3 edge nodes** and aborts above that. Ianvs now rejects
> larger values at config-parse time with a clear message, rather than letting
> them fail deep inside the provisioning script. The process tier is not bound
> by this limit.

## Metric reference

| Config name | Column | Meaning |
|---|---|---|
| `peak_memory` | `peak_memory_mb` | Peak memory of the whole process tree. PSS where available, summed RSS otherwise — `memory_source` records which. |
| `mean_memory` | `mean_memory_mb` | Mean sampled memory |
| `cpu_utilization` | `cpu_utilization_pct` | Percentage of one core. `200` means two cores saturated. |
| `cpu_time` | `cpu_time_s` | Total user + system CPU seconds |
| `wall_time` | `wall_time_s` | Worker launch to worker exit |
| — | `memory_headroom_pct` | Percentage of the memory quota unused. **Negative means the algorithm exceeded its declared edge budget.** |

Use `metrics: ["all"]` to include everything collected.

### Why PSS rather than summed RSS

Summing RSS across a process tree double-counts shared pages — the CUDA
runtime, the interpreter's own text, model weights shared after a fork. For LLM
and VLA workloads with forked data loaders the inflation is substantial, and
wrong in a direction you cannot detect from the output. PSS divides each shared
page by the number of processes mapping it, so the sum is meaningful. Where the
kernel does not expose it, the fallback is recorded rather than assumed.

## Troubleshooting

**`no cgroup v2 write access`** — informational. Quotas fall back to
`RLIMIT_AS` plus an affinity mask. Note that `RLIMIT_AS` bounds *virtual*
address space, and ML runtimes often reserve far more than they touch, so a
tight limit may reject a workload that would have fit. Loosen it, or run where
cgroup v2 is writable.

**`psutil is not installed`** — `pip install psutil`. Without it you get
wall-clock only; the run still works.

**venv creation is slow** — set `isolation: none` if the example's dependencies
do not conflict with your environment.

**Cluster provisioning fails** — check `<workspace>/simulation/provision.log`.
The exact installer script that ran is saved alongside it with its SHA-256, so
the run is reproducible even if upstream changes.

## Verifying the fixes

To reproduce the defects this feature repairs, run against any Ianvs checkout:

```bash
python3 verify_legacy_simulation.py
```

It is read-only, installs nothing, provisions nothing, and exits with the
number of confirmed defects.
