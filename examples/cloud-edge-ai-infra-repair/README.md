# Cloud-edge AI infrastructure repair benchmark

A Python CLI example for evaluating model-assisted diagnosis and bounded repair
of AI inference services. It uses real model inference and a dedicated KubeEdge
fixture; it does not introduce a separate platform. See the
[proposal](../../docs/proposals/scenarios/cloud-edge-ai-infra-repair/proposal.md).

## Scenarios

| ID | Fault | Injection and repair |
| --- | --- | --- |
| S1 | Service configuration | Load a nonexistent model path; restore the approved path. |
| S2 | GPU resource configuration | Limit the target process's CUDA allocator to 2% of GPU memory and observe an actual loading OOM; restore the approved 40% upper bound. |
| S3 | Model artifact | Truncate a safetensors file in a private copy; restore verified original bytes. |
| S4 | Heterogeneous environment | Run an ARM64 gateway image on the AMD64 edge node; restore the identical release built for AMD64. |
| S5 | Cloud-edge consistency | Block the dedicated control link in both directions, publish R2 while the edge retains R1; reconnect and verify R2 reaches the actual edge configuration. |

R1 lacks the required streaming endpoint; R2 implements it. S5 therefore checks
observable serving behavior as well as configuration. S2 changes only the
per-process allocator budget, not the host driver or other processes' memory.

## Environment and topology

Use Linux/WSL, Python 3.10+, Docker, kubectl and a CUDA GPU for all five scenarios.
The lightweight tests use only the Python standard library. Install the example's
`requirements.txt` for real inference and Ianvs Rank export. The reference agent's
vLLM environment is managed separately. The tested development environment uses
Python 3.12, KubeEdge 1.23.1 and a Kubernetes 1.32 control plane.

The reproducible fixture has two dedicated Docker nodes, CloudCore and EdgeCore.
A gateway on the edge node forwards to a separate GPU worker bound to their
private Docker bridge. Validation always traverses the edge gateway. All
components share one physical WSL host: this tests real KubeEdge reconciliation,
container execution and CUDA failures, but is not a geographically distributed
edge or GPU-in-container benchmark. The gateway uses host networking; CNI routing
is not part of the measured inference path.

Model snapshots are local, complete Hugging Face directories. Original artifacts
are read-only to the harness. Every run copies them into a disposable directory.
Do not commit weights, credentials, kubeconfigs or raw workspace contents.

Run commands from the repository root. Substitute paths and Python executables
for your environment. Give the host enough available memory for the independent
agent, target worker and cluster; heavy paging invalidates latency experiments.

```bash
python examples/cloud-edge-ai-infra-repair/scripts/benchmark.py preflight \
  --output workspace/infra-repair/preflight.json
python -m unittest discover -s examples/cloud-edge-ai-infra-repair/tests -v
python .github/workflows/validator/validation_runner.py --static \
  --inventory examples/cloud-edge-ai-infra-repair/validation-inventory.yaml \
  --example ai_infra_repair_pilot
python examples/cloud-edge-ai-infra-repair/scripts/provision.py create
python examples/cloud-edge-ai-infra-repair/scripts/provision.py status
```

The example-local validation inventory lets the existing validator check this
draft scenario without changing repository-wide CI configuration. Registering
it in the shared inventory and enabling CI remain subject to upstream review.

Provisioning pins Keink and KubeEdge source revisions, builds separate AMD64 and
ARM64 gateway images, and imports them sequentially into the dedicated edge
containerd. Network access is needed for Git, Go and Docker dependencies. Existing
clusters are not selected or modified. The fixture's kubeconfig is always passed
explicitly; commands never rely on the user's current context. `images` rebuilds
only the fixture gateway images, and `delete` removes only the two owned nodes.

## Independent reference model

Download `Qwen/Qwen2.5-1.5B-Instruct` into your own model cache, then start its
independent service. Qwen2.5-0.5B-Instruct can be evaluated separately as a control.
Neither model's development results should be mixed with another model's cohort.

```bash
python examples/cloud-edge-ai-infra-repair/scripts/local_agent.py start \
  --python /path/to/vllm-env/bin/python --model-dir /path/to/agent-snapshot \
  --served-model-name Qwen2.5-1.5B-Instruct --port 8002 \
  --gpu-memory-utilization 0.28 --output workspace/infra-repair/local-agent-1.5b
```

Use `--foreground` when a supervisor needs to retain the WSL command. Use `status`
or `stop` with the same `--output` directory to inspect/stop only the recorded
process. The helper uses Linux loopback `127.0.1.1`; some WSL mirrored-network
setups route `127.0.0.1` to Windows. Requests bypass environment proxies and refuse
redirects. Under WSL the helper sets `VLLM_USE_V2_MODEL_RUNNER=0` and
`VLLM_USE_FLASHINFER_SAMPLER=0`, without patching installed vLLM. See the
[vLLM environment documentation](https://docs.vllm.ai/en/latest/configuration/env_vars/).

`--agent triage` collects three fixed read-only observations. The model chooses a
root-cause code and a supporting observation ID; an approved runbook maps that
choice to one bounded repair. This is a model-assisted runbook baseline, not an
unrestricted autonomous agent. `--agent local` also provides a bounded iterative
JSON-action adapter. Neither receives scenario IDs or evaluator answers. The diagnostic runbooks cover
the known fault taxonomy; development uses separate pilot executions of these
fault types. Repeated acceptance runs do not establish generalization to unseen faults.

## Run and verify

```bash
export IANVS_TARGET_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct-snapshot
python examples/cloud-edge-ai-infra-repair/scripts/benchmark.py run \
  --config examples/cloud-edge-ai-infra-repair/pilot.yaml
# Run the separate rollback cohort with the identical workload configuration.
python examples/cloud-edge-ai-infra-repair/scripts/benchmark.py run \
  --config examples/cloud-edge-ai-infra-repair/pilot.yaml --agent failure
python examples/cloud-edge-ai-infra-repair/scripts/benchmark.py acceptance \
  --reference workspace/infra-repair/batch-REFERENCE \
  --rollback workspace/infra-repair/batch-ROLLBACK \
  --output workspace/infra-repair/acceptance.json
```

For repeated WSL runs, use `--output /home/USER/ianvs-repair-runs` on the Linux
filesystem; copying and hashing model workspaces through `/mnt/c` or `/mnt/e`
can be substantially slower. After cleanup, copy the small evidence directories
back to the ignored repository workspace for screenshots.

The YAML is consumed by this example's CLI, not `ianvs -f`. Explicit flags override
YAML settings. It is a candidate experiment contract, not a statement that any
acceptance target has already passed. Each batch preserves all failed attempts.
Formal reporting requires five scenarios with three independent runs each, a
homogeneous agent/workload/artifact contract, and a separate complete rollback
cohort. `acceptance` exits unsuccessfully if any required check fails.

The target uses three fixed prompts, greedy decoding, at most 16 generated tokens
and concurrency one. Sampling repeats this workload; warmup is excluded from SLO.
Model precision, samples, deadlines and thresholds are fixed before each cohort
and remain unchanged during repair. Correctness means exact token regression
against the healthy model, not broad task accuracy. Client-side streaming measures
first generated token, P95 end-to-end latency, request success and token throughput.

Diagnosis requires the right cause/object and an authentic observation supporting
that cause. A correct code with an unrelated stack frame or invented evidence
fails. Repair requires artifact identity, token regression and SLO within budget.
Calibration controller results test the harness only; they never count as agent
success. Failure controller results exercise rollback only.

## Safety and recovery

The agent has no shell, filesystem, Docker or Kubernetes credential tool. The
policy gateway accepts only enumerated actions with exactly `resource: target`,
bounded by call/time budgets. Audit decisions are persisted before mutation.
`--approval interactive` additionally requires an operator to approve the exact
named action in the terminal and rechecks the deadline before execution. Policy
approval is the reproducible default. Arbitrary third-party Python agent code is
not sandboxed or loaded into the evaluator.

Failed repair triggers rollback and independent inference verification. Cleanup
removes the model copy, target process, owned namespace and exact experiment
firewall rules. Cleanup/rollback failure stops the suite. SIGINT attempts rollback;
a killed controller or machine failure requires conservative recovery:

```bash
python examples/cloud-edge-ai-infra-repair/scripts/benchmark.py recover \
  workspace/infra-repair/batch-REPLACE/RUN-ID
python examples/cloud-edge-ai-infra-repair/scripts/provision.py delete
```

Recovery validates process, namespace and dedicated-cluster ownership. Emergency
cleanup does not prove restored inference and is never counted as a verified
rollback. Docker provides the out-of-band control path when the cloud-edge link
is disconnected.

## Reports and screenshots

Each batch writes JSON and CSV; each run preserves configuration, baseline,
artifact identity, fault evidence, model response, tool audit and result. Reports
cover repair rate, evidence-based diagnosis, elapsed/phase times, tool/token
usage, denied/unauthorized operations and inference SLO. Missing measurements are
null; timed-out model calls without returned usage do not contribute invented
zero-token counts. Success-only repair times are reported separately from failed
attempt durations.

```bash
python examples/cloud-edge-ai-infra-repair/scripts/benchmark.py report \
  workspace/infra-repair/batch-REPLACE --ianvs-rank
python examples/cloud-edge-ai-infra-repair/scripts/render_evidence.py
```

The Rank adapter reuses `core.storymanager.rank` and produces `rank/all_rank.csv`
and `rank/selected_rank.csv`. This is local Ianvs reporting, not an official
community leaderboard or a new core learning paradigm.

Evidence pages use ordinary `EXP-001` numbering, including failed experiments.
`experiment-index.json` maps numbers to original records. Documentation screenshots
are captures of these actual-result report pages, not native terminal captures.
Internal artifact hashes verify integrity; they are not experiment labels.

The [FORMAL-001 contributor report](../../docs/proposals/test-reports/testing-cloud-edge-ai-infra-repair.md)
contains a complete local 15-run reference cohort and 15-run rollback cohort,
including failed diagnoses, the failed repair, CSV/JSON/Rank exports and screenshots.
It passed the stated local acceptance checks; upstream review and independent
community reproduction are still pending.
