# KubeEdge Ianvs Example Environment Contract and Result Validity

## Background

Ianvs is a distributed benchmark toolkit. Its output is measurements, and a measurement is only
worth as much as the environment claim behind it.

Three phases of Example Restoration have progressively answered the question *does this example
run?*

- **Phase 1** (2025 Term 3) established the restoration direction.
- **Phase 2** (2026 Term 1) restored specific broken examples and addressed dependency evolution,
  API breakage, and version incompatibility.
- **Phase 3** (2026 Term 2) built the CI validation framework: an example inventory, tiered
  validation, PR-impact classification that distinguishes a regression from a pre-existing failure,
  and published example health.

That machinery works. An example that fails to start is now visible, classified, and attributable.

Phase 4 addresses the case the machinery cannot yet see: an example that **starts, completes, and
reports numbers that do not describe the hardware they were measured on**.

Issue [#888](https://github.com/kubeedge/ianvs/issues/888) documents one instance. Below CUDA
compute capability 8.0, PyTorch emulates `bfloat16` rather than refusing it. Two LLM examples select
`bfloat16` unconditionally on any CUDA device. Measured on a GTX 1080 (CC 6.1), a 4096×4096 matmul
in `bfloat16` runs 1.85× slower than the same matmul in float32 — the dtype chosen as the fast path
is the slowest one available on that hardware. No exception is raised. The benchmark completes and
publishes the result.

This is not a rare corner. It is the predictable consequence of a gap in how Ianvs represents its
environment: examples state what they need in prose, resolve devices independently in code, and
publish results that carry no record of what actually executed.

## Goals

1. Give every example a machine-checkable statement of the environment it requires, instead of
   requirements implied by a hardcoded dtype or a transitive dependency.
2. Verify that statement before a run produces numbers, and refuse the run with an actionable
   message when the environment cannot satisfy it.
3. Record what actually executed alongside the result, so a measurement obtained through a fallback
   is never indistinguishable from a native one.
4. Replace the validator's text-heuristic hardware check with checks against the declared contract,
   and make a violation block a pull request rather than warn about it.
5. Extend CI coverage to the Python versions and upstream dependency updates that silently change
   these guarantees.

## Problem Statement

### 1. A completed run is not a validated result

Phase 3's result levels answer whether a check completed: `PASS`, `FAIL`, `ERROR`, `WARNING`,
`SKIP`. Its smoke validation answers whether an example executes. Neither asks whether the numbers
the example produced are attributable to the hardware named in the report.

For a toolkit whose deliverable is a leaderboard, the difference matters. A crash is visible and
gets fixed. A silently degraded measurement is published.

### 2. The existing hardware check cannot see the cases that matter

Phase 3 already reserves `Failed: Hardware assumption` as a failure cause
(`.github/workflows/validator/services/regression_detector.py:964`) and publishes
`Requires GPU or special hardware` as an example status
(`services/report_generator.py:80`). The vocabulary exists.

The detection behind it is a textual scan. `static_validator.py:474`,
`_check_cuda_only_assumptions()`, searches each Python file for

```python
CUDA_ONLY_RE = re.compile(
    r"(?i)(?:device\s*=\s*[\"']cuda[\"']|torch\.device\([\"']cuda[\"']\)|\.cuda\()"
)
```

and then exempts the file if it contains a fallback:

```python
has_fallback = "torch.cuda.is_available()" in text and "cpu" in text
```

Running that check against the three sites documented in #888 gives:

| File | Regex matches | Exempted as "has fallback" | Selects `bfloat16` | Result |
| --- | --- | --- | --- | --- |
| `.../query-routing/models/eagle_llm.py` | yes | no | yes | `WARNING`, does not block |
| `.../algorithms/block/drafter.py` | **no** | — | **yes** | **not seen** |
| `.../algorithms/block/verifier.py` | **no** | — | **yes** | **not seen** |
| `.../query-routing/models/huggingface_llm.py` | yes | **yes** | no | skipped |
| `.../common/runtime.py` | no | — | no | not seen |

Three observations follow.

**The pattern does not cover dtype selection.** `drafter.py:64` and `verifier.py:109` read

```python
self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
```

`device == "cuda"` is a comparison, not an assignment, so `CUDA_ONLY_RE` does not match it. The two
files that make the most consequential hardware assumption in the repository are invisible to the
check designed to find hardware assumptions.

**The exemption uses the predicate that #888 disproved.** A file is considered safe when it mentions
`torch.cuda.is_available()`. That function answers whether a CUDA device exists. It says nothing
about which generation, and dtype support is a property of the generation. An example can resolve
its device impeccably and still select an emulated dtype on it.

**The finding does not block.** `_check_cuda_only_assumptions` reports at `WARNING`, and per the
classification policy warnings never fail a pull request.

None of this is a defect in Phase 3. A textual scan is the correct first approximation when there
is nothing to check against. Phase 4's argument is that there should be something to check against.

### 3. Declared configuration is not always honoured

`use_gpu` was declared on `TestEnv` and parsed from `testenv.yaml`, but
[#765](https://github.com/kubeedge/ianvs/issues/765) showed it was read in exactly one paradigm,
applied after the algorithm module had already been instantiated, and had no effect at all when set
to `false`. [#767](https://github.com/kubeedge/ianvs/pull/767) fixes it.

The wider pattern is visible in the tree today:

```console
$ grep -rE "[\"']cuda[\"']|\.cuda\(\)|cuda:0" examples/ --include=*.py | wc -l
160
$ grep -rlE "[\"']cuda[\"']|\.cuda\(\)|cuda:0" examples/ --include=*.py | wc -l
73
$ find examples -name testenv.yaml | wc -l
41
$ grep -rl "use_gpu" examples --include=testenv.yaml | wc -l
2
```

160 hardcoded CUDA references across 73 files, against 2 of 41 `testenv.yaml` files that state any
device intent. Device selection is overwhelmingly decided inside example code, beyond the reach of
both configuration and validation. One example has built its own resolver rather than asking core
(`.../common/runtime.py:519`), using the same insufficient predicate.

### 4. Results carry no environment provenance

A benchmark report records metrics, algorithm, and paradigm. It does not record the dtype that
executed, the device generation, or whether a requested configuration was satisfied or silently
substituted. A reader comparing two leaderboard rows cannot tell whether they were produced under
comparable conditions.

## Proposal

Introduce an **environment contract** for Ianvs examples, with four stages.

| Stage | Question | Where it lives |
| --- | --- | --- |
| **Declare** | What does this example require? | `testenv.yaml` |
| **Resolve** | What does this machine actually provide? | `core/common/device.py` |
| **Verify** | Can the requirement be met here? | before the run produces numbers |
| **Record** | What actually executed? | benchmark report |

Each stage is small. The value is in closing the loop: today a requirement is prose, resolution is
per-example code, verification is a regex, and there is no record.

The first two stages are partly implemented already, through work done while preparing this
proposal: [#957](https://github.com/kubeedge/ianvs/pull/957) adds `device_profile()` and
`supports_bf16()` in `core/common/device.py`, an optional `min_compute_capability` key in
`testenv.yaml`, verification in `TestEnv.prepare()`, and a fallback warning at every affected model
load. Phase 4 extends that from one property to a contract, and connects it to the validator and
the report.

## Scope

### In Scope

- A contract vocabulary in `testenv.yaml` covering device class, compute capability, device memory,
  and required dtypes.
- Resolution and verification in `core/`, extending `core/common/device.py`.
- Environment provenance in the benchmark report, including the dtype actually used and any
  substitution that occurred.
- Replacing `_check_cuda_only_assumptions` with contract-aware checks in the Phase 3 validator, and
  promoting an undeclared, detected requirement from `WARNING` to a blocking result.
- Declaring contracts for the examples the checks flag, working outward from the LLM examples where
  the failure mode is proven.
- Extending the CI matrix to the Python versions the project claims to support, and adding a
  scheduled job that surfaces upstream dependency drift affecting these guarantees.
- Documentation: a contributor-facing description of the contract, updated "Required Resources"
  sections, and a debugging playbook entry for environment-contract failures.
- A blog post for the KubeEdge website on why a benchmark that cannot describe its own environment
  reports numbers it cannot defend.

### Out of Scope

- Restoring example logic unrelated to environment assumptions. Phase 2 owns that work and it
  continues independently.
- Reworking Phase 3's inventory, tiering, or PR-impact classification. Phase 4 adds checks inside
  that framework and does not modify its policy.
- Procuring GPU runners. Every check proposed here is either static or runs on the CPU runners CI
  already has; the contract is verified against whatever device is present, and a declared
  requirement that cannot be exercised is reported rather than silently passed.
- Performance optimisation of examples. Phase 4 makes measurement conditions explicit; it does not
  try to improve them.
- Changing published leaderboard numbers retroactively. Existing results stay; new results gain
  provenance.

## Target Users

### Maintainers

Today a maintainer reviewing a pull request that touches an example has no mechanical way to tell
whether it introduced a hardware assumption. The `WARNING` produced by the current check does not
block, does not distinguish a new assumption from an old one in the dtype case, and does not fire at
all for two of the three sites in #888. After Phase 4, an undeclared assumption introduced by a pull
request fails CI with the file, the construct, and the missing declaration named.

A second cost falls on maintainers today when a benchmark result is questioned. Without provenance
there is no way to answer "was this measured under the conditions the report claims" except by
re-running it.

### Contributors

A contributor adding an example currently learns its hardware requirements by running it and seeing
what breaks, and has no place to record what they learned. The failure they hit may be an exception,
or it may be a number that is quietly wrong. After Phase 4 the requirement is a few lines of yaml,
checked at the point it is violated, with a message that names the cause.

For the contributor whose pull request is blocked by a pre-existing failure in an unrelated example,
Phase 3's classification already prevents that, and Phase 4 adds no new blocking for existing debt.

### Benchmark readers

The people who read Ianvs leaderboards are the users this ultimately serves. A published number
whose measurement conditions are unrecorded cannot be compared with confidence to another. Recording
provenance is what makes two rows comparable, or visibly not comparable.

Edge-AI practitioners are disproportionately affected, because they are the population most likely
to be running exactly the hardware where substitution happens silently.

## Design Details

### 1. Contract vocabulary

```yaml
testenv:
  environment:
    device: cuda            # cuda | cpu | any
    min_compute_capability: "8.0"
    min_device_memory_gb: 16
    dtypes: [bfloat16]      # dtypes the example intends to execute natively
```

Every key is optional and absence means "unconstrained", so the 39 of 41 `testenv.yaml` files that
declare nothing today keep their current behaviour unchanged. This is the same compatibility rule
#767 established for `use_gpu`, where an unset key leaves the environment untouched.

`dtypes` is the key that closes #888: an example that intends to run in `bfloat16` says so, and the
declaration is what the validator and the runtime both check against.

### 2. Resolution and verification

`core/common/device.py` already reports the resolved device, its compute capability, its memory,
and whether `bfloat16` executes natively. Phase 4 extends it to evaluate a whole contract rather
than a single property, and reports one of three outcomes:

| Outcome | Meaning | Behaviour |
| --- | --- | --- |
| **Satisfied** | The environment meets every declared requirement | run proceeds, provenance recorded |
| **Substituted** | A requirement cannot be met but a defined fallback exists | run proceeds, substitution recorded and warned at every occurrence |
| **Unsatisfiable** | A requirement cannot be met and no fallback exists | run refused before any measurement |

The distinction between *substituted* and *unsatisfiable* is the design's centre. Refusing every
mismatch would make Ianvs unusable on ordinary hardware, which contradicts an edge-AI benchmark's
purpose. Publishing a substituted result as though it were native is the bug this phase exists to
remove. Recording the substitution keeps both properties.

Verification runs before a measurement is produced. `TestEnv.prepare()` is called from
`core/cmd/obj/benchmarkingjob.py:89`, ahead of `build_testcases()`, and is early enough that nothing
is downloaded for a run that cannot proceed.

### 3. Validator: from heuristic to contract

`_check_cuda_only_assumptions` is replaced by two checks.

**Contract completeness.** An example whose sources reference a hardware-dependent construct must
declare it. The detection set is extended beyond assignment syntax to cover the cases the current
regex misses — dtype literals such as `torch.bfloat16`, comparisons such as `device == "cuda"`,
`device_map="auto"`, and dependencies with their own floors such as `vllm`. The check is static and
needs no GPU.

**Contract consistency.** A declared contract must not contradict the code. An example declaring
`device: cpu` while selecting `bfloat16` on CUDA is reported.

Both produce a blocking result when the finding is newly introduced by a pull request, and are
classified as pre-existing under the Phase 3 policy otherwise, so historical debt stays visible
without blocking unrelated work. The existing `Failed: Hardware assumption` cause label gains a
detector that actually produces it.

The exemption predicate is removed. A file is not judged safe by mentioning
`torch.cuda.is_available()`; it is judged against what it declared.

### 4. Provenance in the report

The benchmark report gains an environment section recording the resolved device and capability, the
dtype actually executed, and any substitution that occurred. Two leaderboard rows then carry enough
information to say whether they are comparable.

### 5. A worked example

`examples/cloud-edge-speculative-decoding-benchmark` selects `bfloat16` in its block algorithm and
`float16` in its ar algorithm, from the same insufficient predicate, and declares neither.

Under the contract its `testenv.yaml` states what it intends:

```yaml
testenv:
  environment:
    device: cuda
    dtypes: [bfloat16]
```

The three stages then behave as follows.

On an A100 (CC 8.0): the contract is **satisfied**. The run proceeds and the report records
`device: NVIDIA A100, capability: 8.0, dtype: bfloat16, substitution: none`.

On a GTX 1080 (CC 6.1): `bfloat16` has no native path. The declared fallback applies, so the outcome
is **substituted**. The run proceeds, a warning is emitted at each affected model load, and the
report records `dtype: float32, substitution: bfloat16 -> float32 (capability 6.1 < 8.0)`. A reader
comparing this row with the A100 row can see immediately that they are not measuring the same thing.

On a CPU-only machine: the declared device cannot be provided and no fallback is declared for it, so
the outcome is **unsatisfiable**. The run is refused before the dataset is fetched, naming the
requirement and what was found.

In the validator, the same declaration makes the completeness check pass for `drafter.py` and
`verifier.py`, which the current heuristic does not examine at all. Removing the `dtypes` line while
leaving the code unchanged makes the check fail — which is the property that keeps the declaration
honest.

### 6. CI matrix

The lint workflow runs Python 3.7, 3.8 and 3.9. `setup.py` declares `python_requires=">=3.6"` while
several examples require 3.10 or newer, and the validator is not exercised across versions at all.
Phase 4 aligns the matrix with what the project claims to support and adds a scheduled run so that
an upstream release changing a dtype default or a capability floor is surfaced by CI rather than by
a contributor with an older card.

## Relationship to Prior Phases and Concurrent Work

This proposal builds on Phase 3 and depends on it. The inventory, the tiering, the PR-impact
classification and the reporting are reused unchanged; Phase 4 supplies checks to run inside them
and a contract for those checks to consult.

Concurrent work in the tree that touches adjacent ground, and how this differs:

- [#535](https://github.com/kubeedge/ianvs/issues/535) / [#536](https://github.com/kubeedge/ianvs/pull/536)
  give `government_rag` local `cuda/mps/cpu` detection. That is device presence, resolved locally.
  The contract makes the same decision once, centrally, and adds the generation.
- Review discussion on #767 proposed a `get_device()` utility in core for examples to consult. That
  is the *Resolve* stage of this contract, and #957 implements it.
- [#845](https://github.com/kubeedge/ianvs/issues/845) proposes a pytest suite for `core/`. Phase 4
  adds tests for the modules it touches and would adopt whatever structure that work establishes.
  [#890](https://github.com/kubeedge/ianvs/pull/890) contributes the first tests under `tests/`.

Where this proposal's scope overlaps with work another contributor has begun, the intent is to
consume it rather than duplicate it.

## Functional Requirements

**FR-1 Contract declaration.** An example may declare an `environment` block in `testenv.yaml`
covering device class, minimum compute capability, minimum device memory, and intended dtypes. Every
key is optional; an absent block means unconstrained.

**FR-2 Backward compatibility.** An example that declares nothing behaves exactly as it does before
this change. This is verified by test, not by inspection.

**FR-3 Environment resolution.** Core reports the resolved device, its compute capability, its
memory, and which dtypes execute natively, as plain data, without adding a torch dependency to
`core`.

**FR-4 Contract evaluation.** A declared contract evaluates to satisfied, substituted, or
unsatisfiable. An unsatisfiable contract refuses the run before any measurement is produced, naming
the requirement, the detected environment, and the component imposing the requirement.

**FR-5 Substitution reporting.** A substitution warns at every occurrence rather than once per
process, because a benchmarking job runs its test cases in one process and a suppressed repeat hides
every affected measurement after the first.

**FR-6 Result provenance.** A benchmark report records the resolved device and capability, the dtype
actually executed, and any substitution that occurred.

**FR-7 Contract completeness check.** The validator detects hardware-dependent constructs including
dtype literals, device comparisons, `device_map`, and dependencies with their own capability floors,
and reports an example that uses one without declaring it.

**FR-8 Contract consistency check.** The validator reports a declaration that contradicts the code
it describes.

**FR-9 PR-impact behaviour.** A newly introduced contract violation blocks the pull request. A
pre-existing one is reported without blocking, following the Phase 3 classification policy
unchanged.

**FR-10 Version coverage.** The validator runs across the Python versions the project supports, and
a scheduled job surfaces upstream dependency changes that alter dtype defaults or capability floors.

## Roadmap

The term runs September 7 to November 27, 2026, with midterm evaluation on October 20.

### Early Phase — Sep 7 to Oct 4

- Finalise the contract vocabulary with maintainers, including which keys are worth declaring and
  which are better inferred.
- Extend `core/common/device.py` from single-property checks to whole-contract evaluation with the
  three outcomes.
- Land the `environment` block in the `testenv.yaml` schema with backward-compatible defaults.
- Tests for every outcome across capability tiers, with no GPU required.

Deliverable: an example can declare a contract and a run is refused, substituted, or satisfied
accordingly.

### Middle Phase — Oct 5 to Nov 1

- Replace `_check_cuda_only_assumptions` with the completeness and consistency checks.
- Wire the checks into the Phase 3 tiers and classification, and promote newly introduced findings
  to blocking.
- Add environment provenance to the benchmark report.
- Declare contracts for the LLM examples where the failure mode is proven, then work outward
  through the examples the checks flag.
- Midterm deliverable, October 20: a pull request that introduces an undeclared hardware assumption
  is blocked by CI, and a run that substitutes a dtype says so in its report.

### Late Phase — Nov 2 to Nov 27

- Extend the CI matrix across supported Python versions and add the scheduled upstream-drift job.
- Complete contract declarations across the remaining flagged examples.
- Documentation: contributor guide section, updated "Required Resources" sections, debugging
  playbook entry for contract failures.
- Blog post for the KubeEdge website.
- Final deliverable, November 24: the contract is documented, enforced, and declared across the
  examples that need it.

## Acceptance Criteria

1. An example can declare an environment contract in `testenv.yaml`, and an example that declares
   nothing behaves exactly as it does today.
2. A run whose contract cannot be satisfied is refused before any measurement, with a message naming
   the requirement, the detected environment, and the component imposing the requirement.
3. A run that substitutes a fallback completes, warns at every occurrence, and records the
   substitution in its report.
4. `drafter.py`, `verifier.py` and `eagle_llm.py` — the three sites in #888 — are all detected by the
   validator, where two of the three are invisible to it today.
5. A pull request that introduces an undeclared hardware assumption is blocked; a pre-existing one is
   reported without blocking.
6. `Failed: Hardware assumption` is produced by a check rather than reserved as a label.
7. The validator runs across the project's supported Python versions.
8. Contract documentation exists for contributors, and the affected "Required Resources" sections
   state capability requirements rather than only quantities.

## Risk Analysis

**Contract keys proliferate into an unmaintainable schema.** Every key must be justified by a
failure mode observed in the repository. The initial set covers device class, capability, memory and
dtypes because each corresponds to a documented case. Additions require the same evidence.

**Promoting warnings to blocking disrupts contributors.** Phase 3's classification already separates
newly introduced findings from pre-existing ones. Only the former block. Contract declaration for
existing examples proceeds incrementally, so the flagged set shrinks rather than gating unrelated
work.

**Declared floors make examples unrunnable on modest hardware.** This is why *substituted* exists as
a distinct outcome from *unsatisfiable*. A hard floor is declared only where no fallback is possible,
such as `vllm` on pre-Turing hardware. Where a fallback exists, the run proceeds and says so.

**Detection produces false positives.** Static detection of hardware dependence is heuristic by
nature, which is why the response is "declare it" rather than "fix it". A contract that says
`device: cpu` resolves a false positive in one line, and the declaration is itself useful.

**No GPU runners exist in CI.** Every check proposed here is static or runs on CPU. The contract is
verified against whatever the runner provides; a requirement that cannot be exercised is reported as
unverifiable rather than passed silently. Verification on real GPUs remains manual, as it is today.

**Core changes may be required beyond the planned scope.** #767 already showed a defect in the
existing device path. Further work in `core/` will be proposed as separate, reviewable changes rather
than folded into large ones.

**Overlap with concurrent contributor work.** Several people are working on adjacent device-selection
and CI topics. The mitigation is to reference and consume rather than duplicate, as the Relationship
section describes, and to keep coordinating in the issue threads.

## Future Work

- Extend the contract beyond device properties to dataset provenance and model revision pinning, so
  a leaderboard row identifies its inputs as precisely as its hardware.
- Publish environment provenance in the leaderboards, so comparability is visible to readers rather
  than only recorded.
- Reference contracts per example class, so a new example starts from a template rather than an
  empty block.

## Summary

Phases 1 through 3 made Ianvs examples runnable and made their health visible. The remaining gap is
that running successfully and measuring meaningfully are different properties, and Ianvs currently
verifies only the first.

Phase 4 closes that gap with one idea applied consistently: an example declares the environment it
needs, core resolves what the machine provides, the run is verified before it produces numbers, and
the report records what actually executed. The validator stops guessing from regular expressions and
starts checking against a declaration.

The concrete evidence that this is needed already exists in the tree: a benchmark that reports
emulated arithmetic as a measurement, and a hardware check that cannot see two of the three files
responsible.
