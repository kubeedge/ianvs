# Example Restoration Proposals

This directory groups the Ianvs Example Restoration proposals by LFX Mentorship phase.

The proposals are organized chronologically so readers can understand how the Example Restoration work evolved from an initial restoration framework into a later, more concrete restoration plan for specific broken examples.

## Phases

| Phase | LFX Mentorship Term | Proposal | Original PR | Current File |
| --- | --- | --- | --- | --- |
| Phase 1 | 2025 Term 3 | Comprehensive Example Restoration for Ianvs | [#263](https://github.com/kubeedge/ianvs/pull/263) | [`phase-1-2025-term-3/example-restoration.md`](./phase-1-2025-term-3/example-restoration.md) |
| Phase 2 | 2026 Term 1 | Example Restoration of Ianvs | [#375](https://github.com/kubeedge/ianvs/pull/375) | [`phase-2-2026-term-1/Example_Restoration.md`](./phase-2-2026-term-1/Example_Restoration.md) |
| Phase 3 | 2026 Term 2 | KubeEdge Ianvs Example Classification CI Validation Framework | [#541](https://github.com/kubeedge/ianvs/pull/541) | [`phase-3-2026-term-2/proposal.md`](./phase-3-2026-term-2/proposal.md) |
| Phase 4 | 2026 Term 3 | KubeEdge Ianvs Example Environment Contract and Result Validity | [#986](https://github.com/kubeedge/ianvs/pull/986) | [`phase-4-2026-term-3/proposal.md`](./phase-4-2026-term-3/proposal.md) |

## Phase 1 — 2025 Term 3

Phase 1 contains the initial Example Restoration proposal introduced through PR #263.

It focuses on establishing the restoration direction for Ianvs examples, including motivation, methodology, roadmap, validation ideas, documentation improvement, and restoration-oriented benchmarking practices.

## Phase 2 — 2026 Term 1

Phase 2 contains the follow-up Example Restoration proposal introduced through PR #375.

It expands the restoration effort into a more concrete plan for restoring multiple broken Ianvs examples, including Cityscapes-Synthia Lifelong Learning examples, LLM-Agent, and LLM-Edge-Benchmark-Suite. It also discusses dependency evolution, API breakage, version incompatibilities, documentation updates, setup improvements, and CI/CD validation.

## Phase 3 — 2026 Term 2

Phase 3 contains the CI validation and classification follow-up introduced through PR #541.

It shifts the Example Restoration effort toward automated example health classification, pull-request validation, and sustainable maintenance workflows. It also uses `examples/llm_simple_qa` as the first concrete restoration target verified by the proposed CI framework.

## Phase 4 — 2026 Term 3

Phase 4 contains the environment contract and result validity follow-up introduced through PR #986.

It shifts the Example Restoration effort from *does an example run?* to *does an example produce a
result that describes the hardware it ran on?* It introduces a declared environment contract in
`testenv.yaml`, resolution and verification in `core/`, environment provenance in the benchmark
report, and replaces the validator's textual CUDA-only heuristic with checks against the
declaration. It uses the emulated-`bfloat16` case documented in issue #888 as the proven failure
mode motivating the work.

## Notes for Maintainers

- Keep future Example Restoration proposals under this directory.
- Add new proposals as additional phase folders, for example `phase-3-<term-name>/`.
- Update this README whenever a new phase is added.
- Prefer lowercase directory names for new phase folders.
- Avoid changing the original proposal text unless required for broken links, path updates, or formatting consistency.
