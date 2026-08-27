# Example Status Directions

Inventory status and published example health are related but distinct.
Inventory status is maintained policy input for each benchmark unit. Published
health combines that inventory state with the latest complete T2/T3 validation
evidence and aggregates it by top-level example.

The [Example Classification Matrix](../../examples/README.md) is the
maintainer-facing published summary. Detailed CI artifacts remain the evidence
behind each status.

See also:

- [Validation rules](validation_rules.md)
- [Classification policy](classification_policy.md)
- [Local validation](local_validation.md)

## Inventory lifecycle status

Each inventory entry has a lifecycle `status`. Only `active` entries are
eligible to execute dynamic stages; selected inactive entries are reported as
`SKIP`. Lifecycle status describes a benchmark unit's maintained state and is
not itself a validator check result or a published top-level badge.

See [inventory rules](validation_rules.md#inventory-rules) and
[validation concepts](validation_rules.md#validation-concepts) for the
authoritative contract.

## Published status badges

| Badge | Meaning |
| --- | --- |
| ![Runnable](https://img.shields.io/badge/status-Runnable-brightgreen) | The top-level example group passed its latest published broad validation. |
| ![Broken](https://img.shields.io/badge/status-Broken-red) | At least one validated benchmark unit in the top-level example group has a blocking failure. |
| ![CI/CD onGoing](https://img.shields.io/badge/status-CI%2FCD%20onGoing-lightgrey) | The top-level example group contains unvalidated inventory or has no published result. |
| ![Example onGoing](https://img.shields.io/badge/status-Example%20onGoing-yellow) | Development of the example itself is still in progress. |
| ![Requires external dataset or model download](https://img.shields.io/badge/status-Requires%20external%20dataset%20or%20model%20download-blue) | Normal CI cannot obtain a required dataset or model. |
| ![Requires GPU or special hardware](https://img.shields.io/badge/status-Requires%20GPU%20or%20special%20hardware-orange) | The required hardware is unavailable on normal validation runners. |
| ![Quarantined](https://img.shields.io/badge/status-Quarantined-8a2be2) | Dynamic validation is intentionally disabled pending repair or a maintainer decision. |
| ![Known issue](https://img.shields.io/badge/status-Known%20issue-critical) | A failure has been triaged and accepted temporarily. |

## Aggregation precedence

Inventory status is policy input, while the current published snapshot is derived from the top-level example's validation evidence with this precedence:

| Evidence or inventory state | Published status |
| --- | --- |
| Any matched benchmark unit has a blocking `FAIL` or `ERROR` | `Broken` |
| A unit is `quarantined` | `Quarantined` |
| A unit is `known_issue` or `known issue` | `Known issue` |
| A unit is `requires_hardware` or `hardware` | `Requires GPU or special hardware` |
| A unit is `onGoing` | `Example onGoing` |
| A unit is `unvalidated`, or the group has no matching result | `CI/CD onGoing` |
| Otherwise | `Runnable` |

## Broken reasons

A reason describes why an example is broken; it is not a second lifecycle status.

| Badge | Interpretation |
| --- | --- |
| ![Dataset or resource unavailable](https://img.shields.io/badge/reason-Dataset%20or%20resource%20unavailable-795548) | A dataset, model, service, or other required resource is unavailable. |
| ![Dependency drift](https://img.shields.io/badge/reason-Dependency%20drift-ff69b4) | A dependency no longer resolves, installs, imports, or behaves compatibly. |

Other report-level reasons, including hardware assumptions and metric edge cases, may appear in CI artifacts even when the matrix shows the broader `Broken` status.

## How to interpret the matrix

- Each benchmark job/YAML is an independent inventory and validation unit, even when rows are visually grouped under one top-level example.
- The current published badge snapshot is aggregated by the top-level `example` value. If any validated benchmark job in that group has a blocking failure, the shared badge is `Broken`; inspect the report to identify the failing job.
- `Last T2/T3 Validation Time` is the timestamp of the latest published broad evidence, not the time of the last README edit.
- T0/T1 results help review a pull request but do not replace T2/T3 health evidence.
- `Runnable` means the validated path passed the checks in its configured tier. It is not a guarantee for every operating system, hardware combination, external resource, or undeclared Python version.
- A `Runtime smoke test (mocked_llm)` result validates integration with substituted responses only. It does not validate a real model, external provider, GPU, or output quality.
- `Known issue` is a triaged state. `Quarantined` additionally means normal dynamic validation is intentionally disabled.
- Pre-existing failures do not block unrelated pull requests, but they remain visible until fixed or explicitly reclassified.

## Status publication

T2/T3 workflows generate top-level-example JSON snapshots and a summary under
`.github/example-status/`, then publish them to the
`ci-managed/example-health-status` branch. `examples/README.md` renders badges
from those snapshots. Each snapshot includes the top-level example identifier,
aggregated status, validation timestamp, and source commit.

If a badge and a workflow report disagree, prefer the newest complete T2/T3 report, verify that snapshot publication succeeded, and then refresh or repair the status branch. Do not manually claim a passing status without matching validation evidence.

A full T2 pull-request run produces broad **base-branch** health evidence. Its completeness is checked against the complete target set selected from the base revision's inventory; head-only additions are not expected base results. That evidence may be published on the next scheduled planning run even if the pull request is still open or is never merged, because it validated the current main-branch target set. Publishing it resets the seven-day T3 cadence from the T2 validation time. The PR-head result remains PR review evidence and must not replace main-branch health.

A later complete T3 run supersedes older T2 evidence. This latest-complete-result rule is important because an example can drift after a passing T2 without any repository change—for example, when a dependency, dataset, model, API, or runner environment changes.

## Known limitations

- Published snapshots are aggregated by top-level `example`; independent
  per-benchmark badges are not currently published.
- The report generator recognizes external-resource and `broken` inventory
  values in its health-display vocabulary, but the current snapshot aggregator
  does not publish those badges from the inventory value alone. `Broken`
  requires a matched blocking result, and a top-level example group with no
  matching result falls back to `CI/CD onGoing`.

## Status evidence policy

- Use the inventory as the maintained policy source and generated T2/T3 snapshots as validation evidence.
- Do not label a mocked LLM run as evidence of a real model or provider passing.
- Do not use resource or hardware statuses to hide an ordinary reproducible software defect.
- Record constraints and follow-up references in inventory metadata or the tracking issue.
- When a failure is fixed, rerun the highest practical tier and keep the resulting report or snapshot as evidence.

## Maintainer actions

| Observed state | Recommended action |
| --- | --- |
| New PR regression | Request a fix or explicitly approve an exception. |
| Pre-existing failure | Keep an unrelated PR unblocked and track the debt separately. |
| Time-based failure | Confirm the cause, update inventory classification, and open a maintenance issue if restoration is planned. |
| External resource or hardware constraint | Document the exact requirement and preserve any available lightweight checks. |
| Repair completed | Return the entry to `active`, run the highest practical tier, and retain the report as evidence. |
