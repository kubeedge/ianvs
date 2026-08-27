# Example Classification Policy

Ianvs separates three concepts that answer different questions:

1. A **check result** (`PASS`, `FAIL`, `ERROR`, `WARNING`, or `SKIP`) describes one validator rule.
2. A **PR-impact classification** determines whether a result is newly introduced by a pull request.
3. **Published example health** is a separate aggregation and publication concern documented in [status directions](status_directions.md).

This separation prevents historical failures from blocking unrelated contributions while keeping maintenance debt visible.

See [validation rules](validation_rules.md) for individual checks and [status directions](status_directions.md) for the badge legend.

## Pull request comparison

CI selects validation targets independently from the base and head revisions'
inventories, so the two target sets are not required to match. For validation
units present on both sides, the regression detector compares issues by unit,
check, file, and diagnostic detail. Line mappings from the Git diff are used so
an unchanged issue is not treated as new merely because surrounding lines
moved.

| Base | PR head | Classification | Blocks the PR |
| --- | --- | --- | --- |
| No blocking issue | New `FAIL` or `ERROR` issue | `Failed: PR regression` | Yes |
| Same blocking issue | Same `FAIL` or `ERROR` issue | `Failed: Pre-existing failure` | No |
| Blocking issue | Issue removed | `Fixed: Pre-existing failure resolved` | No |
| No blocking issue | `WARNING` only | `Passed` with warning details | No |
| Passing | Passing | `Passed` | No |
| Result cannot be compared | Indeterminate | `Unknown` | Maintainer review |

Only newly introduced `FAIL` or `ERROR` details make the regression job fail. Warnings are reported but do not block a pull request.

## Validation unit lifecycle changes

The stable validation-unit identity is the inventory `name` plus `path`. A
missing individual check does not add or remove a unit.

| Base unit | Head unit | Reported change | PR impact |
| --- | --- | --- | --- |
| Absent | Active and passing | Added, passed | Does not block |
| Absent | Active with `FAIL` or `ERROR` | Added, PR regression | Blocks |
| Absent | Inactive eligibility `SKIP` | Added, skipped with inventory status | Does not block |
| Present | Absent | Removed with previous base state | Does not block solely because of removal |

The regression report lists added and removed units separately from
check-level comparisons. Without another stable identifier, a rename is shown
as removal of the old identity and addition of the new identity; no fuzzy
rename matching is attempted.

Absence from the base inventory is a legitimate Added-unit case. If a unit was
selected from the base inventory but its expected result artifact is missing or
cannot be compared reliably, maintainers should inspect the raw artifacts
instead of assuming the head failure is historical.

## Failure causes

The reporter assigns a cause from the failed check and its diagnostics. Supported cause labels include:

- `Failed: Dependency drift`
- `Failed: Dataset/resource drift`
- `Failed: Model/resource drift`
- `Failed: Hardware assumption`
- `Failed: Metric edge case`
- `Failed: Known issue`

Cause and PR ownership are independent. For example, a dependency failure can be a new PR regression, pre-existing debt, or a time-based maintenance failure.

Warnings are also compared as pre-existing, new, and fixed details, but their PR-impact classification remains passing. Reviewers should ask for reasonable warning fixes during normal review; an accepted warning should have a clear reason or follow-up issue rather than disappearing from the report.

## Time-based failures

A failure first found by broad scheduled validation is maintenance evidence rather than proof that an unrelated pull request caused it. Common causes include dependency drift, unavailable datasets or models, provider changes, and runner image changes.

Maintainers should:

1. preserve the report and validation timestamp;
2. reproduce or confirm the failure;
3. choose an accurate inventory status, such as `known issue`, `quarantined`, `requires_external_resource`, or `requires_hardware`;
4. create a follow-up issue when restoration is wanted;
5. return the entry to `active` only after the relevant validation tier passes.

## Maintainer triage

When a report fails, first determine whether it is a PR regression. If it is, request a fix or explicitly approve an exception. If it is pre-existing or time-based, keep the pull request unblocked, update the example classification when needed, and track restoration separately.

## Design rationale: do not share cached PR baselines

CI reruns base and head in the same validation window instead of sharing one
cached T2 base result across pull requests. A dependency, dataset, model, API,
network condition, or runner image can change while the base commit remains the
same; comparing a current head run with an older passing base could incorrectly
classify external drift as a PR regression.

T2 base results may still be reused for health publication and T3 scheduling,
but not for later PR regression comparisons. Package and download caches are
allowed because they cache inputs rather than previous validation conclusions.
