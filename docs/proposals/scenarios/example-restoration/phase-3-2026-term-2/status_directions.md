# Example Status

This file documents the example status model used by the Example Classification Matrix proposed in `proposal.md`.
It is intended to hold the status explanations, legend, classification rules, and maintenance notes that would otherwise make `examples/README.md` too long.
The `examples/README.md` page should keep the latest validation time and example status matrix, then link to this file for detailed status definitions and interpretation guidance.

## Status Legend

![Status transition standard](images/example-status-STD.png)

- ![Runnable](https://img.shields.io/badge/status-Runnable-brightgreen) `Runnable`
- ![Broken](https://img.shields.io/badge/status-Broken-red) `Broken`
- ![Not validated yet](https://img.shields.io/badge/status-Not%20validated%20yet-lightgrey) `Not validated yet`
- ![Requires external dataset or model download](https://img.shields.io/badge/status-Requires%20external%20dataset%20or%20model%20download-blue) `Requires external dataset or model download`
- ![Requires GPU or special hardware](https://img.shields.io/badge/status-Requires%20GPU%20or%20special%20hardware-orange) `Requires GPU or special hardware`
- ![Quarantined](https://img.shields.io/badge/status-Quarantined-8a2be2) `Quarantined`
- ![Known issue](https://img.shields.io/badge/status-Known%20issue-critical) `Known issue`

### Auto-classified broken subtypes

The diagram treats some labels as machine-assigned explanations for why an example is currently `Broken`, instead of as independent top-level lifecycle states.

- ![Dataset or resource unavailable](https://img.shields.io/badge/reason-Dataset%20or%20resource%20unavailable-795548) `Dataset or resource unavailable`
- ![Dependency drift](https://img.shields.io/badge/reason-Dependency%20drift-ff69b4) `Dependency drift`
- ![Documentation issue](https://img.shields.io/badge/reason-Documentation%20issue-607d8b) `Documentation issue`

## Notes

- `Status` is the primary classification field, and each example should normally have one main status.
- `Broken` may include an auto-classified subtype such as `Dataset or resource unavailable`, `Dependency drift`, or `Documentation issue`.
- `Last T2/T3 Validation Time` records the most recent broad validation evidence from Tier 2 or Tier 3 validation and should remain with the status matrix in `examples/README.md`.
- `Known issue` means the failure has already been triaged; `Quarantined` means validation was intentionally disabled until follow-up repair work is ready.
- Use the `Notes` column for additional constraints such as Python version, dataset location, hardware requirements, model download requirements, or whether the failure is pre-existing versus newly introduced.
- If CI automation is added later, the workflow can update badges, the top-level T2/T3 validation timestamp, and classification results in the `examples/README.md` status matrix directly.
