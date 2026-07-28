# Demo Examples README

This file demonstrates the proposed structure for `examples/README.md`.
The real `examples/README.md` should keep the latest validation time and the example classification matrix, then link to `status_directions.md` for detailed status definitions.

For status meanings, badge definitions, and broken-status subtypes, see [`status_directions.md`](status_directions.md).

**Last T2/T3 Validation Time:** `2026-07-07 06:55 UTC`

## Example Classification Matrix

| Example | Path | Status |
| --- | --- | --- |
| Demo Runnable Example | `examples/demo-runnable` | ![Runnable](https://img.shields.io/badge/status-Runnable-brightgreen) |
| Demo Broken Example | `examples/demo-broken` | ![Broken](https://img.shields.io/badge/status-Broken-red) |
| Demo Not Validated Example | `examples/demo-unvalidated` | ![Not validated yet](https://img.shields.io/badge/status-Not%20validated%20yet-lightgrey) |
| Demo External Dataset Example | `examples/demo-dataset` | ![Requires external dataset or model download](https://img.shields.io/badge/status-Requires%20external%20dataset%20or%20model%20download-blue) |
| Demo Model Download Example | `examples/demo-model-download` | ![Requires external dataset or model download](https://img.shields.io/badge/status-Requires%20external%20dataset%20or%20model%20download-blue) |
| Demo GPU Example | `examples/demo-gpu` | ![Requires GPU or special hardware](https://img.shields.io/badge/status-Requires%20GPU%20or%20special%20hardware-orange) |
| Demo Quarantined Example | `examples/demo-quarantined` | ![Quarantined](https://img.shields.io/badge/status-Quarantined-8a2be2) |
| Demo Known Issue Example | `examples/demo-known-issue` | ![Known issue](https://img.shields.io/badge/status-Known%20issue-critical) |
| Demo Dependency Drift Example | `examples/demo-dependency-drift` | ![Broken](https://img.shields.io/badge/status-Broken-red) ![Dependency drift](https://img.shields.io/badge/reason-Dependency%20drift-ff69b4) |
| Demo Resource Unavailable Example | `examples/demo-resource-unavailable` | ![Broken](https://img.shields.io/badge/status-Broken-red) ![Dataset or resource unavailable](https://img.shields.io/badge/reason-Dataset%20or%20resource%20unavailable-795548) |
| Demo Documentation Issue Example | `examples/demo-doc-issue` | ![Broken](https://img.shields.io/badge/status-Broken-red) ![Documentation issue](https://img.shields.io/badge/reason-Documentation%20issue-607d8b) |
