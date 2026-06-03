# Third-Party Backup Wheels (`third_party-bk`)

This directory holds **legacy sedna wheels** that are no longer used by the CI
pipeline but are retained for compatibility with older examples and for rollback
reference. Do **not** install these for new benchmark work.

---

## What is in this directory vs `resources/third_party`

| | `third_party-bk/` | `third_party/` |
|---|---|---|
| **Purpose** | Archive of legacy wheels | Current wheels installed by CI |
| **Used by CI?** | No | Yes (`main.yaml` installs from here) |
| **Wheels** | `sedna-0.4.1`, `sedna-0.4.5` | `sedna-0.6.0.1`, `sedna-0.6.0.2` |
| **Examples that reference these** | `llm-edge-benchmark-suite` (0.4.1) | All other current examples |
| **Install?** | Only if running a legacy 0.4.x example | Default for all new examples |

---

## When to use `third_party-bk`

**Use `sedna-0.4.1`** if you are running `examples/llm-edge-benchmark-suite/` — its
README explicitly installs `sedna-0.4.1-py3-none-any.whl`. This example was written
against the 0.4.x API and has not been ported to 0.6.x.

**Do not use these wheels** for:
- Any lifelong-learning benchmark (cityscapes-synthia, bdd, robot) — they require the
  `knowledge_management` refactor that was introduced in 0.6.x
- Any example whose README references `sedna-0.6.*`
- New example development — always target 0.6.0.2 or later

---

## Wheel-by-wheel summary

### `sedna-0.4.1-py3-none-any.whl`

The oldest wheel in this archive. Despite the low version number this is actually
**feature-rich**: it contains the full lifelong-learning stack including
`my_evaluate()`, `inference_2()`, and `my_inference()` on `LifelongLearning`,
plus `seen_task_learning`, `unseen_task_detection`, and `unseen_task_processing`
algorithm sub-packages.

- **Python files:** 114
- `sedna.core.lifelong_learning.LifelongLearning` — has `train`, `evaluate`,
  `inference`, `inference_2`, `my_evaluate`, `my_inference`
- `sedna.algorithms` — has `seen_task_learning`, `knowledge_management`,
  `unseen_task_detection`, `unseen_task_processing`
- **EDGE_KB_DIR:** `/var/lib/sedna/kb` (requires root / KubeEdge node)
- **datasources:** `np.array(x_data)` (no `dtype=object`; can fail on ragged paths)
- `LCClient.send()` — raises `ConnectionRefusedError` when no LC server is running

### `sedna-0.4.5-py3-none-any.whl`

A **stripped-down release** — fewer files than 0.4.1 (83 vs 114). Many lifelong-learning
sub-packages were removed or not yet re-integrated, making this wheel **incompatible**
with lifelong-learning benchmarks despite being a higher version number than 0.4.1.

- **Python files:** 83
- `sedna.core.lifelong_learning.LifelongLearning` — only `train`, `update`,
  `evaluate`, `inference` (no `inference_2`, no `my_evaluate`, no `my_inference`)
- `sedna.algorithms` — **missing** `seen_task_learning`, `unseen_task_detection`,
  `unseen_task_processing` (dropped in this release)
- **Do not use for any lifelong-learning benchmark.** It will raise `AttributeError`
  on `my_evaluate` and `ValueError` on missing ClassFactory registrations.

---

## Differences: 0.4.1 / 0.4.5 → 0.6.0.1 → 0.6.0.2

### 0.4.x → 0.6.0.1 (upstream changes)

| Area | 0.4.1 | 0.6.0.1 |
|------|-------|---------|
| `LifelongLearning` methods | `train`, `_initial_train`, `_update`, `evaluate`, `inference`, `inference_2`, `my_evaluate`, `my_inference` | `train`, `update`, `evaluate`, `inference` only (removed helpers) |
| Knowledge management location | `sedna.algorithms.knowledge_management` | Moved to `sedna.core.lifelong_learning.knowledge_management` |
| `task_update_decision` registration | Imported and registered | Present but **not imported** in `seen_task_learning/__init__.py` → classes never registered |
| `EDGE_KB_DIR` | `/var/lib/sedna/kb` | `/var/lib/sedna/kb` (same — requires root) |
| `datasources` dtype | `np.array(x_data)` | `np.array(x_data)` (same — ragged-path bug present) |
| `LCClient.send()` | Raises on localhost | Raises on localhost |
| Python files | 114 | ~110 |

### 0.6.0.1 → 0.6.0.2 (ianvs patch, in `third_party/`)

| Area | 0.6.0.1 | 0.6.0.2 |
|------|---------|---------|
| `inference_2`, `my_evaluate`, `my_inference` | Absent → `AttributeError` at runtime | Added back to `LifelongLearning` |
| `task_update_decision` registration | Missing import → `ValueError` on first round | Fixed: `from . import task_update_decision` added |
| `EDGE_KB_DIR` | `/var/lib/sedna/kb` (root required) | `~/.sedna/kb` (works for local user) |
| `LCClient.send()` | Crashes on localhost | Skips silently when server is localhost or unset |
| `datasources` dtype | `np.array(x_data)` | `np.array(x_data, dtype=object)` (safe for path arrays) |
| `KBClient` with no server | `AttributeError` / `ConnectionRefusedError` | Returns early with no-op |
| Nested metrics in `cloud_knowledge_management` | `TypeError` on nested dict | Flattened before comparison |
| Log level for missing LC server | `error` | `debug` (less noise in local mode) |

---

## Which wheel to install for each example

| Example | Install |
|---------|---------|
| `cityscapes-synthia/lifelong_learning_bench/**` | `third_party/sedna-0.6.0.2-py3-none-any.whl` |
| `bdd/lifelong_learning_bench/**` | `third_party/sedna-0.6.0.2-py3-none-any.whl` |
| `robot/lifelong_learning_bench/**` | `third_party/sedna-0.6.0.2-py3-none-any.whl` |
| `cityscapes/lifelong_learning_bench/**` | `third_party/sedna-0.6.0.2-py3-none-any.whl` |
| `llm-edge-benchmark-suite/**` | `third_party-bk/sedna-0.4.1-py3-none-any.whl` |
| `TAB/**`, `cloud-edge-*`, `phys_scene_gen/**` | `third_party/sedna-0.6.0.1-py3-none-any.whl` or `0.6.0.2` |
| `cifar100/**`, `MOT17/**`, `GovDoc2Poster/**` | Either 0.6.x (only use `sedna.common.*`, `sedna.datasources`) |

> **Note:** `sedna-0.6.0.2` is a strict superset of `0.6.0.1` — all API changes are
> additive. Installing `0.6.0.2` will not break any example that worked with `0.6.0.1`.
