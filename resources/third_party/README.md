# Third-Party Wheels

This directory contains vendored Python wheels that are installed as part of the
ianvs CI pipeline and local development setup.

## sedna-0.6.0.2-py3-none-any.whl

**Based on:** sedna 0.6.0.1 (upstream kubeedge/sedna)  
**Modified version:** 0.6.0.2 (ianvs local patch)  
**Why vendored:** The upstream sedna release does not expose the APIs required
by the cityscapes-synthia and bdd lifelong-learning benchmarks when run without
a live KubeEdge cluster.

### Changes from 0.6.0.1 → 0.6.0.2

| File | Change | Reason |
|------|--------|--------|
| `core/lifelong_learning/lifelong_learning.py` | Added `inference_2()`, `my_evaluate()`, `my_inference()` methods | ianvs paradigm layer calls these methods; they are absent from the upstream release |
| `algorithms/seen_task_learning/__init__.py` | Added `from . import task_update_decision` | Without this import the `@ClassFactory.register` decorators on `UpdateStrategyDefault` and `UpdateStrategyByFinetune` never run, causing a `ValueError` at benchmark startup |
| `service/client.py` | Skip `LCClient.send()` when server is localhost or unset; guard `KBClient` methods when `kbserver` is `None` | Silences `ConnectionRefusedError` spam when running ianvs without a KubeEdge LC/KB server |
| `common/constant.py` | Changed `EDGE_KB_DIR` from `/var/lib/sedna/kb` to `~/.sedna/kb` | The original path requires root access on a KubeEdge edge node; the new path works for local benchmark runs |
| `datasources/__init__.py` | `np.array(x_data, dtype=object)` | Prevents numpy from trying to infer a numeric dtype for arrays of file-path strings, which raises a `ValueError` on ragged inputs |
| `core/lifelong_learning/knowledge_management/cloud_knowledge_management.py` | Flatten nested metrics dict before computing best-score comparison | Fixes a `TypeError` when the metrics dict returned by `evaluate()` is nested (e.g. `{"accuracy": {"round1": 0.3}}`) |
| `algorithms/seen_task_learning/seen_task_learning.py` | Guard for non-dict `task_definition`/`seen_task_allocation`; warn on empty `seen_task_groups` | Allows task definition/allocation classes to be passed directly as objects rather than config dicts |
| `core/base.py` | Downgrade LC-server connection error from `error` to `debug` | Reduces log noise in local mode |

### Compatibility

All changes are additive or fix error paths that only trigger in standalone (non-KubeEdge)
mode. The public API surface of every sedna module is unchanged:

- **Single-task / federated examples** (`cifar100`, `GovDoc2Poster`, `llm-*`, `MOT17`, etc.)
  only import `sedna.datasources`, `sedna.common.*`, and `sedna.service.*`. None of these
  interfaces were changed in a breaking way.

- **Lifelong-learning examples** (`bdd/curb-detection`, `robot/semantic-segmentation`,
  `cityscapes/unseen_task_processing`) gain two previously missing class registrations
  (`UpdateStrategyDefault`, `UpdateStrategyByFinetune`) and three new helper methods on
  `LifelongLearning`. Existing calls to `train()`, `evaluate()`, and `predict()` behave
  identically.

### Reproducing the wheel

If you need to rebuild the wheel from source:

```shell
git clone https://github.com/kubeedge/sedna.git sedna_src
# apply the patches listed above
cd sedna_src
# bump version to 0.6.0.2 in lib/sedna/VERSION or setup.cfg
pip install wheel
python setup.py bdist_wheel --dist-dir /tmp/sedna_wheel
cp /tmp/sedna_wheel/sedna-0.6.0.2-py3-none-any.whl resources/third_party/
```

## sedna-0.6.0.1-py3-none-any.whl

The original upstream sedna wheel. Retained for reference and rollback. Not used
by CI; the `main.yaml` workflow installs `sedna-0.6.0.2-py3-none-any.whl` explicitly.
