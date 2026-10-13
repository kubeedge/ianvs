# Debugging Playbook: Common Ianvs Example Failure Modes

This guide catalogs failure patterns that have actually been found and fixed in the Ianvs codebase and its examples. Each entry cites the real file where the pattern occurred, so you can see the exact bug rather than a hypothetical one.

If your benchmark run is failing, silently producing no output, or behaving unexpectedly, check whether it matches one of the patterns below before assuming the problem is in your own configuration.

---

## 1. Silent `save_mode` misconfiguration

**Where this happened:** `core/storymanager/rank/rank.py`

### Symptom

Your benchmark job runs to completion with no errors, but the rank/leaderboard output files you expect (`all_rank.csv`, `selected_rank.csv`, or the rank pictures) are missing or incomplete — and nothing in the log tells you why.

### Root cause

`Rank.save_mode` is a string field (default `"selected_and_all"`) that selects which branch of `Rank.save()` runs. The validation that was supposed to catch invalid values used to be:

```python
# the old, buggy version
if not self.save_mode and not isinstance(self.save_mode, list):
    raise ValueError(...)
```

This checks `save_mode` against `list`, even though it's always a `str` — so the `isinstance` check could never usefully fail, and the `and` meant validation only raised when `save_mode` was *both* falsy *and* not a list. Almost any string value, valid or not, passed straight through.

Downstream, `save()` used a sequence of independent `if` statements instead of `if/elif/else`:

```python
# the old, buggy version
if self.save_mode == "selected_and_all":
    ...
if self.save_mode == "selected_only":
    ...
if self.save_mode == "selected_and_all_and_picture":
    ...
# no else — an unrecognized save_mode falls through and writes nothing, silently
```

So a typo'd or unsupported `save_mode` in your `testenv.yaml` / job config wouldn't error at config-parse time (weak validation) or at save time (no `else` branch) — it just silently skipped writing output.

### How to diagnose

1. Check what `save_mode` your job actually resolves to — add a print/log right before `Rank.save()` is called, or check the value in your parsed config.
2. Confirm it's exactly one of: `"selected_and_all"`, `"selected_only"`, `"selected_and_all_and_picture"`. Anything else — including a similar-looking string — is silently accepted by old versions and produces no output.
3. If you're on a version before this was fixed, this is the first thing to check when rank output is missing with no error.

### Fix

Validate against an explicit allow-list at config time, and raise on the `else` branch of `save()` instead of falling through silently:

```python
valid_save_modes = ("selected_and_all", "selected_only", "selected_and_all_and_picture")
if not isinstance(self.save_mode, str) or self.save_mode not in valid_save_modes:
    raise ValueError(f"rank's save_mode({self.save_mode}) must be one of {valid_save_modes}.")
```

```python
if self.save_mode == "selected_and_all":
    ...
elif self.save_mode == "selected_only":
    ...
elif self.save_mode == "selected_and_all_and_picture":
    ...
else:
    raise ValueError(f"rank's save_mode({self.save_mode}) is not supported by save().")
```

The general lesson: **a config field with a fixed set of valid values needs an explicit allow-list check, not a truthiness/type check**, and any dispatch on that field needs an `else`/default branch that raises rather than silently no-ops.

---

## 2. Hardcoded `False` feature flags that ignore whether the dependency is actually installed

**Where this happened:** `examples/industrialEI/pose-estimation-llio/testalgorithms/llio_fusion/utils.py`

### Symptom

Point-cloud downsampling or visualization output is missing or degraded, even though `matplotlib` and `open3d` are correctly installed in your environment. No crash — just a warning log easy to miss, and reduced functionality.

### Root cause

The file used to hardcode the availability flags instead of probing for the packages:

```python
# the old, buggy version — no try/except at all
MATPLOTLIB_AVAILABLE = False
OPEN3D_AVAILABLE = False
```

These flags gate real functionality elsewhere in the file:

```python
def downsample_points(vel, cfg):
    if OPEN3D_AVAILABLE:
        return velo2downpcd(vel, cfg["voxel_size"])
    else:
        LOGGER.warning("Warning: Open3D not available, returning original point cloud")
        return vel

def visualize(vis_material):
    if not MATPLOTLIB_AVAILABLE:
        LOGGER.warning("Warning: Matplotlib not available, visualization disabled")
        return
    ...
```

Because the flags were hardcoded `False` rather than detected, these branches always took the "unavailable" path — silently skipping downsampling and disabling visualization — **regardless of whether the packages were actually present**. The failure is invisible unless you're specifically watching the logs for that warning line.

### How to diagnose

1. If a benchmark using this module runs but produces no plots or unexpectedly returns raw (non-downsampled) point clouds, grep the run log for `"not available"` warnings.
2. Check the top of the relevant `utils.py` for how `*_AVAILABLE` flags are set. If you see a bare assignment (`FLAG = False`) instead of a `try: import ... except ImportError:` block, the flag is not actually reflecting your environment.

### Fix

Detect availability with a real import attempt:

```python
try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
```

The general lesson: **an "is this optional dependency available" flag must be derived from an actual import attempt, not hardcoded** — otherwise the flag lies about your environment, and the failure mode is a silent capability downgrade rather than a crash, which is much harder to notice.

---

## 3. Self-referential pip dependency (`ianvs` listed as its own example's dependency)

**Where this happened:** `examples/GovDoc2Poster/singletask_learning_bench/requirements.txt:5` and `examples/imagenet/multiedge_inference_bench/requirements.txt:27`

### Symptom

`pip install -r requirements.txt` for the example fails with something like:

```
ERROR: Could not find a version that satisfies the requirement ianvs>=0.1.0 (from versions: none)
ERROR: No matching distribution found for ianvs>=0.1.0
```

### Root cause

Both requirements files list `ianvs` itself as a pip dependency:

```text
# examples/GovDoc2Poster/singletask_learning_bench/requirements.txt
ianvs>=0.1.0
sedna>=0.1.0
```

```text
# examples/imagenet/multiedge_inference_bench/requirements.txt
ianvs>=0.3.0
```

But `ianvs` is not published on PyPI (confirmed: `https://pypi.org/pypi/ianvs/json` returns `{"message": "Not Found"}`). It's the framework you're already running *inside of* — every example is invoked via `ianvs -f examples/<name>/benchmarkingjob.yaml` from a source checkout, not imported as an installed library. Listing it as a pip dependency of one of its own examples is circular and, since the name isn't registered on PyPI, unresolvable.

### How to diagnose

If a fresh `pip install -r requirements.txt` fails specifically on the `ianvs` line (not on some other package), this is almost certainly it — check the requirements file for a bare `ianvs` line near the top.

### Fix

Remove the `ianvs` line from the example's `requirements.txt` entirely. It should never appear there — you already have Ianvs from the source checkout you're running the example inside of. If a minimum Ianvs version genuinely matters for the example, note it as a comment in the README instead of a pip requirement line, since it can't be resolved as one.

The general lesson: **an example's `requirements.txt` should only list packages that are actually installable from a package index and are not the framework itself.**

---

## 4. Permanently non-reproducible examples (proprietary data, no Ianvs integration)

**Where this happened:** `examples/aoa/single_task_bench/TForest`

### Symptom

There's no `benchmarkingjob.yaml` anywhere in the example's directory tree, so it can't be run via `ianvs -f` at all. If you try running the standalone script directly (`python3 task_main.py`), it fails immediately:

```
FileNotFoundError: [Errno 2] No such file or directory: 'single_tree.joblib'
```

### Root cause

This is a different class of problem from the others in this guide: it isn't a bug you can patch. The script's only enabled entry point (`test_main()`) loads a `.joblib` artifact that was never committed to the repo. Tracing why leads to a data-generation function (`tools.py:data_io()`) that reads from `data/processed/`, a directory that doesn't exist in the repo — and the example's own `README.md` states the underlying dataset "is not authorized to be disclosed." There is also no `benchmarkingjob.yaml` / `testenv.yaml` / `algorithm.yaml` anywhere under `examples/aoa/`, meaning it was never wired into the Ianvs framework in the first place — it's a standalone research script tree, not an Ianvs paradigm plugin.

### How to diagnose

1. `find examples/<name> -iname "*.yaml"` — if this returns nothing, the directory isn't an Ianvs example at all; don't look for a config bug, there isn't one to find.
2. Read the README for any note about data availability/licensing before spending time trying to reconstruct missing input files. If the README says the data can't be disclosed, no fix will make the example reproducible with the real data.

### Fix

There isn't a code fix for the missing-data constraint itself. The correct handling is **classification, not repair**:
- Confirm with maintainers whether the directory should be marked `Quarantined` (excluded from normal validation/repair effort) rather than `Broken`.
- If it's kept, remove dead/duplicate code and committed binary artifacts that no runnable path ever reaches, and document clearly in the README that clean-environment reproduction isn't possible.
- Add a `requirements.txt` for whatever *is* declared as imported, even if the example can't fully run, so at least the dependency surface is documented.

The general lesson: **before debugging an example, check whether it was ever wired into the Ianvs framework and whether its data dependency is even obtainable** — some "broken" examples are actually unreproducible by design, and no amount of code fixing will change that.

---

## 5. `_check_fields` validation using `and` instead of `or` — invalid types pass through silently

**Where this happened:** `core/testcasecontroller/algorithm/algorithm.py:132-134`

### Symptom

An `Algorithm` config with an invalid (non-string) `name` field doesn't raise a validation error at config-parse time. The failure instead surfaces later, confusingly, wherever the name is first used as a string (e.g. string formatting, path construction).

### Root cause

```python
def _check_fields(self):
    if not self.name and not isinstance(self.name, str):
        raise ValueError(f"algorithm name({self.name}) must be provided and be string type.")
```

This is the same anti-pattern as the original `rank.py` `save_mode` bug above: two conditions joined with `and` means *both* have to be true to raise. A non-string, truthy value (e.g. an integer, a list, a dict parsed from malformed YAML) is not falsy, so `not self.name` is `False` — and the whole condition short-circuits to `False` regardless of the `isinstance` check. Only a falsy *and* non-string value (which is a narrow, mostly-empty-value case) actually raises. Anything else, including clearly wrong types, is silently accepted.

### How to diagnose

If you're getting a confusing downstream error (e.g. `TypeError` or unexpected formatting) that seems to originate from `self.name` having the wrong type, but no validation error was raised when the config was first loaded, check `_check_fields()` for this exact `and`-based pattern rather than assuming your YAML is fine just because it parsed without error.

### Fix

Use `or`, and check the type first:

```python
def _check_fields(self):
    if not isinstance(self.name, str) or not self.name:
        raise ValueError(f"algorithm name({self.name}) must be provided and be string type.")
```

The general lesson, common to both this and pattern #1: **when validating "must be a non-empty value of type X," write it as `not isinstance(x, X) or not x`, not `not x and not isinstance(x, X)`.** The `and` version only rejects values that fail both checks simultaneously, which lets almost everything through. This exact mistake has appeared in more than one place in this codebase — grep for `and not isinstance` when auditing other `_check_fields`/`_parse_config` methods.

---

## Quick reference

| Pattern | Symptom | First thing to check |
|---|---|---|
| Silent `save_mode` misconfig | No error, but expected output files missing | Exact value of `save_mode`; presence of `else`/default branch in the dispatch |
| Hardcoded `False` feature flag | Functionality silently degraded despite dependency installed | Is the availability flag set via `try/except ImportError`, or hardcoded? |
| Self-referential pip dependency | `pip install -r requirements.txt` fails on a package named after the framework itself | Does the requirements file list `ianvs` (or `sedna`, if not actually needed) as a dependency? |
| Permanently non-reproducible example | No `benchmarkingjob.yaml`; standalone script fails on a missing data/model artifact | `find examples/<name> -iname "*.yaml"`; README notes on data availability |
| `and` instead of `or` in field validation | Invalid value/type passes validation, fails confusingly later | Grep the relevant `_check_fields`/`_parse_config` for `and not isinstance` |
