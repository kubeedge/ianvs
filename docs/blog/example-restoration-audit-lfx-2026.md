# What We Learned Fixing Ianvs Examples: Five Bugs, One Quarantine, and a New CI Check

Over the current LFX Mentorship application period, several contributors — myself included — spent time doing something unglamorous but overdue: actually running the examples under `examples/` in the [Ianvs](https://github.com/kubeedge/ianvs) repo and seeing what broke. Ianvs' example set has grown quickly across paradigms — single-task, incremental, lifelong, federated, and LLM benchmarking — and growth outpaced maintenance. This post summarizes what that audit found: five failure patterns that show up repeatedly, one example that turned out to be un-fixable by design, and a new CI workflow aimed at catching regressions before they land.

## Five failure patterns

Rather than treat each broken example as a one-off, we tried to name the *shape* of the bug, since the same shape shows up in more than one place.

| Pattern | Symptom | Where we found it |
|---|---|---|
| **Silent config misconfiguration** | Job runs to completion, but expected output files are just missing — no error | `core/storymanager/rank/rank.py`'s `save_mode` validation used `not X and not isinstance(X, list)`, which barely validates anything, and `save()` had no `else` branch to catch an unrecognized value |
| **Hardcoded feature flags** | Optional functionality (visualization, point-cloud downsampling) silently disabled even when the dependency *is* installed | `pose-estimation-llio/utils.py` hardcoded `MATPLOTLIB_AVAILABLE = False` / `OPEN3D_AVAILABLE = False` instead of probing with `try/except ImportError` |
| **Self-referential pip dependency** | `pip install -r requirements.txt` fails outright | Two examples listed `ianvs` itself as a dependency — not published on PyPI, since you're already running inside an Ianvs checkout |
| **Non-reproducible examples** | No config anywhere, or an error tracing back to data that was never public | Covered below |
| **Inverted boolean validation** | Invalid config values pass silently, then fail confusingly somewhere downstream | `_check_fields()` in `core/testcasecontroller/algorithm/algorithm.py` used `not X and not isinstance(X, str)` — the `and` means a value has to fail *both* checks to get rejected, so almost anything gets through |

The last one is worth dwelling on for a second: it's the same logical inversion as the first pattern, just in a different file. Once you've seen it once, it's a two-second `grep -rn "and not isinstance"` to check whether it's hiding anywhere else. We found and fixed two instances this way. The full writeup, with exact file:line citations and before/after diffs, lives in [`docs/guides/debugging-common-example-failures.md`](https://github.com/kubeedge/ianvs/blob/main/docs/guides/debugging-common-example-failures.md).

## Case study: the example that can't be fixed

Not every broken example is a bug. `examples/aoa/single_task_bench/TForest` looked, at first glance, like a straightforward missing-file problem: its one runnable entry point crashes immediately trying to load a `.joblib` artifact that was never committed. But tracing the call graph back further tells a different story — the function that *would* generate that artifact reads a dataset the example's own README says "is not authorized to be disclosed." There's also no `benchmarkingjob.yaml` anywhere in the directory tree, meaning it was never wired into the Ianvs framework to begin with, and a second, entirely separate PyTorch-based implementation sits alongside it in an `other_attempts/` folder — also dead, also dependent on the same unavailable data.

This isn't repairable by patching code, because the blocker isn't code. It's a good reminder that "broken" and "impossible to reproduce" are different classifications, and treating them the same wastes contributor time. We're proposing `Quarantined` as a distinct status for cases like this, rather than routing them into the same repair queue as an example that just needs an import fixed.

## What the new CI check does

[PR #851](https://github.com/kubeedge/ianvs/pull/851) adds `.github/workflows/example-lint.yml`, running on any pull request that touches `examples/`, across Python 3.9, 3.10, and 3.11. It does two things: runs `pylint` across the example tree, and checks that every directory with a `benchmarkingjob.yaml` also has a `requirements.txt` next to it.

Right now, both checks run with `continue-on-error: true` — deliberately non-blocking. Only 13 of 29 top-level example directories currently have any `requirements.txt` at all, so flipping this to a hard gate today would turn nearly every example PR red, including ones that have nothing to do with the failure. The plan is to tighten it once the backlog is actually clean — advisory now, blocking later.

## Where to help

If you're looking for a way into the Ianvs codebase, this is a good one: these bugs are real, scoped, and don't require deep ML expertise — mostly careful reading and a willingness to run things and see what breaks. Pick an example missing a `requirements.txt`, grep for the `and not isinstance` pattern in a file nobody's checked yet, or just run an example from a clean checkout and see how far you get. If you find something, open an issue with exact file paths and what you tried — that's what made every fix in this post possible in the first place.
