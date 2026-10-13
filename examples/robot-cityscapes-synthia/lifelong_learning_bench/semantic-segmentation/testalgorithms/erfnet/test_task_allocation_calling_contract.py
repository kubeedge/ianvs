"""
Regression test for the Sedna calling contract fixed in issue #473.

Sedna's STP (seen-task-processing) lifecycle constructs the module with
task_extractor passed to __init__, then invokes the instance later with
only samples: instance(samples=...). Before this fix,
TaskAllocationByOrigin.__call__ required task_extractor as a call-time
argument, which raised a missing-argument TypeError under that contract.

This test does not exercise the full benchmarking pipeline or require the
mdil-ss dataset -- it only pins the constructor/call signature so a future
change can't silently reintroduce the mismatch.
"""
import unittest

from task_allocation_by_domain import TaskAllocationByOrigin


class _MockBaseDataSource:
    """Minimal stand-in for sedna.datasources.BaseDataSource.

    Only implements what TaskAllocationByOrigin.__call__ actually uses:
    .x as a sequence of (path,) tuples, and len(samples.x).
    """

    def __init__(self, paths):
        # Mirrors the shape sedna's TxtDataParse produces: samples.x is a
        # sequence of one-element (or more) tuples, index 0 being a path.
        self.x = [(p,) for p in paths]


class TestSednaCallingContract(unittest.TestCase):
    def test_task_extractor_passed_via_init_not_call(self):
        """Sedna's actual pattern: task_extractor in __init__, samples in __call__."""
        task_extractor = {"Synthia": 0, "Cityscapes": 1, "Cloud-Robotics": 2}
        stp = TaskAllocationByOrigin(task_extractor=task_extractor)

        samples = _MockBaseDataSource([
            "/data/Synthia/img_001.png",
            "/data/Cityscapes/img_002.png",
            "/data/Cloud-Robotics/img_003.png",
        ])

        # This is the exact call shape Sedna's core lifecycle uses.
        returned_samples, allocations = stp(samples=samples)

        self.assertIs(returned_samples, samples)
        self.assertEqual(allocations, [0, 1, 2])

    def test_call_with_only_samples_kwarg_does_not_raise(self):
        """Calling with samples only (no task_extractor) must not TypeError.

        This is the exact failure mode from #473: __call__ used to require
        task_extractor as a second positional/keyword argument.
        """
        stp = TaskAllocationByOrigin(
            task_extractor={"Synthia": 0, "Cityscapes": 1, "Cloud-Robotics": 2}
        )
        samples = _MockBaseDataSource(["/data/Cityscapes/img_001.png"])

        try:
            stp(samples=samples)
        except TypeError as exc:
            self.fail(f"__call__ raised TypeError under Sedna's calling contract: {exc}")

    def test_missing_task_extractor_falls_back_to_default_mapping(self):
        """__init__ tolerates task_extractor=None by using the hardcoded fallback."""
        stp = TaskAllocationByOrigin(task_extractor=None)
        samples = _MockBaseDataSource(["/data/Synthia/img_001.png"])

        _, allocations = stp(samples=samples)

        self.assertEqual(allocations, [0])


if __name__ == "__main__":
    unittest.main()
