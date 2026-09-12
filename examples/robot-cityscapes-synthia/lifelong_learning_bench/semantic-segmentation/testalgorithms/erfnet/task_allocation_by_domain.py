from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('TaskAllocationByDomain',)


@ClassFactory.register(ClassType.STP, alias="TaskAllocationByDomain")
class TaskAllocationByOrigin:
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    origins: List[Metadata]
        metadata is usually a class feature
        label with finite values.
    """

    def __init__(self, task_extractor=None, **kwargs):
        self.default_origin = kwargs.get("default", None)
        # Sedna passes task_extractor to __init__, not __call__.
        # Store it here; fall back to a hardcoded mapping if not provided.
        self.task_extractor = task_extractor if task_extractor is not None else {
            "Synthia": 0, "Cityscapes": 1, "Cloud-Robotics": 2
        }

    def __call__(self, samples: BaseDataSource):
        if self.default_origin:
            return samples, [int(self.task_extractor.get(self.default_origin))] * len(samples.x)

        # List of all possible origins
        categories = ["Cityscapes", "Synthia", "Cloud-Robotics"]

        sample_origins = []
        for _x in samples.x:
            sample_origin = None
            for category in categories:
                if category in _x[0]:
                    sample_origin = category
                    break
            if sample_origin is None:
                sample_origin = self.default_origin if self.default_origin else categories[0]
            sample_origins.append(sample_origin)

        allocations = [int(self.task_extractor.get(sample_origin))
                       for sample_origin in sample_origins]

        return samples, allocations
