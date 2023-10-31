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

    def __init__(self, **kwargs):
        self.default_origin = kwargs.get("default", None)

    def __call__(self, task_extractor, samples: BaseDataSource):
        self.task_extractor = {"Synthia": 0, "Cityscapes": 1, "Cloud-Robotics": 2}  # Mapping of origins to task indices

        if self.default_origin:
            return samples, [int(self.task_extractor.get(self.default_origin))] * len(samples.x)

        categories = ["Cityscapes", "Synthia", "Cloud-Robotics"]  # List of all possible origins

        sample_origins = []
        for _x in samples.x:
            sample_origin = None
            for category in categories:
                if category in _x[0]:
                    sample_origin = category
                    break
            if sample_origin is None:
                # If none of the categories match, assign a default origin
                sample_origin = self.default_origin if self.default_origin else categories[0]
            sample_origins.append(sample_origin)

        allocations = [int(self.task_extractor.get(sample_origin)) for sample_origin in sample_origins]

        return samples, allocations
