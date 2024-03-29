from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('TaskAllocationByOrigin',)


@ClassFactory.register(ClassType.STP, alias="TaskAllocationByOrigin")
class TaskAllocationByOrigin:
    """
    Corresponding to `TaskDefinitionByOrigin`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, **kwargs):
        self.default_origin = kwargs.get("default", None)

    def __call__(self, task_extractor, samples: BaseDataSource):
        #self.task_extractor = task_extractor
        self.task_extractor = {"front": 0, "garden": 1}
        if self.default_origin:
            return samples, [int(self.task_extractor.get(
                self.default_origin))] * len(samples.x)

        cities = [
            "front"
            ]

        sample_origins = []
        for _x in samples.x:
            is_real = False
            for city in cities:
                if city in _x[0]:
                    is_real = True
                    sample_origins.append("front")
                    break
            if not is_real:
                sample_origins.append("garden")

        allocations = [int(self.task_extractor.get(sample_origin))
                       for sample_origin in sample_origins]

        return samples, allocations
