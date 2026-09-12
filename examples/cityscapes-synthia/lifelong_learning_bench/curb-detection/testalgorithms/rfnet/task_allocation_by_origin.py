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
        self.task_extractor = None

    def __call__(self, task_extractor=None, samples: BaseDataSource = None):
        if task_extractor is not None:
            self.task_extractor = task_extractor
        if self.task_extractor is None:
            self.task_extractor = {"real": 0, "sim": 1}

        if self.default_origin:
            return samples, [int(self.task_extractor.get(
                self.default_origin, 0))] * len(samples.x)

        cities = [
            "aachen",
            "berlin",
            "bochum",
            "bremen",
            "cologne",
            "darmstadt",
            "dusseldorf",
            "erfurt",
            "hamburg",
            "hanover",
            "jena",
            "krefeld",
            "monchengladbach",
            "strasbourg",
            "stuttgart",
            "tubingen",
            "ulm",
            "weimar",
            "zurich"]

        sample_origins = []
        for _x in samples.x:
            if _x is None or (hasattr(_x, '__len__') and len(_x) == 0):
                sample_origins.append("real")
                continue
            sample_path = _x[0] if isinstance(_x, (list, tuple)) else str(_x)
            is_real = any(city in sample_path for city in cities)
            sample_origins.append("real" if is_real else "sim")

        allocations = [int(self.task_extractor.get(origin, 0))
                       for origin in sample_origins]

        return samples, allocations
