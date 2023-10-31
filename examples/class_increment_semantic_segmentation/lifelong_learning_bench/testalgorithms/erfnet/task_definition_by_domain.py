from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

__all__ = ('TaskDefinitionByDomain',)


@ClassFactory.register(ClassType.STP, alias="TaskDefinitionByDomain")
class TaskDefinitionByOrigin:
    """
    Dividing datasets based on their origins.

    Parameters
    ----------
    origins： List[Metadata]
        metadata is usually a class feature label with finite values.
    """

    def __init__(self, **kwargs):
        self.origins = kwargs.get("origins", ["Cityscapes", "Synthia", "Cloud-Robotics"])

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        categories = self.origins

        tasks = []
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y

        task_index = dict(zip(categories, range(len(categories))))

        data_sources = {category: BaseDataSource(data_type=d_type) for category in categories}
        print(data_sources)

        for category in data_sources.values():
            category.x = []
            category.y = []

        for i in range(samples.num_examples()):
            for category in categories:
                if category in x_data[i]:
                    data_sources[category].x.append(x_data[i])
                    data_sources[category].y.append(y_data[i])
                    break

        for category, data_source in data_sources.items():
            task_name = f"{category}_semantic_segmentation_model"
            task_obj = Task(entry=task_name, samples=data_source, meta_attr=category)
            tasks.append(task_obj)

        return tasks, task_index, samples