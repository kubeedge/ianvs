from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

__all__ = ('TaskDefinitionByOrigin',)


@ClassFactory.register(ClassType.STP, alias="TaskDefinitionByOrigin")
class TaskDefinitionByOrigin:
    """
    Dividing datasets based on the their origins.

    Parameters
    ----------
    origins： List[Metadata]
        metadata is usually a class feature label with a finite values.
    """

    def __init__(self, **kwargs):
        self.origins = kwargs.get("origins", ["front", "garden"])

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        cities = [
            "front"
            ]

        tasks = []
        d_type = samples.data_type
        x_data = samples.x
        y_data = samples.y

        task_index = dict(zip(self.origins, range(len(self.origins))))

        real_df = BaseDataSource(data_type=d_type)
        real_df.x, real_df.y = [], []
        sim_df = BaseDataSource(data_type=d_type)
        sim_df.x, sim_df.y = [], []

        for i in range(samples.num_examples()):
            is_real = False
            for city in cities:
                if city in x_data[i][0]:
                    is_real = True
                    real_df.x.append(x_data[i])
                    real_df.y.append(y_data[i])
                    break
            if not is_real:
                sim_df.x.append(x_data[i])
                sim_df.y.append(y_data[i])

        g_attr = "front_semantic_segamentation_model"
        task_obj = Task(entry=g_attr, samples=real_df, meta_attr="front")
        tasks.append(task_obj)

        g_attr = "garden_semantic_segamentation_model"
        task_obj = Task(entry=g_attr, samples=sim_df, meta_attr="garden")
        tasks.append(task_obj)

        return tasks, task_index, samples
