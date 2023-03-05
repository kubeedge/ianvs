# Report for Lifelong Learning of Cityscapes Dataset

## 1 Project Background

The current mainstream machine learning paradigm is to run machine learning algorithms on a given set of data to generate a model, and then apply this model to a task in a real environment, which we can call "isolated learning". The main problem with this learning paradigm is that the model does not retain and accumulate previously learned knowledge and cannot use it in future learning, and the learning environment is static and closed, in contrast to the human learning process. In reality, the situation is so varied that it is clearly impossible to label every possible task or to collect large amounts of data before training in order for a machine learning algorithm to learn. Lifelong machine learning was created to address these problems.

Lifelong learning has five key characteristics.

1. a process of continuous learning.
2. the accumulation and retention of knowledge in the knowledge base.
3. the ability to use accumulated learned knowledge to aid future learning
4. the ability to discover new tasks.
5. the ability to learn while working.

Relying on the lifelong learning system built by KubeEdge+Sedna+Ianvs distributed collaborative AI joint inference framework, the core task of this project is to complete the unknown task identification algorithm module and embed it in the framework, with the aim of equipping the system with the ability to discover new tasks.

Traditional machine learning performs test set inference by training known samples, whose knowledge is limited, and the resulting models cannot effectively identify unknown samples in new classes, which will be treated as known samples. In a real production environment, it is difficult to guarantee that the training set contains samples from all classes. If the unknown class samples cannot be identified, the accuracy and confidence of the model will be greatly affected, and the cost consumed for model improvement is incalculable. This project aims to reproduce the algorithm of the CVPR2021 paper "Learning placeholders for open-set recognition". The paper proposes placeholders that mimic the emergence of new classes, thus helping to transform closed training into open training to accomplish recognition of unknown classes of data.

In this project, the algorithm is packaged as a python callable module and embedded in the lib library of Ianvs' lifelong learning testing system. The algorithm developer does not need to develop additional algorithms for unknown task recognition and can directly test the performance of the currently developed algorithms in combination with the dataset and testing environment provided by Ianvs. At the same time, Ianvs provides local and cloud-based algorithm performance rankings for developers to facilitate the exchange of lifelong machine learning researchers and thus promote the development of the lifelong learning research field.

## 2 Goals

1. Realize the benchmark of cityscape for lifelong learning

## 3 Proposal

### 3.1 task definition

The samples will be defined as two tasks based on whether the sample is in the cities of "aachen, berlin, bochum, bremen, cologne, darmstadt, dusseldorf, erfurt, frankfurt".

~~~python
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
        self.origins = kwargs.get("origins", ["real", "sim"])

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        cities = [
            "aachen",
            "berlin",
            "bochum",
            "bremen",
            "cologne",
            "darmstadt",
            "dusseldorf",
            "erfurt",
            "frankfurt"]
            
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

        g_attr = "real_semantic_segamentation_model"
        task_obj = Task(entry=g_attr, samples=real_df, meta_attr="real")
        tasks.append(task_obj)

        g_attr = "sim_semantic_segamentation_model"
        task_obj = Task(entry=g_attr, samples=sim_df, meta_attr="sim")
        tasks.append(task_obj)

        return tasks, task_index, samples

~~~

### 3.2 task allocation

The samples will be allocated into two tasks based on whether the sample is in the cities of "aachen, berlin, bochum, bremen, cologne, darmstadt, dusseldorf, erfurt, frankfurt".

~~~pyt
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
        self.task_extractor = task_extractor
        if self.default_origin:
            return samples, [int(self.task_extractor.get(
                self.default_origin))] * len(samples.x)

        cities = [
            "aachen",
            "berlin",
            "bochum",
            "bremen",
            "cologne",
            "darmstadt",
            "dusseldorf",
            "erfurt",
            "frankfurt"]

        sample_origins = []
        for _x in samples.x:
            is_real = False
            for city in cities:
                if city in _x[0]:
                    is_real = True
                    sample_origins.append("real")
                    break
            if not is_real:
                sample_origins.append("sim")

        allocations = [int(self.task_extractor.get(sample_origin))
                       for sample_origin in sample_origins]

        return samples, allocations

~~~

## 4 Result

| rank |        algorithm        |       accuracy      | samples_transfer_ratio |     paradigm     | basemodel |    task_definition     |    task_allocation     | basemodel-learning_rate | basemodel-epochs | task_definition-origins | task_allocation-origins |         time        |                                                          url                                                          |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  1   | rfnet_lifelong_learning | 0.23840559042245466 |         0.5789         | lifelonglearning | BaseModel | TaskDefinitionByOrigin | TaskAllocationByOrigin |          0.0001         |        1         |     ['real', 'sim']     |     ['real', 'sim']     | 2023-03-05 17:40:36 | /ianvs/lifelong_learning_bench/workspace/benchmarkingjob/rfnet_lifelong_learning/773c3610-bb38-11ed-8d45-0242ac110007 |