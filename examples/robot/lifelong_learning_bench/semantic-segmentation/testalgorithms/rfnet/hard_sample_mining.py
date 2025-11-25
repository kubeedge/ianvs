# @inproceedings{zhou2021learning,
# author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan},
# title = {Learning Placeholders for Open-Set Recognition},
# booktitle = {CVPR},
# pages = {4401-4410},
# year = {2021}
# }
from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType
import random

__all__ = ('HardSampleMining',)

@ClassFactory.register(ClassType.UTD, alias="HardSampleMining")
class HardSampleMining:
    """
    Dividing the data set based on whether the sample is a known class of data.

    Parameters
    ----------
    model_path: string
        Path of the model
    """
    def __init__(self, **kwargs):
        self.threhold = kwargs.get("threhold")

    def __call__(self,
                 samples: BaseDataSource, scores, **kwargs) -> Tuple[BaseDataSource,
                                                   BaseDataSource]:
        '''
        Parameters
        ----------
        samples : BaseDataSource
            inference samples

        Returns
        -------
        seen_task_samples: BaseDataSource
        unseen_task_samples: BaseDataSource
        '''
        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)

        if scores[0] > self.threhold:
            print(f"found easy sample, confidence score: {scores[0]}")
            seen_task_samples.x = samples.x
            unseen_task_samples.x = []

        else:
            print(f"found hard sample, confidence score: {scores[0]}")
            seen_task_samples.x = []
            unseen_task_samples.x = samples.x

        return seen_task_samples, unseen_task_samples
