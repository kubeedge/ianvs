# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hard Example Mining Algorithms"""

import abc
import math
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('ThresholdFilter', 'CrossEntropyFilter', 'IBTFilter')

class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __call__(self, infer_result=None):
        """
        predict function, judge the sample is hard or not.

        Parameters
        ----------
        infer_result : array_like
            prediction result

        Returns
        -------
        is_hard_sample : bool
            `True` means hard sample, `False` means not.
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Check the data in [0,1]."""
        return 0 <= float(data) <= 1
    

@ClassFactory.register(ClassType.HEM, alias="BERT")
class BERTFilter(BaseFilter, abc.ABC):
    def __init__(self, **kwargs):
        # self.classifier = pipeline(
        #     "text-classification", 
        #     model=model_path, 
        #     trust_remote_code=True
        # )
        pass
    
    def _predict(self, data):
        # result = self.classifier(data)
        # return result
        return False

    def __call__(self, data=None) -> bool:
        import random
        res = bool(random.randint(0,1))
        return res#self._predict(data)