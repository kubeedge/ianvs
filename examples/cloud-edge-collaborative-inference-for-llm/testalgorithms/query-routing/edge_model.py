# Copyright 2022 The KubeEdge Authors.
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

from __future__ import absolute_import, division, print_function

import os
import tempfile
import time
import zipfile
import logging

import numpy as np
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory

from models import HuggingfaceLLM, VllmLLM, APIBasedLLM

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto


logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

@ClassFactory.register(ClassType.GENERAL, alias="EdgeModel")
class BaseModel:
    """
        This is actually the Edge Model.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_url = kwargs.get("model_name", None)
        self.backend = kwargs.get("backend", "huggingface") 
        self.quantization = kwargs.get("quantization", "full")
        self._set_config()
        # 'backend' means serving framework: "huggingface", "vllm"
        # 'quantization' means quantization mode："full","4-bit","8-bit"
    
    def _set_config(self):
        # Some parameters are passed to Sedna through environment variables 
        parameters = os.environ
        # EdgeModel URL, see at https://github.com/kubeedge/sedna/blob/ac623ab32dc37caa04b9e8480dbe1a8c41c4a6c2/lib/sedna/core/base.py#L132
        parameters["MODEL_URL"] = self.model_url

    def load(self, model_url=None):
        if self.backend == "huggingface":
            self.model = HuggingfaceLLM(model_url, self.quantization)
        elif self.backend == "vllm":
            self.model = VllmLLM(model_url, self.quantization)
        else:
            raise Exception(f"Backend {self.backend} is not supported")
        
        self.model.load(model_url=model_url)
            
        # TODO cloud service must be configured in JointInference

    def predict(self, data, input_shape=None, **kwargs):
        answer_list = []
        for line in data:
            response = self.model.inference(line)
            answer_list.append(response)
        return answer_list