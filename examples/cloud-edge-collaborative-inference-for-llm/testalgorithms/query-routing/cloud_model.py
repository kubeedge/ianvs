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

from models import HuggingfaceLLM, APIBasedLLM, VllmLLM

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

os.environ['BACKEND_TYPE'] = 'TORCH'

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

@ClassFactory.register(ClassType.GENERAL, alias="CloudModel")
class CloudModel:
    def __init__(self, **kwargs):
        # The API KEY and API URL are confidential data and should not be written in yaml.

        self.model = APIBasedLLM(**kwargs)

        self.model.load(model = kwargs.get("model", "gpt-4o-mini"))
    
    def inference(self, data, input_shape=None, **kwargs):
        return self.model.inference(data)
    
    def cleanup(self):
        self.model.save_cache()
        self.model.cleanup()