<<<<<<< HEAD
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

from __future__ import absolute_import, division

import os
import tempfile
import time
import zipfile
import logging

import numpy as np
import random
from tqdm import tqdm
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from core.common.log import LOGGER


from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto


logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="gen")
class BaseModel:

    def __init__(self, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            "/home/icyfeather/models/Qwen2-0.5B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("/home/icyfeather/models/Qwen2-0.5B-Instruct")

    def train(self, train_data, valid_data=None, **kwargs):
        LOGGER.info("BaseModel train")
        

    def save(self, model_path):
        LOGGER.info("BaseModel save")

    def predict(self, data, input_shape=None, **kwargs):
        LOGGER.info("BaseModel predict")
        LOGGER.info(f"Dataset: {data.dataset_name}")
        LOGGER.info(f"Description: {data.description}")
        LOGGER.info(f"Data Level 1 Dim: {data.level_1_dim}")
        LOGGER.info(f"Data Level 2 Dim: {data.level_2_dim}")
        
        answer_list = []
        for line in tqdm(data.x, desc="Processing", unit="question"):
            # 3-shot
            indices = random.sample([i for i, l in enumerate(data.x) if l != line], 3)
            history = []
            for idx in indices:
                history.append({"role": "user", "content": data.x[idx]})
                history.append({"role": "assistant", "content": data.y[idx]})
            history.append({"role": "user", "content": line})
            response = self._infer(history)
            answer_list.append(response)
        return answer_list

    def load(self, model_url=None):
        LOGGER.info("BaseModel load")

    def evaluate(self, data, model_path, **kwargs):
        LOGGER.info("BaseModel evaluate")
        
    def _infer(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature = 0.1,
            top_p = 0.9
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
=======
version https://git-lfs.github.com/spec/v1
oid sha256:35f6f6967af9bc30bd25ac4816f71f2a32cd652a032824b5d0c428524aa88cac
size 3514
>>>>>>> 9676c3e (ya toh aar ya toh par)
