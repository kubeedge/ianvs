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

import logging
import os
import random

import torch
from tqdm import tqdm
from sedna.common.class_factory import ClassType, ClassFactory
from core.common.log import LOGGER

from transformers import AutoModelForCausalLM, AutoTokenizer


logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="gen")
class BaseModel:

    def __init__(self, **kwargs):
        self.model_name_or_path = os.getenv(
            "GOVERNMENT_BENCH_MODEL",
            "Qwen/Qwen2-0.5B-Instruct",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": "auto"}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        if not torch.cuda.is_available():
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def train(self, train_data, valid_data=None, **kwargs):
        LOGGER.info("BaseModel train")
        

    def save(self, model_path):
        LOGGER.info("BaseModel save")

    def predict(self, data, input_shape=None, **kwargs):
        LOGGER.info("BaseModel predict")
        if hasattr(data, "dataset_name"):
            LOGGER.info(f"Dataset: {data.dataset_name}")
            LOGGER.info(f"Description: {data.description}")
            LOGGER.info(f"Data Level 1 Dim: {data.level_1_dim}")
            LOGGER.info(f"Data Level 2 Dim: {data.level_2_dim}")
        
        questions = list(getattr(data, "x", data))
        labels = list(getattr(data, "y", []))
        answer_list = []
        for line in tqdm(questions, desc="Processing", unit="question"):
            candidates = [
                i for i, question in enumerate(questions)
                if question != line and i < len(labels)
            ]
            indices = random.sample(candidates, min(3, len(candidates)))
            history = []
            for idx in indices:
                history.append({"role": "user", "content": questions[idx]})
                history.append({"role": "assistant", "content": labels[idx]})
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
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
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
