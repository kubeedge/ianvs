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
import random
from tqdm import tqdm
from sedna.common.config import Context
from sedna.common.class_factory import ClassType, ClassFactory
from openai import OpenAI

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
        print("BaseModel train")
        

    def save(self, model_path):
        print("BaseModel save")

    def predict(self, data, input_shape=None, **kwargs):
        print("BaseModel predict")
        
        if 'infer_system_prompt' in data.prompts:
            infer_system_prompt = data.prompts['infer_system_prompt']
        
        answer_list = []
        for line in tqdm(data.question, desc="Processing", unit="question"):
            history = []
            query = line.split('||')[0]
            if infer_system_prompt:
                history.append({"role": "system", "content": infer_system_prompt})
            history.append({"role": "user", "content": query})
            print(query)
            response = self._infer(history)
            print(response)
            answer_list.append(response)

        judgement_list = []

        # evaluate by llm
        for index in tqdm(range(len(answer_list)), desc="Evaluating", ascii=False, ncols=75):
            prompt = data.prompts['eval_user_template'].replace('{question}', data.question[index].split('||')[0]).replace('{reference}', data.question[index].split('||')[1]).replace('{answer}', answer_list[index])
            print(prompt)
            judgement = self._openai_generate(prompt)
            print(judgement)
            judgement_list.append(judgement)

        return judgement_list

    def load(self, model_url=None):
        print("BaseModel load")

    def evaluate(self, data, model_path, **kwargs):
        print("BaseModel evaluate")
        
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
        # print(response)
        # raise ValueError('stop')
        return response


    def _openai_generate(self, user_question, system=None):
        client = OpenAI(api_key="", base_url="https://api.deepseek.com")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_question})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )

        res = response.choices[0].message.content

        return res