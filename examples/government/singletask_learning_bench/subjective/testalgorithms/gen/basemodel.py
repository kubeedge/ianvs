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

"""Base model for government subjective singletask learning benchmark."""

from __future__ import absolute_import, division

import logging
import os

# pylint: disable=import-error
import torch
from openai import OpenAI
from sedna.common.class_factory import ClassFactory, ClassType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.common.log import LOGGER

# pylint: disable=invalid-name
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="gen")
class BaseModel:
    """Base model wrapper for causal language model inference & subjective judgment."""

    # pylint: disable=unused-argument
    def __init__(self, **kwargs):
        model_name = os.environ.get("BASE_MODEL_URL") or "Qwen/Qwen2-0.5B-Instruct"
        device_map = "auto" if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # pylint: disable=unused-argument
    def train(self, train_data, valid_data=None, **kwargs):
        """Train model (not implemented)."""
        LOGGER.info("BaseModel train")

    # pylint: disable=unused-argument
    def save(self, model_path):
        """Save model (not implemented)."""
        LOGGER.info("BaseModel save")

    # pylint: disable=unused-argument
    def predict(self, data, input_shape=None, **kwargs):
        """Predict results and evaluate them using DeepSeek LLM judge."""
        LOGGER.info("BaseModel predict")
        LOGGER.info("Dataset: %s", data.dataset_name)
        LOGGER.info("Description: %s", data.description)
        LOGGER.info("Data Level 1 Dim: %s", data.level_1_dim)
        LOGGER.info("Data Level 2 Dim: %s", data.level_2_dim)

        answer_list = []
        for line in tqdm(data.x, desc="Processing", unit="question"):
            history = [{"role": "user", "content": line}]
            response = self._infer(history)
            answer_list.append(response)

        judgement_list = []

        # evaluate by llm
        for index in tqdm(range(len(answer_list)), desc="Evaluating", ascii=False, ncols=75):
            prompt = data.judge_prompts[index] + answer_list[index]
            judgement = self._openai_generate(prompt)
            judgement_list.append(judgement)

        return judgement_list

    # pylint: disable=unused-argument
    def load(self, model_url=None):
        """Load model (not implemented)."""
        LOGGER.info("BaseModel load")

    # pylint: disable=unused-argument
    def evaluate(self, data, model_path, **kwargs):
        """Evaluate model (not implemented)."""
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
            temperature=0.1,
            top_p=0.9
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        return response

    def _openai_generate(self, user_question, system=None):
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("You should set DEEPSEEK_API_KEY in your env.")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

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

# end of file