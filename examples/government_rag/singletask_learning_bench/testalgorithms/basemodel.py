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
from core.common.log import LOGGER
from tqdm import tqdm


from transformers import AutoModelForCausalLM, AutoTokenizer

from gov_rag import GovernmentRAG

device = "cuda" # the device to load the model onto


logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="gen")
class BaseModel:

    def __init__(self, **kwargs):
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     "/home/icyfeather/models/Qwen2-0.5B-Instruct",
        #     torch_dtype="auto",
        #     device_map="auto"
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained("/home/icyfeather/models/Qwen2-0.5B-Instruct")
        pass

    def get_model_response(self, prompt):
        # Please install OpenAI SDK first: `pip3 install openai`

        from openai import OpenAI

        client = OpenAI(api_key="sk-0c6cc04bcdc6407997c14570093fddee", base_url="https://api.deepseek.com")
        # client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                # {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        res = response.choices[0].message.content

        return res

    def preprocess(self, **kwargs):
        print("BaseModel preprocess")
        # input('stop here preprocess')
        self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db")
        LOGGER.info("RAG initialized")
        # rag_res = self.rag.query("中华人民共和国宪法是什么？", k=1)
        # print(rag_res)

    def train(self, train_data, valid_data=None, **kwargs):
        print("BaseModel doesn't need to train")
        

    def save(self, model_path):
        print("BaseModel doesn't need to save")

    def predict(self, data, input_shape=None, **kwargs):
        LOGGER.info("BaseModel predict")
        LOGGER.info(f"Dataset: {data.dataset_name}")
        LOGGER.info(f"Description: {data.description}")
        LOGGER.info(f"Data Level 1 Dim: {data.level_1_dim}")
        LOGGER.info(f"Data Level 2 Dim: {data.level_2_dim}")
        LOGGER.info(f"Data Level 3 Dim: {data.level_3_dim}")
        LOGGER.info(f"Data Level 4 Dim: {data.level_4_dim}")

        answer_list = []

        # data.x is query list, data.y is ground truth list, data.level_4 is location list

        # data.level_4 是各个不同的省份
        # 求出来 data.level_4 的全集
        # '陕西省', '山西省', '河北省', '云南省', '海南省', '江西省', '江苏省', '安徽省', '北京市', '山东省', '黑龙江省', '福建省', '南京市', '湖北省', '湖南省', '四川省', '天津市', '广东省', '河南省', '浙江省', '辽宁省', '全国', '重庆市', '广西壮族自治区', '上海市', '甘肃省', '吉林省', '贵州省'
        all_locations = set(data.level_4)
        # print(all_locations)
        # input('stop here predict')

        # 首先用全局的知识库来测一遍所有的省份，省份的位置信息要加到answer里面
        self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db")
        for i in range(len(data.x)):
            relevant_docs = self.rag.query(data.x[i], k=1)
            response = self.get_model_response("在你回答问题之前，你被提供了以下可能相关的信息：" + relevant_docs + "\n现在请你回答问题：" + data.x[i])
            answer_list.append(response + "||" + data.y[i] + "||" + data.level_4[i] + "||" + "[global]")
        
        # 然后对每个省份，用该省份的知识库来测一遍
        
        for i in range(len(data.x)):
            self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db", provinces=[data.level_4[i]])
            relevant_docs = self.rag.query(data.x[i], k=1)
            response = self.get_model_response("在你回答问题之前，你被提供了以下可能相关的信息：" + relevant_docs + "\n现在请你回答问题：" + data.x[i])
            answer_list.append(response + "||" + data.y[i] + "||" + data.level_4[i] + "||" + "[local]")

        # 然后对于每个省份，用除了该省份之外的知识库来测一遍
        for i in range(len(data.x)):
            self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db", provinces=list(all_locations - set([data.level_4[i]])))
            relevant_docs = self.rag.query(data.x[i], k=1)
            response = self.get_model_response("在你回答问题之前，你被提供了以下可能相关的信息：" + relevant_docs + "\n现在请你回答问题：" + data.x[i])
            answer_list.append(response + "||" + data.y[i] + "||" + data.level_4[i] + "||" + "[other]")

        # 最后不用任何知识库，直接用模型来测一遍
        for i in tqdm(range(len(data.x)), desc="Processing [model]"):
            response = self.get_model_response(data.x[i])
            print(response)
            answer_list.append(response + "||" + data.y[i] + "||" + data.level_4[i] + "||" + "[model]")

        return answer_list




    def load(self, model_url=None):
        print("BaseModel load")

    def evaluate(self, data, model_path, **kwargs):
        print("BaseModel evaluate")
        

