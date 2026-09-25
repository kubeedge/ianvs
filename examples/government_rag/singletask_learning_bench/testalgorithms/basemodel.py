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
import json
import tempfile
import time
import zipfile
import logging
import concurrent.futures
import threading
from typing import List, Tuple

import numpy as np
import torch
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
        self.gpu_lock = threading.Lock()
        self.rag = None
        self.all_locations = []
        self._cached_query_locations = None
        self.get_model_response = self.get_model_response_qianfan

    def get_model_response_deepseek(self, prompt):
        # Please install OpenAI SDK first: `pip3 install openai`

        from openai import OpenAI

        client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

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

    def get_model_response_siliconflow(self, prompt):
        import requests

        url = "https://api.siliconflow.cn/v1/chat/completions"

        payload = {
            "model": "THUDM/chatglm3-6b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "min_p": 0.05,
            "stop": None,
            "temperature": 0.1,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }
        headers = {
            "Authorization": "Bearer <token>",  # Replace with your actual token
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        print(response_data)
        
        # Extract the response content from the API response
        # Note: You might need to adjust this based on the actual response structure
        res = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        res = self.get_last_letter(res)
        
        return res

    def get_model_response_qianfan(self, prompt):
        import requests
        import json

        def get_access_token():
            url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=[应用API Key]&client_secret=[应用Secret Key]"
            
            payload = json.dumps("")
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.request("POST", url, headers=headers, data=payload)
            return response.json().get("access_token")

        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed?access_token=" + get_access_token()
        
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        response_data = response.json()
        
        # Extract the response content from the API response
        res = response_data.get('result', '')
        res = self.get_first_letter(res)

        
        return res

    def preprocess(self, **kwargs):
        print("BaseModel preprocess")
        # input('stop here preprocess')
        self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db")
        LOGGER.info("RAG initialized")

    def train(self, train_data, valid_data=None, **kwargs):
        print("BaseModel doesn't need to train")

    def _load_locations_from_dataset(self, data):
        """
        Read the original JSONL dataset to build a mapping of query -> location.

        Each line in the JSONL has a 'level_4_dim' field that contains the
        province name (e.g. 'Shanghai', 'Beijing'). We use this as the
        source of truth for location, instead of relying on os.getcwd()
        which just returns the project root directory.

        Results are cached after the first call to avoid re-parsing the
        dataset file on every predict() invocation.
        """
        # Return cached result if we already parsed the dataset once
        if self._cached_query_locations is not None:
            return self._cached_query_locations

        query_to_location = {}
        all_locations = set()

        # The train_data path points to the JSONL file that was configured
        # in testenv.yaml. Try to find it from the dataset object.
        dataset_path = getattr(data, 'data_file', None)

        # If data object doesn't expose the file path, we can still
        # build the mapping by scanning query text against known provinces.
        # But first, try reading the JSONL directly if possible.
        if dataset_path and os.path.isfile(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    query = entry.get('query', '')
                    location = entry.get('level_4_dim', 'Unknown')
                    query_to_location[query] = location
                    # Only track valid province names, skip unknowns
                    if location and location != 'Unknown':
                        all_locations.add(location)
        else:
            # Fallback: figure out the location from the query text.
            # The government_rag dataset queries mention the province name
            # in Chinese, so we scan for known province keywords.
            zh_to_en = {
                "北京": "Beijing", "上海": "Shanghai", "天津": "Tianjin",
                "重庆": "Chongqing", "河北": "Hebei", "山西": "Shanxi",
                "辽宁": "Liaoning", "吉林": "Jilin", "黑龙江": "Heilongjiang",
                "江苏": "Jiangsu", "浙江": "Zhejiang", "安徽": "Anhui",
                "福建": "Fujian", "江西": "Jiangxi", "山东": "Shandong",
                "河南": "Henan", "湖北": "Hubei", "湖南": "Hunan",
                "广东": "Guangdong", "海南": "Hainan", "四川": "Sichuan",
                "贵州": "Guizhou", "云南": "Yunnan", "陕西": "Shaanxi",
                "甘肃": "Gansu", "青海": "Qinghai", "台湾": "Taiwan",
                "内蒙古": "Inner Mongolia", "广西": "Guangxi", "西藏": "Tibet",
                "宁夏": "Ningxia", "新疆": "Xinjiang", "香港": "Hong Kong",
                "澳门": "Macau"
            }

            for i in range(len(data.x)):
                query = data.x[i]
                location = "Unknown"
                for zh_name, en_name in zh_to_en.items():
                    if zh_name in query:
                        location = en_name
                        all_locations.add(en_name)
                        break
                query_to_location[query] = location

        result = (query_to_location, list(all_locations))
        self._cached_query_locations = result
        return result

    def save(self, model_path):
        print("BaseModel doesn't need to save")

    def process_query(self, query: str, ground_truth: str, location: str, rag_type: str) -> str:
        """Process a single query with the specified RAG type."""
        try:
            if rag_type == "[model]":
                response = self.get_model_response(query)
            else:
                with self.gpu_lock:
                    if rag_type == "[global]":
                        if self.rag is None:
                            self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db")
                    elif rag_type == "[local]":
                        self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db", provinces=[location])
                    else:  # [other]
                        all_locations = set(self.all_locations)
                        self.rag = GovernmentRAG(model_name="/home/icyfeather/models/bge-m3", device="cuda", persist_directory="./chroma_db", provinces=list(all_locations - set([location])))
                    
                    relevant_docs = self.rag.query(query, k=1)
                    
                    # Clear GPU cache after query
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                response = self.get_model_response("在你回答问题之前，你被提供了以下可能相关的信息：" + relevant_docs + "\n现在请你回答问题：" + query)
            
            return response + "||" + ground_truth + "||" + location + "||" + rag_type
        except Exception as e:
            LOGGER.error(f"Error in process_query: {str(e)}")
            # Clear GPU cache in case of error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    def predict(self, data, input_shape=None, **kwargs):
        print("BaseModel predict")
        LOGGER.info("BaseModel predict")

        answer_list = []

        # Build a lookup of query -> province from the dataset itself.
        # The old code used os.path.basename(os.getcwd()) which always
        # returned the project root name (e.g. "ianvs") instead of an
        # actual province, breaking all location-based RAG filtering.
        # Results are cached internally so repeated calls don't re-parse.
        query_to_location, all_locations = self._load_locations_from_dataset(data)
        self.all_locations = all_locations

        # Create tasks for all queries
        tasks = []
        for i in range(len(data.x)):
            query = data.x[i]
            location = query_to_location.get(query, "Unknown")

            tasks.append((query, data.y[i], location, "[global]"))
            tasks.append((query, data.y[i], location, "[local]"))
            tasks.append((query, data.y[i], location, "[other]"))
            tasks.append((query, data.y[i], location, "[model]"))

        # Process tasks in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # Reduced number of workers
            futures = [executor.submit(self.process_query, query, gt, loc, rag_type) 
                      for query, gt, loc, rag_type in tasks]
            
            # Use tqdm to show progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing queries"):
                try:
                    result = future.result()
                    answer_list.append(result)
                except Exception as e:
                    LOGGER.error(f"Error processing query: {e}")

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return answer_list

    def load(self, model_url=None):
        print("BaseModel load")

    def evaluate(self, data, model_path, **kwargs):
        print("BaseModel evaluate")
        
    def get_last_letter(self, text: str) -> str:
        """
        Extract the last English letter from a string.
        
        Args:
            text (str): Input string
            
        Returns:
            str: The last English letter in the string, or empty string if no English letters found
        """
        # Find all English letters in the string
        letters = [char for char in text if char.isalpha() and char.isascii()]
        # Return the last letter if any exist, otherwise return empty string
        return letters[-1] if letters else ""

    def get_first_letter(self, text: str) -> str:
        """
        Extract the first English letter from a string.
        
        Args:
            text (str): Input string
            
        Returns:
            str: The first English letter in the string, or empty string if no English letters found
        """
        # Find all English letters in the string
        letters = [char for char in text if char.isalpha() and char.isascii()]
        # Return the first letter if any exist, otherwise return empty string
        return letters[0] if letters else ""

