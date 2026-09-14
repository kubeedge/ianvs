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

    # Read the embedding model path from an environment variable so it works
    # on any machine. Falls back to the default path if not set.
    DEFAULT_MODEL_NAME = os.environ.get("GOV_RAG_MODEL_PATH", "/home/icyfeather/models/bge-m3")
    DEFAULT_PERSIST_DIR = "./chroma_db"

    def __init__(self, **kwargs):
        self.gpu_lock = threading.Lock()
        self.all_locations = []
        self.get_model_response = self.get_model_response_qianfan

        # Cache RAG instances by their scope key (e.g. "global", "local:Beijing")
        # so each unique province configuration only loads the embedding model
        # and ChromaDB once, instead of reloading for every single query.
        self._rag_cache = {}

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
        # Pre-warm the global RAG instance so it's ready before queries start.
        # This loads the embedding model and vector store once upfront.
        self._get_rag_instance("global")
        LOGGER.info("RAG pre-warmed for global scope")

    def train(self, train_data, valid_data=None, **kwargs):
        print("BaseModel doesn't need to train")
        

    def save(self, model_path):
        print("BaseModel doesn't need to save")

    def _get_rag_instance(self, scope_key, provinces=None):
        """Get or create a cached GovernmentRAG instance for the given scope.

        Each unique scope (e.g. 'global', 'local:Beijing', 'other:Beijing')
        gets its own RAG instance. The first call for a scope loads the
        embedding model and ChromaDB; subsequent calls return the cached
        instance immediately. This avoids the original bug where the model
        was reloaded from scratch for every single query.
        """
        if scope_key not in self._rag_cache:
            self._rag_cache[scope_key] = GovernmentRAG(
                model_name=self.DEFAULT_MODEL_NAME,
                device="cuda",
                persist_directory=self.DEFAULT_PERSIST_DIR,
                provinces=provinces
            )
        return self._rag_cache[scope_key]

    def process_query(self, query: str, ground_truth: str, location: str, rag_type: str) -> str:
        """Process a single query with the specified RAG type.

        Uses cached RAG instances per scope so each province configuration
        only loads once. The gpu_lock protects the vector similarity search
        (a quick GPU operation), not the model loading.
        """
        try:
            if rag_type == "[model]":
                # No RAG needed, just ask the LLM directly
                response = self.get_model_response(query)
            else:
                with self.gpu_lock:
                    if rag_type == "[global]":
                        rag = self._get_rag_instance("global")
                    elif rag_type == "[local]":
                        rag = self._get_rag_instance(f"local:{location}", provinces=[location])
                    else:  # [other]
                        other_provinces = list(set(self.all_locations) - {location})
                        rag = self._get_rag_instance(f"other:{location}", provinces=other_provinces)

                    relevant_docs = rag.query(query, k=1)

                # Build the augmented prompt with retrieved context
                augmented_prompt = (
                    "在你回答问题之前，你被提供了以下可能相关的信息："
                    + relevant_docs
                    + "\n现在请你回答问题："
                    + query
                )
                response = self.get_model_response(augmented_prompt)
            
            return response + "||" + ground_truth + "||" + location + "||" + rag_type
        except Exception as e:
            LOGGER.error(f"Error in process_query: {str(e)}")
            raise e

    def predict(self, data, input_shape=None, **kwargs):
        print("BaseModel predict")
        LOGGER.info("BaseModel predict")

        # Make sure the RAG system is ready before processing queries.
        # Delegates to preprocess() to avoid duplicating init parameters.
        if not self._rag_cache:
            LOGGER.info("RAG not initialized yet, running preprocess...")
            self.preprocess()

        answer_list = []
        
        # Get location from the directory name
        current_dir = os.path.basename(os.getcwd())
        
        # Create tasks for all queries
        tasks = []
        for i in range(len(data.x)):
            tasks.append((data.x[i], data.y[i], current_dir, "[global]"))
            tasks.append((data.x[i], data.y[i], current_dir, "[local]"))
            tasks.append((data.x[i], data.y[i], current_dir, "[other]"))
            tasks.append((data.x[i], data.y[i], current_dir, "[model]"))

        # Process tasks in parallel using ThreadPoolExecutor.
        # RAG instances are cached per scope, so threads only block briefly
        # on the gpu_lock for vector search. The LLM API calls run outside
        # the lock and truly execute in parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_query, query, gt, loc, rag_type) 
                      for query, gt, loc, rag_type in tasks]
            
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

