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

    def __init__(self, **kwargs):
        self.gpu_lock = threading.Lock()
        self.rag = None
        self.get_model_response = self.get_model_response_qianfan
        pass

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
        LOGGER.info(f"Dataset: {data.dataset_name}")
        LOGGER.info(f"Description: {data.description}")

        answer_list = []
        
        # Get location from the directory name
        current_dir = os.path.basename(os.getcwd())
        
        # Create tasks for all queries
        tasks = []
        for i in range(len(data.x)):
            # Add global task
            tasks.append((data.x[i], data.y[i], current_dir, "[global]"))
            # Add local task
            tasks.append((data.x[i], data.y[i], current_dir, "[local]"))
            # Add other task
            tasks.append((data.x[i], data.y[i], current_dir, "[other]"))
            # Add model task
            tasks.append((data.x[i], data.y[i], current_dir, "[model]"))

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

