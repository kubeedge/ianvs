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

"""Base model for government_rag benchmark."""

from __future__ import absolute_import, division, print_function

import os
import sys
import logging
import concurrent.futures
import threading

# pylint: disable=import-error
import torch
from sedna.common.class_factory import ClassType, ClassFactory
from tqdm import tqdm

from core.common.log import LOGGER

# Dynamically resolve import path of gov_rag module
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# pylint: disable=wrong-import-position, wrong-import-order
from gov_rag import GovernmentRAG

# pylint: disable=invalid-name
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="gen")
class BaseModel:
    """BaseModel wrapper that combines LLM reasoning with retrieval augmentation."""

    # pylint: disable=unused-argument
    def __init__(self, **kwargs):
        self.gpu_lock = threading.Lock()
        self.rag = None
        self.get_model_response = self.get_model_response_qianfan

        self.base_path = os.environ.get("RAG_BASE_PATH") or "./dataset/gov_rag"
        self.all_locations = []
        dataset_dir = os.path.join(self.base_path, "dataset")
        if os.path.exists(dataset_dir):
            self.all_locations = [
                d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ]

    def get_model_response_deepseek(self, prompt):
        """Invoke DeepSeek LLM API response."""
        # pylint: disable=import-outside-toplevel
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY") or "<DeepSeek API Key>"
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        res = response.choices[0].message.content
        return res

    def get_model_response_siliconflow(self, prompt):
        """Invoke SiliconFlow API response."""
        # pylint: disable=import-outside-toplevel
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
        token = os.getenv("SILICONFLOW_API_KEY") or "<token>"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response_data = response.json()
        res = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        res = self.get_last_letter(res)
        return res

    def get_model_response_qianfan(self, prompt):
        """Invoke Baidu Qianfan API response."""
        # pylint: disable=import-outside-toplevel
        import requests
        import json

        def get_access_token():
            client_id = os.getenv("QIANFAN_ACCESS_KEY") or "[应用API Key]"
            client_secret = os.getenv("QIANFAN_SECRET_KEY") or "[应用Secret Key]"
            url = (
                "https://aip.baidubce.com/oauth/2.0/token"
                f"?grant_type=client_credentials&client_id={client_id}"
                f"&client_secret={client_secret}"
            )

            payload = json.dumps("")
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload, timeout=60)
            return response.json().get("access_token")

        token_val = get_access_token()
        url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
            f"?access_token={token_val}"
        )

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

        response = requests.request("POST", url, headers=headers, data=payload, timeout=60)
        response_data = response.json()
        res = response_data.get('result', '')
        res = self.get_first_letter(res)
        return res

    # pylint: disable=unused-argument
    def preprocess(self, **kwargs):
        """Preprocess knowledge base retrieval vector index."""
        print("BaseModel preprocess")
        model_name = os.environ.get("RAG_MODEL_URL") or "BAAI/bge-large-zh-v1.5"
        self.rag = GovernmentRAG(
            base_path=self.base_path,
            model_name=model_name,
            device=device,
            persist_directory="./chroma_db"
        )
        LOGGER.info("RAG initialized")

    # pylint: disable=unused-argument
    def train(self, train_data, valid_data=None, **kwargs):
        """Train (not implemented)."""
        print("BaseModel doesn't need to train")

    # pylint: disable=unused-argument
    def save(self, model_path):
        """Save model (not implemented)."""
        print("BaseModel doesn't need to save")

    def process_query(self, query: str, ground_truth: str, location: str, rag_type: str) -> str:
        """Process a single query with the specified RAG type."""
        # pylint: disable=broad-exception-caught
        try:
            model_name = os.environ.get("RAG_MODEL_URL") or "BAAI/bge-large-zh-v1.5"
            if rag_type == "[model]":
                response = self.get_model_response(query)
            else:
                with self.gpu_lock:
                    if rag_type == "[global]":
                        if self.rag is None:
                            self.rag = GovernmentRAG(
                                base_path=self.base_path,
                                model_name=model_name,
                                device=device,
                                persist_directory="./chroma_db"
                            )
                    elif rag_type == "[local]":
                        self.rag = GovernmentRAG(
                            base_path=self.base_path,
                            model_name=model_name,
                            device=device,
                            persist_directory="./chroma_db",
                            provinces=[location]
                        )
                    else:  # [other]
                        all_locations = set(self.all_locations)
                        provinces = list(all_locations - set([location]))
                        self.rag = GovernmentRAG(
                            base_path=self.base_path,
                            model_name=model_name,
                            device=device,
                            persist_directory="./chroma_db",
                            provinces=provinces
                        )

                    relevant_docs = self.rag.query(query, k=1)

                    # Clear GPU cache after query
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                response = self.get_model_response(
                    "在你回答问题之前，你被提供了以下可能相关的信息："
                    + relevant_docs
                    + "\n现在请你回答问题："
                    + query
                )

            return response + "||" + ground_truth + "||" + location + "||" + rag_type
        except Exception as e:
            LOGGER.error("Error in process_query: %s", str(e))
            # Clear GPU cache in case of error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    # pylint: disable=unused-argument
    def predict(self, data, input_shape=None, **kwargs):
        """Predict results for all queries in parallel."""
        print("BaseModel predict")
        LOGGER.info("BaseModel predict")
        LOGGER.info("Dataset: %s", data.dataset_name)
        LOGGER.info("Description: %s", data.description)

        answer_list = []

        # Get location from the directory name
        current_dir = os.path.basename(os.getcwd())

        # Create tasks for all queries
        tasks = []
        for i, x_val in enumerate(data.x):
            y_val = data.y[i]
            # Add global task
            tasks.append((x_val, y_val, current_dir, "[global]"))
            # Add local task
            tasks.append((x_val, y_val, current_dir, "[local]"))
            # Add other task
            tasks.append((x_val, y_val, current_dir, "[other]"))
            # Add model task
            tasks.append((x_val, y_val, current_dir, "[model]"))

        # Process tasks in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.process_query, query, gt, loc, r_type)
                for query, gt, loc, r_type in tasks
            ]

            # Use tqdm to show progress
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing queries"
            ):
                try:
                    result = future.result()
                    answer_list.append(result)
                # pylint: disable=broad-exception-caught
                except Exception as e:
                    LOGGER.error("Error processing query: %s", e)

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return answer_list

    # pylint: disable=unused-argument
    def load(self, model_url=None):
        """Load model (not implemented)."""
        print("BaseModel load")

    # pylint: disable=unused-argument
    def evaluate(self, data, model_path, **kwargs):
        """Evaluate model (not implemented)."""
        print("BaseModel evaluate")

    def get_last_letter(self, text: str) -> str:
        """Extract the last English letter from a string."""
        letters = [char for char in text if char.isalpha() and char.isascii()]
        return letters[-1] if letters else ""

    def get_first_letter(self, text: str) -> str:
        """Extract the first English letter from a string."""
        letters = [char for char in text if char.isalpha() and char.isascii()]
        return letters[0] if letters else ""
