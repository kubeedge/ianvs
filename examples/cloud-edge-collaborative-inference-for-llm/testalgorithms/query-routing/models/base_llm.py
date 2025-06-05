# Copyright 2024 The KubeEdge Authors.
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

import os
import json

def extract_prediction(input_string):
    """Extract the prediction from the completion. This function is used when caching the responses.
    """
    if not input_string or not any(char.isalpha() for char in input_string):
        return None
    # Find the last letter in the string
    for char in reversed(input_string):
        if 'A' <= char <= 'D':
            return char
    return None


class BaseLLM:
    def __init__(self, **kwargs) -> None:
        """ Initialize the BaseLLM class

        Parameters
        ----------
        kwargs : dict
            Parameters that are passed to the model. For details, see `_parse_kwargs()`
        """
        self.config = kwargs
        self._parse_kwargs(**kwargs)
        self.is_cache_loaded = False
        self.model_loaded = False

    def _load(self):
        """Interface for Model Loading

        Raises
        ------
        NotImplementedError
            When the method is not implemented
        """
        raise NotImplementedError


    def _infer(self, messages):
        """Interface for Model Inference

        Parameters
        ----------
        messages : list
            OpenAI style message chain. Example:
        ```
        [{"role": "user", "content": "Hello, how are you?"}]
        ```

        Raises
        ------
        NotImplementedError
            When the method is not implemented
        """
        raise NotImplementedError

    def _parse_kwargs(self, **kwargs):
        """Parse the kwargs and set the attributes

        Parameters
        ----------
        kwargs : dict
            Parameters that are passed to the model. Possible keys are:
            - `model`: str, default None. Model name
            - `temperature`: float, default 0.8. Temperature for sampling
            - `top_p`: float, default 0.8. Top p for sampling
            - `repetition_penalty`: float, default 1.05. Repetition penalty
            - `max_tokens`: int, default 512. Maximum tokens to generate
            - `use_cache`: bool, default True. Whether to use reponse cache
        """

        self.model_name = kwargs.get("model", None)
        self.temperature = kwargs.get("temperature", 0.8)
        self.top_p = kwargs.get("top_p", 0.8)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.05)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.use_cache = kwargs.get("use_cache", True)

    def inference(self, data):
        """Inference the model

        Parameters
        ----------
        data : dict
            The input data. Example:
            ```
            # With Gold Answer (For special uses like OracleRouter)
            {"query": "What is the capital of China?", "gold": "A"}
            # Without Gold Answer
            {"query": "What is the capital of China?"}
            ```

        Returns
        -------
        dict
            Formatted Response. See `_format_response()` for more details.

        Raises
        ------
        ValueError
            If the data is not a dict
        """

        if isinstance(data, dict):
            gold = data.get("gold", None)
            query = data.get("query")

            messages = self.get_message_chain(query)
            question = messages[-1]["content"]

            if self.use_cache:
                response = self._try_cache(question)
                if response is not None:
                    return response

            if not self.model_loaded:
                self._load(self.model_name)
                self.model_loaded = True

            response = self._infer(messages)

            prediction = extract_prediction(response.get("completion"))

            response["prediction"] = prediction

            if self.use_cache:
                self._update_cache(question, response, prediction, gold)

            return response

        else:
            raise ValueError(f"DataType {type(data)} is not supported, it must be `dict`")

    def get_message_chain(self, question, system = "You are a helpful assistant."):
        """Get the OpenAI Chat style message chain

        Parameters
        ----------
        question : str
            User prompt.
        system : str, optional
            System Prompt, by default None

        Returns
        -------
        list
            OpenAI Chat style message chain.
        """

        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": question}
            ]
        else:
            messages = [
                {"role": "user", "content": question}
            ]

        return messages


    def _format_response(self, text, prompt_tokens, completion_tokens, time_to_first_token, internal_token_latency, throughput):
        """Format the response

        Parameters
        ----------
        text : str
            The completion text
        prompt_tokens : int
            The number of tokens in the prompt
        completion_tokens : int
            The number of tokens in the completion
        time_to_first_token : float
            The time consumed to generate the first token. Unit: s(seconds)
        internal_token_latency : float
            The average time consumed to generate a token. Unit: s(seconds)
        throughput : float
            The throughput of the completion. Unit: tokens/s

        Returns
        -------
        dict
            Example:
            ```
            {
                "completion": "A",
                "usage": {
                    "prompt_tokens": 505,
                    "completion_tokens": 1,
                    "total_tokens": 506
                },
                "perf": {
                    "time_to_first_token": 0.6393,
                    "internal_token_latency": 0.0005,
                    "throughput": 1750.6698
                }
            }
            ```
        """

        total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        perf = {
            "time_to_first_token": time_to_first_token,
            "internal_token_latency": internal_token_latency,
            "throughput": throughput
        }

        resposne = {
            "completion": text,
            "usage":usage,
            "perf":perf
        }

        return resposne

    def _load_cache(self):
        """Load cached Responses from `$RESULT_SAVED_URL/cache.json`.
        """
        self.cache = None
        self.cache_hash = {}
        self.cache_models = []

        cache_file = os.path.join(os.environ["RESULT_SAVED_URL"], "cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache_models = json.load(f)
                for cache in self.cache_models:
                    if cache["config"] == self.config:
                        self.cache = cache
                        self.cache_hash = {item["query"]:item['response'] for item in cache["result"]}
        self.is_cache_loaded = True

    def _try_cache(self, question):
        """Try to get the response from cache

        Parameters
        ----------
        question : str
            User prompt

        Returns
        -------
        dict
            If the question is found in cache, return the Formatted Response. Otherwise, return None.
        """

        if not self.is_cache_loaded:
            self._load_cache()

        return self.cache_hash.get(question, None)

    def _update_cache(self, question, response, prediction, gold):
        """Update the cache with the new item

        Parameters
        ----------
        question : str
            User prompt
        response : dict
            Formatted Response. See `_format_response()` for more details.
        prediction : str
            The prediction extracted from the response
        gold : str
            The gold answer for the question
        """

        if not self.is_cache_loaded:
            self._load_cache()

        new_item = {
            "query": question,
            "response": response,
            "prediction": prediction,
            "gold": gold
        }

        self.cache_hash[question] = response

        if self.cache is not None:
            self.cache["result"].append(new_item)
        else:
            self.cache = {"config": self.config, "result": [new_item]}
            self.cache_models.append(self.cache)

    def save_cache(self):
        """Save the cache to `$RESULT_SAVED_URL/cache.json`.
        """

        cache_file = os.path.join(os.environ["RESULT_SAVED_URL"], "cache.json")

        if self.is_cache_loaded:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_models, f, indent=4)

    def cleanup(self):
        """Default Cleanup Method to release resources
        """
        pass