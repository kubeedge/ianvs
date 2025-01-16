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
import time

from openai import OpenAI
from models.base_llm import BaseLLM
from retry import retry

class APIBasedLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        """ Initialize the APIBasedLLM class
        """
        BaseLLM.__init__(self, **kwargs)

        api_key=os.environ.get("OPENAI_API_KEY")
        base_url=os.environ.get("OPENAI_BASE_URL")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _load(self, model):
        """Set the model to be used.

        Parameters
        ----------
        model : str
            Existing model from your OpenAI provider. Example: `gpt-4o-mini`
        """

        self.model = model

    @retry(tries=3, delay=4, max_delay=10)
    def _infer(self, messages):
        """Call the OpenAI API to get the response

        Parameters
        ----------
        messages : list
            OpenAI style message chain. Example:
        ```
        [{"role": "user", "content": "Hello, how are you?"}]
        ```

        Returns
        -------
        dict
            Formatted Response. See `_format_response()` for more details.
        """

        time_to_first_token = 0.0
        internal_token_latency = []
        st = time.perf_counter()
        most_recent_timestamp = st
        generated_text = ""

        stream = self.client.chat.completions.create(
            messages = messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.repetition_penalty,
            stream=True,
            stream_options={"include_usage":True}
        )

        for chunk in stream:
            timestamp = time.perf_counter()
            if time_to_first_token == 0.0:
                time_to_first_token = time.perf_counter() - st
            else:
                internal_token_latency.append(timestamp - most_recent_timestamp)
            most_recent_timestamp = timestamp
            if chunk.choices:
                generated_text += chunk.choices[0].delta.content or ""
            if chunk.usage:
                usage = chunk.usage

        text = generated_text
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        internal_token_latency = sum(internal_token_latency) / len(internal_token_latency)
        throughput = 1 / internal_token_latency

        response = self._format_response(
            text,
            prompt_tokens,
            completion_tokens,
            time_to_first_token,
            internal_token_latency,
            throughput
        )

        return response
