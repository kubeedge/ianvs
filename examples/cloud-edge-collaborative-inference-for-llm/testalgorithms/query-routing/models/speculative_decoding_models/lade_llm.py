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
from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from models.base_llm import BaseLLM

device = "cuda"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class LadeSpecDecLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        import lade 

        lade.augment_all()
        #For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7 
        lade.config_lade(LEVEL=7, WINDOW_SIZE=20, GUESS_SET_SIZE=20, DEBUG=1)

        """ Initialize the HuggingfaceLLM class

        Parameters
        ----------
        kwargs : dict
            Parameters that are passed to the model. Details can be found in the BaseLLM class.
        """
        BaseLLM.__init__(self, **kwargs)

    def _load(self, model):
        """Load the model via Hugging Face API

        Parameters
        ----------
        model : str
            Hugging Face style model name. Example: `Qwen/Qwen2.5-0.5B-Instruct`
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True
        )

    def _infer(self, messages):
        """Call the transformers inference API to get the response

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

        st = time.perf_counter()
        most_recent_timestamp = st

        # messages = self.get_message_chain(question, system_prompt)

        streamer = TextIteratorStreamer(self.tokenizer)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )

        thread.start()
        time_to_first_token = 0
        internal_token_latency = []
        generated_text = ""
        completion_tokens = 0

        for chunk in streamer:
            timestamp = time.perf_counter()
            if time_to_first_token == 0:
                time_to_first_token = time.perf_counter() - st
            else:
                internal_token_latency.append(timestamp - most_recent_timestamp)
            most_recent_timestamp = timestamp
            generated_text += chunk
            completion_tokens += 1

        text = generated_text.replace("<|im_end|>", "")
        prompt_tokens = len(model_inputs.input_ids[0])
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