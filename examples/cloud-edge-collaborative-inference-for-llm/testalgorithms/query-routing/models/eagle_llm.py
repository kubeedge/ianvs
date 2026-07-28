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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from models.base_llm import BaseLLM

device = "cuda"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class EagleSpecDecModel(BaseLLM):
    def __init__(self, **kwargs) -> None:
        """ Initialize the HuggingfaceLLM class

        Parameters
        ----------
        kwargs : dict
            Parameters that are passed to the model. Details can be found in the BaseLLM class.
        """
        BaseLLM.__init__(self, **kwargs)
        # breakpoint()

    def _load(self, model):
        from eagle.model.ea_model import EaModel
        # breakpoint()
        self.model = EaModel.from_pretrained(
            base_model_path=self.config.get("model", None),
            ea_model_path=self.config.get("draft_model", None),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=-1
        )
        self.model.eval()

    @staticmethod
    def truncate_list(lst, num):
        if num not in lst:
            return lst
        first_index = lst.index(num)
        return lst[:first_index + 1]

    def _infer(self, messages):
        st = time.perf_counter()
        most_recent_timestamp = st

        prompt = self.model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids=self.model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()

        time_to_first_token = 0
        internal_token_latency = []
        generated_text = ""
        completion_tokens = 0

        prompt_tokens = input_ids.shape[1]
        generate_len = prompt_tokens

        for output_ids in self.model.ea_generate(
            input_ids, 
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_tokens
        ):
            timestamp = time.perf_counter()
            if time_to_first_token == 0:
                time_to_first_token = time.perf_counter() - st
            else:
                internal_token_latency.append(
                    timestamp - most_recent_timestamp
                )
            decode_ids = output_ids[0, generate_len:].tolist()
            decode_ids = self.truncate_list(decode_ids, self.model.tokenizer.eos_token_id)
            chunk = self.model.tokenizer.decode(
                decode_ids, 
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True
            )
            # print(chunk, end="", flush=True)
            most_recent_timestamp = timestamp
            generate_len = output_ids.shape[1]
            generated_text += chunk
        
        text = generated_text
        completion_tokens = output_ids.shape[1] - prompt_tokens
        
        internal_token_latency = sum(internal_token_latency) / completion_tokens
        
        if internal_token_latency != 0:
            throughput = 1 / internal_token_latency
        else:
            throughput = 0
        
        return self._format_response(
            text,
            prompt_tokens,
            completion_tokens,
            time_to_first_token,
            internal_token_latency,
            throughput,
        )
