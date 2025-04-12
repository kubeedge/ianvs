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
import torch  
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from models.base_llm import BaseLLM

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

device = "cuda" if torch.cuda.is_available() else "cpu"

class VllmLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        """ Initialize the VllmLLM class

        Parameters
        ----------
        kwargs : dict
            Parameters that are passed to the model. Details can be found in the BaseLLM class.

            Special keys:
            - `tensor_parallel_size`: int, default 1. Number of tensor parallelism.
            - `gpu_memory_utilization`: float, default 0.8. GPU memory utilization.

            See details about special parameters in [vLLM's Named Arguments](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
        """

        BaseLLM.__init__(self, **kwargs)

        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.8)

    def _load(self, model):
        """Load the model via vLLM API

        Parameters
        ----------
        model : str
            Hugging Face style model name. Example: `Qwen/Qwen2.5-0.5B-Instruct`
        """
        self.model = LLM(
            model=model,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len = 8192
            #quantization=self.quantization # TODO need to align with vllm API
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens
        )

        # Warmup to make metrics more accurate
        self.warmup()

    def warmup(self):
        """Warm up the Model for more accurate performance metrics
        """

        try:
            self.model.chat(
                [{"role": "user", "content": "Hello"}],
                self.sampling_params,
                use_tqdm=False
            )
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")
        
    def _infer(self, messages):
        """Call the vLLM Offline Inference API to get the response

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

        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )
        metrics = outputs[0].metrics
        # Completion Text
        text = outputs[0].outputs[0].text
        # Prompt Token Count
        prompt_tokens = len(outputs[0].prompt_token_ids)
        # Completion Token Count
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        # Time to First Token (s)
        time_to_first_token = metrics.first_token_time - metrics.arrival_time
        # Internal Token Latency (s)
        internal_token_latency = (metrics.finished_time - metrics.first_token_time) / completion_tokens
        # Completion Throughput (Token/s)
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

    def cleanup(self):
        """Release the model from GPU
        """
        try:
            destroy_model_parallel()
            destroy_distributed_environment()
            if hasattr(self, "model"):
                del self.model.llm_engine.model_executor
        except Exception as e:
            raise RuntimeError(f"Cleanup failed: {e}")