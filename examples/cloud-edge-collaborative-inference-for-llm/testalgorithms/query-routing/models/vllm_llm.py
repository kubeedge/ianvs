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

# ---------------------------------------------------------
# Platform & dependency guard (fail early)
# ---------------------------------------------------------
import platform
import importlib.util
import torch

if platform.system() == "Windows":
    raise RuntimeError(
        "Cloud-edge LLM example requires Linux + GPU. "
        "The vLLM backend is not supported on Windows."
    )

if importlib.util.find_spec("vllm") is None:
    raise RuntimeError(
        "vLLM is required for the cloud-edge LLM example but is not installed. "
        "Please run this example on Linux with GPU support and install vllm."
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "Cloud-edge LLM example requires a GPU, but CUDA is not available. "
        "Please run this example on a machine with a GPU and CUDA installed."
    )

# ---------------------------------------------------------
# Safe imports (only reached on supported platforms)
# ---------------------------------------------------------
import os
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from models.base_llm import BaseLLM

# ---------------------------------------------------------
# Environment variables
# ---------------------------------------------------------
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"


class VllmLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        """
        Initialize the VllmLLM class.

        Parameters
        ----------
        kwargs : dict
            Parameters forwarded to the BaseLLM class.

            Special keys:
            - tensor_parallel_size : int, default 1
                Number of tensor parallelism shards.
            - gpu_memory_utilization : float, default 0.8
                Fraction of GPU memory to use.

        See vLLM named arguments:
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        super().__init__(**kwargs)

        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.8)

    def _load(self, model):
        """
        Load the model using the vLLM API.

        Parameters
        ----------
        model : str
            Hugging Face model identifier, e.g. `Qwen/Qwen2.5-0.5B-Instruct`
        """
        self.model = LLM(
            model=model,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=8192,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
        )

        # Warm up for accurate performance metrics
        self.warmup()

    def warmup(self):
        """
        Perform a warm-up inference to stabilize performance metrics.
        """
        try:
            self.model.chat(
                [{"role": "user", "content": "Hello"}],
                self.sampling_params,
                use_tqdm=False,
            )
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    def _infer(self, messages):
        """
        Run inference using vLLM.

        Parameters
        ----------
        messages : list
            OpenAI-style message list.

        Returns
        -------
        dict
            Formatted inference response including latency and throughput.
        """
        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )

        metrics = outputs[0].metrics
        text = outputs[0].outputs[0].text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        time_to_first_token = metrics.first_token_time - metrics.arrival_time
        internal_token_latency = (
            metrics.finished_time - metrics.first_token_time
        ) / completion_tokens
        throughput = 1 / internal_token_latency

        return self._format_response(
            text,
            prompt_tokens,
            completion_tokens,
            time_to_first_token,
            internal_token_latency,
            throughput,
        )

    def cleanup(self):
        """
        Release GPU and distributed resources.
        """
        try:
            destroy_model_parallel()
            destroy_distributed_environment()
            if hasattr(self, "model"):
                del self.model.llm_engine.model_executor
        except Exception as e:
            raise RuntimeError(f"Cleanup failed: {e}")
