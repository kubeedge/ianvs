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
# Platform & dependency guard (MUST be first)
# ---------------------------------------------------------
import platform
import importlib.util

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

# ---------------------------------------------------------
# Safe imports (only reached on supported platforms)
# ---------------------------------------------------------
import os
import torch
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

device = "cuda" if torch.cuda.is_available() else "cpu"


class VllmLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        """Initialize the VllmLLM class"""

        super().__init__(**kwargs)

        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.8)

    def _load(self, model):
        """Load the model via vLLM API"""

        self.model = LLM(
            model=model,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=8192,
            # quantization=self.quantization  # TODO align with vLLM API
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
        )

        # Warmup for accurate metrics
        self.warmup()

    def warmup(self):
        """Warm up the model"""

        try:
            self.model.chat(
                [{"role": "user", "content": "Hello"}],
                self.sampling_params,
                use_tqdm=False,
            )
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    def _infer(self, messages):
        """Run inference using vLLM"""

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
        """Release GPU resources"""

        try:
            destroy_model_parallel()
            destroy_distributed_environment()
            if hasattr(self, "model"):
                del self.model.llm_engine.model_executor
        except Exception as e:
            raise RuntimeError(f"Cleanup failed: {e}")
