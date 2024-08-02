from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from models.base_llm import BaseLLM
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

device = "cuda"

class VllmLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        BaseLLM.__init__(self, **kwargs)

        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.8)
        
    def load(self, model_url):
        self.model = LLM(
            model=model_url, 
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
        self.model.chat(
            [{"role": "user", "content": "Hello"}], 
            self.sampling_params,
            use_tqdm=False
        )

    def _infer(self, question, system_prompt):
        messages = self.get_message_chain(question, system_prompt)

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
        destroy_model_parallel()
        destroy_distributed_environment()

        del self.model.llm_engine.model_executor