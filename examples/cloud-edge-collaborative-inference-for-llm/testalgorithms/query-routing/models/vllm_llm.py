from vllm import LLM, SamplingParams
from base_llm import BaseLLM

device = "cuda"

class VllmLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        BaseLLM.__init__(self, **kwargs)

    def load(self, model_url):
        self.model = LLM(
            model=model_url, 
            trust_remote_code=True,
            quantization=self.quantization # TODO need to align with vllm API
        )
    
    def _infer(self, prompt, system=None):
        sampling_params = SamplingParams(
            temperature=0.8, 
            top_p=0.95, 
            max_tokens=2048
        )
        outputs = self.model.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text
        return response