from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from models.base_llm import BaseLLM

device = "cuda"

class VllmLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        BaseLLM.__init__(self, **kwargs)

    def load(self, model_url):
        self.model = LLM(
            model=model_url, 
            trust_remote_code=True,
            dtype="float16",
            max_model_len=1024
            #quantization=self.quantization # TODO need to align with vllm API
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_url, 
            trust_remote_code=True
        )
    
    def _infer(self, prompt, system=None):
        messages = self.get_message_chain(prompt, system)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

        outputs = self.model.generate([text], sampling_params)
        response = outputs[0].outputs[0].text
        return response