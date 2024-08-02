from transformers import AutoModelForCausalLM, AutoTokenizer
from base_llm import BaseLLM

device = "cuda"

class HuggingfaceLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        BaseLLM.__init__(self, **kwargs)

    def load(self, model_url):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,#"/root/autodl-tmp/Qwen/Qwen-1_8B-Chat",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            quantization = self.quantization # Need to align with HF API
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_url,
            trust_remote_code=True
        )
        
    def _infer(self, prompt, system=None):
        if system:   
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response