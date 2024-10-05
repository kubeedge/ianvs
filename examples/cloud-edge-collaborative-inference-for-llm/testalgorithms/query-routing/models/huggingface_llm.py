from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from models.base_llm import BaseLLM
from threading import Thread
import time
import os

device = "cuda"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class HuggingfaceLLM(BaseLLM):
    def __init__(self, **kwargs) -> None:
        BaseLLM.__init__(self, **kwargs)

    def load(self, model_url):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            # quantization = self.quantization # Need to align with HF API
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_url,
            trust_remote_code=True
        )

    def _infer(self, messages):
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