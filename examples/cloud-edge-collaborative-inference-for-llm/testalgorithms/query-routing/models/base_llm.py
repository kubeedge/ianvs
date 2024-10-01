import os
import json

class BaseLLM:
    def __init__(self, **kwargs) -> None:
        self.config = kwargs
        self.parse_kwargs(**kwargs)
        self.is_cache_loaded = False
        
    def load(self):
        raise NotImplementedError
    
    def parse_kwargs(self, **kwargs):
        self.quantization = kwargs.get("quantization", "full")
        self.temperature = kwargs.get("temperature", 0.8)
        self.top_p = kwargs.get("top_p", 0.8)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.05)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.use_cache = kwargs.get("use_cache", True)
    
    def inference(self, data):
        
        if isinstance(data, dict):
            return [self._infer(line) for line in data]
        
        elif isinstance(data, str):
            return self._infer(data)
        
        elif isinstance(data, list):
            # from viztracer import VizTracer
            # import sys
            # with VizTracer(output_file="optional.json") as tracer:
            # question, system_prompt = self.parse_input(data)
            messages = data
            system_prompt = messages[0]["content"]
            question = messages[-1]["content"]

            if self.use_cache:
                response = self.try_cache(question, system_prompt)
                if response is not None:
                    return response

            response = self._infer(messages)
            if self.use_cache:
                self._update_cache(question, system_prompt, response)

            # sys.exit(0)
            return response
        
        else:
            raise ValueError(f"DataType {type(data)} is not supported, it must be `list` or `str` or `dict`")
        
    def get_message_chain(self, question, system = None):
        if system:   
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": question}
            ]
        else:
            messages = [
                {"role": "user", "content": question}
            ]
        
        return messages
    
    def validate_input(self, data):
        expected_format = """{'question':'Lorem', "prompts": {infer_system_prompt:"Lorem"}}"""

        if "question" not in data:
            raise ValueError(f"Missing Key 'question' in data, data should have format like {expected_format}")
        if "prompts" not in data:
            raise ValueError(f"Missing Key 'prompts' in data, data should have format like {expected_format}")

    def parse_input(self,data):
        self.validate_input(data)
        # data should have format like: 
        # {"question":"Lorem", "prompt": {infer_system_prompt:"Lorem"}}
        question = data.get("question")
        prompt_dict = data.get("prompts")
        system_prompt = prompt_dict.get("infer_system_prompt", "")

        return question, system_prompt

    def _infer(self, messages):
        raise NotImplementedError
    
    def _format_response(self, text, prompt_tokens, completion_tokens, time_to_first_token, internal_token_latency, throughput):

        total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        perf = {
            "time_to_first_token": time_to_first_token,
            "internal_token_latency": internal_token_latency,
            "throughput": throughput
        }

        resposne = {
            "completion": text, 
            "usage":usage,
            "perf":perf
        }

        return resposne
    
    def _load_cache(self):
        self.cache = None
        self.cache_hash = {}
        self.cache_models = []

        cache_file = os.path.join(os.environ["RESULT_SAVED_URL"], "cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self.cache_models = json.load(f) 
                for cache in self.cache_models:
                    if cache["config"] == self.config:
                        self.cache = cache
                        self.cache_hash = {(item["question"], item["system_prompt"]):item['response'] for item in cache["result"]}
        self.is_cache_loaded = True

    def try_cache(self, question, system_prompt):
        
        if not self.is_cache_loaded:
            self._load_cache()

        return self.cache_hash.get((question, system_prompt), None)
    
    def _update_cache(self, question, system_prompt, response):
        
        if not self.is_cache_loaded:
            self._load_cache()

        new_item = {
            "question": question,
            "system_prompt": system_prompt,
            "response": response
        }

        self.cache_hash[(question, system_prompt)] = response

        if self.cache is not None:
            self.cache["result"].append(new_item)
        else:
            self.cache = {"config": self.config, "result": [new_item]}
            self.cache_models.append(self.cache)
        
    def save_cache(self):
        
        cache_file = os.path.join(os.environ["RESULT_SAVED_URL"], "cache.json")

        if self.is_cache_loaded:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_models, f, indent=4)

    def cleanup(self):
        pass
