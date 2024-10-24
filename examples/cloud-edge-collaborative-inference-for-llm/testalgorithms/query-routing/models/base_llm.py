import os
import json
# from evals import extract_prediction

def extract_prediction(input_string):
    # 检查输入是否为空或只包含非字母字符
    if not input_string or not any(char.isalpha() for char in input_string):
        return None
    # 倒序遍历字符串，找到最后一个字母
    for char in reversed(input_string):
        if 'A' <= char <= 'D':
            return char
    # 如果没有找到字母，返回None
    return None


class BaseLLM:
    def __init__(self, **kwargs) -> None:
        self.config = kwargs
        self._parse_kwargs(**kwargs)
        self.is_cache_loaded = False
        self.model_loaded = False

    def load(self):
        raise NotImplementedError

    def _parse_kwargs(self, **kwargs):
        self.model_name = kwargs.get("model", None)
        self.quantization = kwargs.get("quantization", "full")
        self.temperature = kwargs.get("temperature", 0.8)
        self.top_p = kwargs.get("top_p", 0.8)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.05)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.use_cache = kwargs.get("use_cache", True)

    def inference(self, data):

        if isinstance(data, list):
            return [self._infer(line) for line in data]

        elif isinstance(data, str):
            return self._infer(data)

        elif isinstance(data, dict):

            gold = data.get("gold", None)
            query = data.get("query")

            messages = self.get_message_chain(query)
            question = messages[-1]["content"]

            if self.use_cache:
                response = self._try_cache(question)
                if response is not None:
                    return response

            if not self.model_loaded:
                self.load(self.model_name)
                self.model_loaded = True

            response = self._infer(messages)

            prediction = extract_prediction(response.get("completion"))

            response["prediction"] = prediction

            if self.use_cache:
                self._update_cache(question, response, prediction, gold)

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
                        self.cache_hash = {item["query"]:item['response'] for item in cache["result"]}
        self.is_cache_loaded = True

    def _try_cache(self, question):

        if not self.is_cache_loaded:
            self._load_cache()

        return self.cache_hash.get(question, None)

    def _update_cache(self, question, response, prediction, gold):

        if not self.is_cache_loaded:
            self._load_cache()

        new_item = {
            "query": question,
            "response": response,
            "prediction": prediction,
            "gold": gold
        }

        self.cache_hash[question] = response

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
