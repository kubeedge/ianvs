# edge_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import json
import os
from sedna.common.class_factory import ClassFactory, ClassType

class HuggingfaceLLM:
    def __init__(self,** kwargs):
        self.model_name = kwargs.get("model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
    def inference(self, data, **kwargs):
        inputs = self.tokenizer(data["query"], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0)
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": result}


@ClassFactory.register(ClassType.GENERAL, alias="EdgeModel")
class EdgeModel:

    
    def __init__(self,** kwargs):
        self.kwargs = kwargs
        self.model_name = kwargs.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.backend = kwargs.get("backend", "huggingface")
        self.cache_dir = "./cache/edge_model"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Force offline mode to avoid network calls to Hugging Face
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.model = None
        self._set_config()
        self.load()
        

        self.cache = self._load_cache()
    
    def _set_config(self):
        os.environ["MODEL_PATH"] = self.model_name
        os.environ["BACKEND_TYPE"] = self.backend
    
    def load(self, **kwargs):
        if self.backend == "huggingface":
            self.model = HuggingfaceLLM(**self.kwargs)
        else:
            raise ValueError(f"unsupport: {self.backend}")
    
    def _load_cache(self):

        cache_file = os.path.join(self.cache_dir, "cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):

        cache_file = os.path.join(self.cache_dir, "cache.json")
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def predict(self, data, **kwargs):

        query_hash = hash(data["query"])
        if query_hash in self.cache:
            return self.cache[query_hash]
        

        start_time = time.time()
        result = self.model.inference(data,** {**self.kwargs,** kwargs})
        inference_time = time.time() - start_time
        

        output = {
            "result": result,
            "inference_time": inference_time,
            "model_used": self.model_name,
            "backend": self.backend
        }
        self.cache[query_hash] = output
        self._save_cache()
        
        return output
    
    def get_model_info(self):
        return {
            "name": self.model_name,
            "backend": self.backend,
            "parameters": self.kwargs
        }
    
    def infer(self, data, context=None):
        "" "ianvs Standard Reasoning Interface" ""
        return self.predict(data)