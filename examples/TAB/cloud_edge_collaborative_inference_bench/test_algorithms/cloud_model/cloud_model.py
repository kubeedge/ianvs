# cloud_model.py
import time
import json
import os
import sys
import requests
from dotenv import load_dotenv
from sedna.common.class_factory import ClassFactory, ClassType
import logging


current_dir = os.path.dirname(os.path.abspath(__file__))


try:
    from ..privacy_desensitization import (
        RegexPseudonymization,
        NERMasking,
        DifferentialPrivacy
    )
except Exception:
    bench_root = os.path.abspath(os.path.join(current_dir, "../.."))
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)
    from test_algorithms.privacy_desensitization import (
        RegexPseudonymization,
        NERMasking,
        DifferentialPrivacy
    )


class APIBasedLLM:
    def __init__(self, **kwargs):
        self.provider = kwargs.get("api_provider", "openai")
        self.api_key_env = kwargs.get("api_key_env", "OPENAI_API_KEY")
        self.api_base_url = kwargs.get("api_base_url", "OPENAI_API_URL") or os.getenv(kwargs.get("api_base_url_env"))
        self.api_key = os.getenv(self.api_key_env)
        self.model = None
        
    
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
        except ImportError:
            raise ImportError("please install openai")
    
    def _load(self, model):
        self.model = model
    
    def inference(self, data,** kwargs):
        if not self.model:
            raise ValueError("please use load function")
            
        messages = [{"role": "user", "content": data["query"]}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0)
        )

        
        text_out = None
        prompt_tokens = completion_tokens = total_tokens = None

     
        if hasattr(response, "choices") and response.choices:
            choice0 = response.choices[0]
            if hasattr(choice0, "message") and getattr(choice0.message, "content", None):
                text_out = choice0.message.content
            elif hasattr(choice0, "text") and choice0.text:
                text_out = choice0.text
            # usage 
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

        if text_out is None and isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    msg = c0.get("message", {})
                    text_out = msg.get("content") or c0.get("text")
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

        if text_out is None:

            try:
                text_out = str(response)
            except Exception:
                text_out = ""

        return {
            "generated_text": text_out,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }

@ClassFactory.register(ClassType.GENERAL, alias="CloudModelAPI")
class CloudModelAPI:
   
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_name = kwargs.get("model", "deepseek-chat")
        self.timeout = 30
        self.cache_dir = "./cache/cloud_model_api"
        os.makedirs(self.cache_dir, exist_ok=True)
        
       
        self.privacy_methods = {
            1: RegexPseudonymization(),
            2: NERMasking(),
            3: DifferentialPrivacy(epsilon=1.0)
        }
        
       
        self.model = APIBasedLLM(** kwargs)
        self.load(self.model_name)
        
        self.cache = self._load_cache()
    
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
    
    def load(self, model):
       
        self.model._load(model=model)
    
    def apply_privacy_protection(self, text, complexity_level=None, method=None):
      
        if method is not None:
            #Process by method name (regex/ner/dp)
            method_mapping = {
                "regex": 1,
                "ner": 2,
                "dp": 3
            }
            if method not in method_mapping:
                raise ValueError(f"Invalid privacy protection method: {method}")
            complexity_level = method_mapping[method]
        elif complexity_level is None:
            raise ValueError("Either complexity_level or method must be provided")

        if complexity_level not in self.privacy_methods:
            raise ValueError(f"Invalid privacy protection complexity level: {complexity_level}")
        
        if complexity_level == 1:
            return self.privacy_methods[1].anonymize(text)
        elif complexity_level == 2:
            return self.privacy_methods[2].mask(text)
        elif complexity_level == 3:
            return self.privacy_methods[3].add_noise(text)
    
    def inference(self, data, privacy_method=None, **kwargs):
        
        input_text = None
        input_sensitive_entities = []
        if isinstance(data, dict):
            input_text = data.get("query") or data.get("text")
            se = data.get("sensitive_entities")
            if isinstance(se, list):
                input_sensitive_entities = se
        elif isinstance(data, str):
            input_text = data
        
        else:
            raise KeyError("input must be a dict with 'query' or 'text', or a plain string")
        if not isinstance(input_text, str):
             raise KeyError("input must include 'query' or 'text'")

        cache_key = (
            f"{hash(input_text)}_{privacy_method}_{self.model_name}_"
            f"{kwargs.get('max_tokens', 1024)}_{kwargs.get('temperature', 0.7)}_"
            f"{kwargs.get('top_p', 0.9)}_{kwargs.get('presence_penalty', 0.0)}_"
            f"{kwargs.get('frequency_penalty', 0.0)}"
        )
        if cache_key in self.cache:
            return self.cache[cache_key]
        
  
        start_time = time.time()
        if privacy_method is None:
            protected_text, privacy_time = input_text, 0.0
        else:
            protected_text, privacy_time = self.apply_privacy_protection(input_text, method=privacy_method)
        

        api_result, api_time = self._call_api(protected_text)
        
        total_time = time.time() - start_time
        output = {
            "original_text": input_text,
            "protected_text": protected_text,
            "result": api_result,
            "total_time": total_time,
            "privacy_time": privacy_time,
            "api_time": api_time,
            "privacy_method": privacy_method,
            "model_used": self.model_name,
            "api_endpoint": self.model.api_base_url,
            "sensitive_entities": input_sensitive_entities
        }
        self.cache[cache_key] = output
        self._save_cache()
        
        return output
    
    def _call_api(self, text):
        start_time = time.time()
        result = self.model.inference({"query": text}, **self.kwargs)
        api_time = time.time() - start_time
        return result, api_time
    
    def get_model_info(self):
        return {
            "name": self.model_name,
            "type": "api-based",
            "provider": self.model.provider,
            "endpoint": self.model.api_base_url,
            "supported_privacy_levels": list(self.privacy_methods.keys())
        }
    
    def infer(self, data, context=None):
        privacy_method = context.get("privacy_method") if context else None
        return self.inference(data, privacy_method=privacy_method)