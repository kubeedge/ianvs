from sedna.common.class_factory import ClassFactory, ClassType
from llama_cpp import Llama
from contextlib import redirect_stderr
import os
import psutil
import time
import io
import statistics

@ClassFactory.register(ClassType.GENERAL, alias="LlamaCppModel")
class LlamaCppModel:
    def __init__(self, **kwargs):
        """
        init llama-cpp
        """
        model_path = kwargs.get("model_path")
        if not model_path:
            raise ValueError("Model path is required.")
        quantization_type = kwargs.get("quantization_type", None)
        if quantization_type:
            print(f"Using quantization type: {quantization_type}")
        # Init LLM model
        self.model = Llama(
            model_path=model_path,
            n_ctx=kwargs.get("n_ctx", 512),
            n_gpu_layers=kwargs.get("n_gpu_layers", 0),
            seed=kwargs.get("seed", -1),
            f16_kv=kwargs.get("f16_kv", True),
            logits_all=kwargs.get("logits_all", False),
            vocab_only=kwargs.get("vocab_only", False),
            use_mlock=kwargs.get("use_mlock", False),
            embedding=kwargs.get("embedding", False),
        )

    def predict(self, data, input_shape=None, **kwargs):
        print("=============================")
        print(len(data))
        print("=============================")
        data = data[:10]
        process = psutil.Process(os.getpid())
        start_time = time.time()

        results = []
        total_times = []
        prefill_latencies = []
        mem_usages = []

        for prompt in data:
            prompt_start_time = time.time()
            
            f = io.StringIO()
            with redirect_stderr(f):
                output = self.model(
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", 32),
                    stop=kwargs.get("stop", ["Q:", "\n"]),
                    echo=kwargs.get("echo", True),
                    temperature=kwargs.get("temperature", 0.8),
                    top_p=kwargs.get("top_p", 0.95),
                    top_k=kwargs.get("top_k", 40),
                    repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                )
            stdout_output = f.getvalue()

            # parse timing info
            timings = self._parse_timings(stdout_output)
            prefill_latency = timings.get('prompt_eval_time', 0.0)  # ms
            generated_text = output['choices'][0]['text']

            prompt_end_time = time.time()
            prompt_total_time = (prompt_end_time - prompt_start_time) * 1000  # convert to ms

            result_with_time = {
                "generated_text": generated_text,
                "total_time": prompt_total_time,
                "prefill_latency": prefill_latency,
                "mem_usage":process.memory_info().rss,
            }

            results.append(result_with_time)

        predict_dict = {
            "results": results,
        }

        return predict_dict

    def _parse_timings(self, stdout_output):
        import re
        timings = {}
        print("================================")
        print(stdout_output)
        print("================================")
        for line in stdout_output.split('\n'):
            match = re.match(r'llama_print_timings:\s*(.+?)\s*=\s*([0-9\.]+)\s*ms', line)
            if match:
                key = match.group(1).strip()
                value = float(match.group(2))

                key = key.lower().replace(' ', '_')
                timings[key] = value
                print(f"Captured timing: {key} = {value}")
            else:
                print("No match found for this line.")

        return timings

    def evaluate(self, data, model_path=None, **kwargs):
        """
        evaluate model
        """
        if data is None or data.x is None:
            raise ValueError("Evaluation data is None.")

        if model_path:
            self.load(model_path)

        # do predict
        predict_dict = self.predict(data.x, **kwargs)

        # compute metrics
        metric = kwargs.get("metric")
        if metric is None:
            raise ValueError("No metric provided in kwargs.")
        
        metric_name, metric_func = metric  

        if callable(metric_func):
            metric_value = metric_func(None, predict_dict["results"])
            return {metric_name: metric_value}
        else:
            raise ValueError(f"Metric function {metric_name} is not callable or not provided.")
    
    def save(self, model_path):
        pass

    def load(self, model_url):
        pass

    def train(self, train_data, valid_data=None, **kwargs):
        return