"""Run streamed inference with the quantized GGUF and collect request metrics."""

import logging
import os
import time

import psutil
from llama_cpp import Llama
from sedna.common.class_factory import ClassFactory, ClassType

logging.getLogger().setLevel(logging.INFO)


@ClassFactory.register(ClassType.GENERAL, alias="LlamaCppModel")
class LlamaCppModel:
    """Adapter for Ianvs to load and benchmark a compressed llama.cpp model."""

    def __init__(self, **kwargs):
        self.model_path = kwargs.get("model_path")
        if not self.model_path:
            raise ValueError("Model path is required.")

        self.model = None
        self.model_options = {
            "n_ctx": kwargs.get("n_ctx", 512),
            "n_gpu_layers": kwargs.get("n_gpu_layers", 0),
            "n_threads": kwargs.get("n_threads", 4),
            "n_threads_batch": kwargs.get("n_threads", 4),
            "seed": kwargs.get("seed", -1),
            "logits_all": kwargs.get("logits_all", False),
            "vocab_only": kwargs.get("vocab_only", False),
            "use_mlock": kwargs.get("use_mlock", False),
            "embedding": kwargs.get("embedding", False),
        }

    def preprocess(self, data=None, **_kwargs):
        """Pass input text through to the model."""
        return data

    def predict(self, data, input_shape=None, **kwargs):  # pylint: disable=unused-argument
        """Generate responses and measure latency, prefill time and RSS."""
        if self.model is None:
            raise RuntimeError("Load the compressed GGUF before running inference.")
        data = data[:10]
        process = psutil.Process(os.getpid())

        results = []

        for prompt in data:
            prompt_start_time = time.perf_counter()

            output_stream = self.model.create_chat_completion(
                messages=[{"role": "user", "content": str(prompt)}],
                max_tokens=kwargs.get("max_tokens", 32),
                stop=kwargs.get("stop"),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 40),
                repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                stream=True,
            )

            generated_text = ""
            prefill_latency = 0.0

            # Some streaming chunks carry metadata but no generated text.
            for chunk in output_stream:
                text = chunk["choices"][0]["delta"].get("content") or ""
                if text and not generated_text:
                    prefill_latency = (time.perf_counter() - prompt_start_time) * 1000
                generated_text += text

            if not generated_text:
                raise RuntimeError(
                    "Inference returned no generated text for a benchmark prompt."
                )

            results.append({
                "generated_text": generated_text,
                "total_time": (time.perf_counter() - prompt_start_time) * 1000,
                "prefill_latency": prefill_latency,
                "mem_usage": process.memory_info().rss,
            })

        return {"results": results}

    def postprocess(self, predict_output=None, **_kwargs):
        """Pass the prediction output through unchanged."""
        return predict_output

    def evaluate(self, data, model_path=None, **kwargs):
        """Evaluate the loaded model with the requested metric."""
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

        if not callable(metric_func):
            raise ValueError(f"Metric function {metric_name} is not callable or not provided.")
        return {metric_name: metric_func(None, predict_dict["results"])}

    def save(self, model_path):  # pylint: disable=unused-argument
        """Leave the pretrained checkpoint on disk unchanged."""

    def load(self, model_url):
        """Load the output from Ianvs' compression step for inference."""
        if (
            not model_url
            or os.path.realpath(model_url) == os.path.realpath(self.model_path)
            or not os.path.isfile(model_url)
        ):
            raise RuntimeError(
                "Compression did not produce a GGUF model. Check the FP16 source "
                "model and llama-quantize setup in the example README."
            )

        if self.model is not None:
            self.model.close()
        logging.info("Loading compressed GGUF: %s", model_url)
        self.model = Llama(model_path=model_url, **self.model_options)

    def train(self, train_data, valid_data=None, **kwargs):  # pylint: disable=unused-argument
        """Skip training and use the pretrained weights for inference."""
        logging.info("Training step bypassed: Using pre-trained weights for LLM inference.")
        return kwargs.get("model_path", "")
