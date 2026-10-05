from .api_llm import APIBasedLLM
from .huggingface_llm import HuggingfaceLLM
# vllm is only installable on Linux with CUDA; keep it optional so the
# 'huggingface' and 'api' backends remain usable on other platforms.
try:
    from .vllm_llm import VllmLLM
except ImportError:
    VllmLLM = None
from .base_llm import BaseLLM
from .eagle_llm import EagleSpecDecModel
