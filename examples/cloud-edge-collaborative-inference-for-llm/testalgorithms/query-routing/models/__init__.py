"""Init models module exposing LLM classes."""
# pylint: disable=wrong-import-position,unused-import,duplicate-code
from .api_llm import APIBasedLLM
from .huggingface_llm import HuggingfaceLLM
try:
    from .vllm_llm import VllmLLM
except ImportError:
    VllmLLM = None
from .base_llm import BaseLLM
try:
    from .eagle_llm import EagleSpecDecModel
except ImportError:
    EagleSpecDecModel = None
