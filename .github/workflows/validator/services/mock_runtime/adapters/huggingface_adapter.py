"""Minimal Transformers adapter used by the llm_simple_qa smoke test."""

from collections.abc import Mapping


class _MockBatch:
    def __init__(self, size):
        self.input_ids = [[0] for _ in range(size)]

    def to(self, _device):
        return self


class _MockModel:
    def generate(self, input_ids, **_kwargs):
        return [list(input_ids[index]) + [index + 1] for index in range(len(input_ids))]


class _MockTokenizer:
    def __init__(self, responses):
        self._responses = responses
        self._response_index = 0
        self._pending_prompts = []

    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(str(message.get("content", "")) for message in messages)

    def __call__(self, prompts, **_kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        self._pending_prompts.extend(str(prompt) for prompt in prompts)
        return _MockBatch(len(prompts))

    def batch_decode(self, generated_ids, **_kwargs):
        return [self._next_response(self._next_prompt()) for _ in generated_ids]

    def _next_prompt(self):
        if self._pending_prompts:
            return self._pending_prompts.pop(0)
        return None

    def _next_response(self, prompt):
        prompt_responses = self._responses.get("prompt_responses", {})
        if isinstance(prompt_responses, Mapping) and prompt in prompt_responses:
            return str(prompt_responses[prompt])

        sequence = self._responses.get("sequence", [])
        if isinstance(sequence, (list, tuple)) and self._response_index < len(sequence):
            response = sequence[self._response_index]
            self._response_index += 1
            return str(response)
        return str(self._responses.get("default", ""))


def install(responses):
    """Patch only the Transformers factories exercised by llm_simple_qa."""
    if not isinstance(responses, Mapping):
        raise TypeError("Hugging Face mock responses must be a mapping")

    import transformers

    def load_model(_cls, *_args, **_kwargs):
        return _MockModel()

    def load_tokenizer(_cls, *_args, **_kwargs):
        return _MockTokenizer(responses)

    transformers.AutoModelForCausalLM.from_pretrained = classmethod(load_model)
    transformers.AutoTokenizer.from_pretrained = classmethod(load_tokenizer)
