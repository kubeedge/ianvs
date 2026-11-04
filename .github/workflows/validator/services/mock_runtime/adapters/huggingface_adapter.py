"""Minimal Transformers adapter used by Ianvs example smoke tests.

The adapter deliberately implements only the Transformers surface that Ianvs
examples actually use. Any attribute an example touches that is not modelled
here raises ``AttributeError`` rather than returning a plausible-looking
value, so a smoke test can never pass by accident on a fabricated result.
"""

from collections.abc import Mapping

#: Token id handed back for every mocked token. It stays 0 because that is
#: what the adapter has always returned, and the llm_simple_qa smoke test
#: depends on those exact ids.
_MOCK_TOKEN_ID = 0

#: Sentinel token ids. Each differs from the others and from _MOCK_TOKEN_ID,
#: so ordinary tokens are never mistaken for padding and an off-by-one in
#: example code stays visible.
_MOCK_BOS_TOKEN_ID = 1
_MOCK_EOS_TOKEN_ID = 2
_MOCK_MASK_TOKEN_ID = 3
_MOCK_PAD_TOKEN_ID = 4


class _MockBatch:
    """Stand-in for ``BatchEncoding``.

    Examples reach into the tokenizer result in two different ways: attribute
    access (``batch.input_ids``) and mapping access
    (``batch["input_ids"]``). Real ``BatchEncoding`` supports both, so the mock
    must too, otherwise an example that tokenises with
    ``add_special_tokens=False`` and then subscripts the result fails inside
    the mock rather than in its own logic.
    """

    def __init__(self, size, length=1):
        self.input_ids = [[_MOCK_TOKEN_ID] * length for _ in range(size)]
        self.attention_mask = [[1] * length for _ in range(size)]

    def to(self, _device):
        return self

    def __getitem__(self, key):
        # Only the encoding keys are items. Delegating to getattr would make
        # method names such as "to" or "keys" look like valid keys.
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.keys()

    def keys(self):
        return ("input_ids", "attention_mask")

    def __iter__(self):
        return iter(self.keys())


class _MockModel:
    def generate(self, input_ids, **_kwargs):
        return [list(input_ids[index]) + [index + 1] for index in range(len(input_ids))]

    def eval(self):
        return self

    def train(self, _mode=True):
        return self

    def half(self):
        return self

    def float(self):
        return self

    def to(self, _device):
        return self

    def parameters(self):
        return iter(())


class _MockTokenizer:
    eos_token_id = _MOCK_EOS_TOKEN_ID
    bos_token_id = _MOCK_BOS_TOKEN_ID
    pad_token_id = _MOCK_PAD_TOKEN_ID
    mask_token_id = _MOCK_MASK_TOKEN_ID
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

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

    def decode(self, _token_ids, **_kwargs):
        """Single-sequence counterpart of :meth:`batch_decode`.

        Examples that generate one sequence at a time consume the same
        response sequence as ``batch_decode``, so both must draw from the same
        cursor to keep a fixture's responses in the order the fixture declares.
        """
        return self._next_response(self._next_prompt())

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
    """Patch ``AutoModelForCausalLM`` and ``AutoTokenizer`` loading.

    Only these two ``from_pretrained`` entry points are replaced. Examples
    that load through any other class, such as ``AutoModel``,
    ``AutoModelForSeq2SeqLM`` or ``BertTokenizer``, are not mocked and will
    still try to load a real model.
    """
    if not isinstance(responses, Mapping):
        raise TypeError("Hugging Face mock responses must be a mapping")

    import transformers

    def load_model(_cls, *_args, **_kwargs):
        return _MockModel()

    def load_tokenizer(_cls, *_args, **_kwargs):
        return _MockTokenizer(responses)

    transformers.AutoModelForCausalLM.from_pretrained = classmethod(load_model)
    transformers.AutoTokenizer.from_pretrained = classmethod(load_tokenizer)
