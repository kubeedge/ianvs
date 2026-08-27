"""Deterministic OpenAI SDK adapter for opt-in LLM smoke tests.

The adapter replaces both the current client API and the legacy module-level
chat API.  It intentionally implements only text generation; no request can
reach an OpenAI-compatible service while the mock runtime is enabled.
"""

import json
import os
import re
import threading
import warnings
from collections.abc import Mapping
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


_MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_TIMEOUT_SECONDS = 5
_OPENAI_COMPATIBLE_NPM = "@ai-sdk/openai-compatible"
_ENDPOINT_PLACEHOLDER = re.compile(r"\$\{[^}]+\}")


def _normalise_endpoint(endpoint):
    """Return a stable representation for comparing API base URLs."""
    parts = urlsplit(str(endpoint).strip())
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")
    )


def _endpoint_pattern(endpoint):
    """Compile a models.dev endpoint, including its ${ENV} placeholders."""
    endpoint = _normalise_endpoint(endpoint)
    chunks = _ENDPOINT_PLACEHOLDER.split(endpoint)
    placeholders = _ENDPOINT_PLACEHOLDER.findall(endpoint)
    pattern = "^" + re.escape(chunks[0])
    for _placeholder, chunk in zip(placeholders, chunks[1:]):
        pattern += r"[^/]+" + re.escape(chunk)
    return re.compile(pattern + "$")


@lru_cache(maxsize=1)
def _models_dev_endpoints():
    """Fetch and index models.dev data once for this Python process."""
    try:
        request = Request(_MODELS_DEV_URL, headers={"User-Agent": "ianvs-validator"})
        with urlopen(request, timeout=_MODELS_DEV_TIMEOUT_SECONDS) as response:
            providers = json.load(response)
        if not isinstance(providers, Mapping):
            raise ValueError("the response root is not an object")
    except (OSError, TimeoutError, ValueError) as exc:
        warnings.warn(
            "Unable to validate the OpenAI endpoint and model because {} did not "
            "respond with valid data: {}".format(_MODELS_DEV_URL, exc),
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    endpoint_models = {}
    for provider in providers.values():
        if not isinstance(provider, Mapping):
            continue
        provider_endpoint = provider.get("api")
        provider_npm = provider.get("npm")
        models = provider.get("models", {})
        if not isinstance(models, Mapping):
            continue

        for model_key, model_data in models.items():
            model_endpoint = None
            model_npm = None
            if isinstance(model_data, Mapping):
                model_provider = model_data.get("provider")
                if isinstance(model_provider, Mapping):
                    model_endpoint = model_provider.get("api")
                    model_npm = model_provider.get("npm")

            if (model_npm or provider_npm) != _OPENAI_COMPATIBLE_NPM:
                continue

            endpoint = model_endpoint or provider_endpoint
            if not isinstance(endpoint, str) or not endpoint.strip():
                continue

            model_ids = endpoint_models.setdefault(endpoint, set())
            model_ids.add(str(model_key))
            if isinstance(model_data, Mapping) and model_data.get("id") is not None:
                model_ids.add(str(model_data["id"]))

    return [
        (_endpoint_pattern(endpoint), models)
        for endpoint, models in endpoint_models.items()
    ]


def _validate_endpoint_model(endpoint, model):
    endpoint_models = _models_dev_endpoints()
    if endpoint_models is None:
        return

    normalised_endpoint = _normalise_endpoint(endpoint)
    matched_models = set()
    for pattern, models in endpoint_models:
        if pattern.fullmatch(normalised_endpoint):
            matched_models.update(models)

    if not matched_models:
        raise RuntimeError(
            "OpenAI endpoint {!r} was not found in {}. Model {!r} could not be "
            "validated.".format(str(endpoint), _MODELS_DEV_URL, model),
        )

    if str(model) not in matched_models:
        raise RuntimeError(
            "Model {!r} is not available at OpenAI endpoint {!r} according to {}."
            .format(model, str(endpoint), _MODELS_DEV_URL)
        )


def _configured_endpoint(base_url=None):
    return base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"


def _resolve_endpoint(endpoint):
    return endpoint() if callable(endpoint) else endpoint


class _MockObject(dict):
    """Small attribute-accessible mapping resembling OpenAI response models."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__

    def model_dump(self, **_kwargs):
        return _to_plain_dict(self)

    def dict(self, **_kwargs):
        return self.model_dump()

    def json(self, **kwargs):
        return json.dumps(self.model_dump(), **kwargs)


def _to_plain_dict(value):
    if isinstance(value, Mapping):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_dict(item) for item in value]
    return value


def _message_text(message):
    if isinstance(message, Mapping):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    # Multimodal-style content may contain several typed blocks.  Text blocks
    # are enough to make fixture matching useful without modelling the SDK.
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, Mapping):
                parts.append(str(part.get("text", part.get("content", ""))))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _prompt_candidates(kwargs):
    messages = kwargs.get("messages")
    if isinstance(messages, (list, tuple)):
        parts = [_message_text(message) for message in messages]
        candidates = ["\n".join(parts)]
        if parts:
            candidates.append(parts[-1])
        return candidates

    value = kwargs.get("prompt", kwargs.get("input", ""))
    if isinstance(value, (list, tuple)):
        parts = [_message_text(item) for item in value]
        return ["\n".join(parts)] + ([parts[-1]] if parts else [])
    return [str(value)]


class _ResponseSelector:
    def __init__(self, responses):
        self._responses = responses
        self._response_index = 0
        self._lock = threading.Lock()

    def next(self, kwargs):
        prompt_responses = self._responses.get("prompt_responses", {})
        if isinstance(prompt_responses, Mapping):
            for prompt in _prompt_candidates(kwargs):
                if prompt in prompt_responses:
                    return str(prompt_responses[prompt])

        with self._lock:
            sequence = self._responses.get("sequence", [])
            if isinstance(sequence, (list, tuple)) and self._response_index < len(
                sequence
            ):
                response = sequence[self._response_index]
                self._response_index += 1
                return str(response)
        return str(self._responses.get("default", ""))


def _usage(kwargs, content):
    prompt = _prompt_candidates(kwargs)[0]
    prompt_tokens = len(prompt.split())
    completion_tokens = len(content.split())
    return _MockObject(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _chat_response(content, kwargs):
    return _MockObject(
        id="ianvs-mock-chat-completion",
        object="chat.completion",
        model=str(kwargs.get("model", "ianvs-mock")),
        choices=[
            _MockObject(
                index=0,
                finish_reason="stop",
                message=_MockObject(role="assistant", content=content),
                text=content,
            )
        ],
        usage=_usage(kwargs, content),
    )


def _completion_response(content, kwargs):
    return _MockObject(
        id="ianvs-mock-completion",
        object="text_completion",
        model=str(kwargs.get("model", "ianvs-mock")),
        choices=[_MockObject(index=0, finish_reason="stop", text=content)],
        usage=_usage(kwargs, content),
    )


def _response_api_response(content, kwargs):
    return _MockObject(
        id="ianvs-mock-response",
        object="response",
        model=str(kwargs.get("model", "ianvs-mock")),
        status="completed",
        output_text=content,
        output=[
            _MockObject(
                type="message",
                role="assistant",
                content=[_MockObject(type="output_text", text=content)],
            )
        ],
        usage=_usage(kwargs, content),
    )


class _SyncStream:
    def __init__(self, content, kwargs):
        self._chunks = iter(
            [
                _MockObject(
                    id="ianvs-mock-chat-completion",
                    object="chat.completion.chunk",
                    model=str(kwargs.get("model", "ianvs-mock")),
                    choices=[
                        _MockObject(
                            index=0,
                            finish_reason=None,
                            delta=_MockObject(role="assistant", content=content),
                        )
                    ],
                    usage=None,
                ),
                _MockObject(
                    id="ianvs-mock-chat-completion",
                    object="chat.completion.chunk",
                    model=str(kwargs.get("model", "ianvs-mock")),
                    choices=[],
                    usage=_usage(kwargs, content),
                ),
            ]
        )

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


class _AsyncStream:
    def __init__(self, stream):
        self._stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._stream)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False


class _ChatCompletions:
    def __init__(self, selector, endpoint):
        self._selector = selector
        self._endpoint = endpoint

    def create(self, *args, **kwargs):
        if args:
            raise TypeError("Mock chat completions accept keyword arguments only")
        _validate_endpoint_model(_resolve_endpoint(self._endpoint), kwargs.get("model"))
        content = self._selector.next(kwargs)
        if kwargs.get("stream"):
            return _SyncStream(content, kwargs)
        return _chat_response(content, kwargs)


class _Completions:
    def __init__(self, selector, endpoint):
        self._selector = selector
        self._endpoint = endpoint

    def create(self, *args, **kwargs):
        if args:
            raise TypeError("Mock completions accept keyword arguments only")
        _validate_endpoint_model(_resolve_endpoint(self._endpoint), kwargs.get("model"))
        content = self._selector.next(kwargs)
        return _completion_response(content, kwargs)


class _Responses:
    def __init__(self, selector, endpoint):
        self._selector = selector
        self._endpoint = endpoint

    def create(self, *args, **kwargs):
        if args:
            raise TypeError("Mock responses accept keyword arguments only")
        _validate_endpoint_model(_resolve_endpoint(self._endpoint), kwargs.get("model"))
        content = self._selector.next(kwargs)
        return _response_api_response(content, kwargs)


class _AsyncChatCompletions(_ChatCompletions):
    async def create(self, *args, **kwargs):
        result = super().create(*args, **kwargs)
        if isinstance(result, _SyncStream):
            return _AsyncStream(result)
        return result


class _AsyncCompletions(_Completions):
    async def create(self, *args, **kwargs):
        return super().create(*args, **kwargs)


class _AsyncResponses(_Responses):
    async def create(self, *args, **kwargs):
        return super().create(*args, **kwargs)


def _client(selector, endpoint, asynchronous=False):
    client = _MockObject()
    chat_completions = (
        _AsyncChatCompletions(selector, endpoint)
        if asynchronous
        else _ChatCompletions(selector, endpoint)
    )
    client.chat = _MockObject(completions=chat_completions)
    client.completions = (
        _AsyncCompletions(selector, endpoint)
        if asynchronous
        else _Completions(selector, endpoint)
    )
    client.responses = (
        _AsyncResponses(selector, endpoint)
        if asynchronous
        else _Responses(selector, endpoint)
    )
    return client


def install(responses):
    """Replace OpenAI text-generation entry points with deterministic mocks."""
    if not isinstance(responses, Mapping):
        raise TypeError("OpenAI mock responses must be a mapping")

    import openai

    selector = _ResponseSelector(responses)

    class MockOpenAI:
        def __new__(cls, *_args, **_kwargs):
            return _client(selector, _configured_endpoint(_kwargs.get("base_url")))

    class MockAsyncOpenAI:
        def __new__(cls, *_args, **_kwargs):
            return _client(
                selector,
                _configured_endpoint(_kwargs.get("base_url")),
                asynchronous=True,
            )

    def module_endpoint():
        return _configured_endpoint(
            getattr(openai, "base_url", None) or getattr(openai, "api_base", None)
        )

    # Current SDK entry points.
    openai.OpenAI = MockOpenAI
    openai.Client = MockOpenAI
    openai.AsyncOpenAI = MockAsyncOpenAI
    openai.AsyncClient = MockAsyncOpenAI
    openai.chat = _MockObject(
        completions=_ChatCompletions(selector, module_endpoint)
    )
    openai.completions = _Completions(selector, module_endpoint)
    openai.responses = _Responses(selector, module_endpoint)

    # Pre-1.0 SDK entry points, still used by some Ianvs examples.
    openai.ChatCompletion = _ChatCompletions(selector, module_endpoint)
    openai.Completion = _Completions(selector, module_endpoint)
