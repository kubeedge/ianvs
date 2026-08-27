"""Install opt-in LLM smoke-test adapters during Python startup."""

import importlib
import os


def _install_mock_runtime():
    if os.environ.get("IANVS_LLM_MOCK") != "1":
        return

    fixture = importlib.import_module("ianvs_mock_fixture")
    adapters = getattr(fixture, "ADAPTERS", None)
    responses = getattr(fixture, "RESPONSES", None)
    if not isinstance(adapters, (list, tuple)) or not adapters:
        raise RuntimeError("ianvs_mock_fixture.ADAPTERS must be a non-empty list")
    if not isinstance(responses, dict):
        raise RuntimeError("ianvs_mock_fixture.RESPONSES must be a dictionary")

    for adapter_name in adapters:
        if not isinstance(adapter_name, str) or not adapter_name:
            raise RuntimeError("Mock adapter names must be non-empty strings")
        adapter = importlib.import_module("adapters.{}_adapter".format(adapter_name))
        install = getattr(adapter, "install", None)
        if not callable(install):
            raise RuntimeError(
                "Mock adapter '{}' does not provide install(responses)".format(
                    adapter_name
                )
            )
        install(responses.get(adapter_name, {}))


try:
    _install_mock_runtime()
except Exception as exc:
    # Python normally logs sitecustomize errors and continues startup. Mock
    # mode must fail closed so a broken fixture can never fall through to a
    # real model download or provider call.
    raise SystemExit("Ianvs Mock LLM runtime failed: {}".format(exc))
