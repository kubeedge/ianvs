"""Self-checking tests for the Hugging Face mock adapter.

Run directly, with no test framework required::

    python .github/workflows/validator/services/mock_runtime/adapters/test_huggingface_adapter.py

The first group pins the surface that ``examples/llm_simple_qa`` already
depends on, so an extension to the adapter cannot silently change the one
example that is currently activated. The second group covers the surface added
for fine-tuning examples. The third group pins the honesty property: an
attribute the adapter does not model must raise rather than return a
plausible-looking value, because a smoke test that passes on a fabricated
result is worse than one that fails.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.huggingface_adapter import (  # noqa: E402
    _MockBatch,
    _MockModel,
    _MockTokenizer,
)

#: The fixture shipped by examples/llm_simple_qa/scripts/mock_runtime.
LLM_SIMPLE_QA_RESPONSES = {"sequence": ["A", "C", "B", "D", "A", "C", "A", "B", "A", "C"]}


def test_llm_simple_qa_flow_is_unchanged():
    """The activated example's exact call sequence still yields its fixture."""
    tokenizer = _MockTokenizer(LLM_SIMPLE_QA_RESPONSES)
    model = _MockModel()

    decoded = []
    for index in range(10):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "question {}".format(index)}]
        )
        batch = tokenizer(prompt)
        generated = model.generate(batch.input_ids, max_new_tokens=8)
        decoded.append(tokenizer.batch_decode(generated)[0])

    assert decoded == LLM_SIMPLE_QA_RESPONSES["sequence"], decoded


def test_apply_chat_template_joins_message_content():
    tokenizer = _MockTokenizer({})
    joined = tokenizer.apply_chat_template([{"content": "a"}, {"content": "b"}])
    assert joined == "a\nb", joined


def test_generate_appends_one_token_per_sequence():
    assert _MockModel().generate([[0], [0]]) == [[0, 1], [0, 2]]


def test_prompt_responses_take_precedence_over_sequence():
    tokenizer = _MockTokenizer(
        {"prompt_responses": {"special": "Z"}, "sequence": ["A"]}
    )
    tokenizer("special")
    assert tokenizer.batch_decode([[0]]) == ["Z"]


def test_default_is_used_once_the_sequence_is_exhausted():
    tokenizer = _MockTokenizer({"sequence": ["A"], "default": "fallback"})
    tokenizer("one")
    tokenizer("two")
    assert tokenizer.batch_decode([[0], [0]]) == ["A", "fallback"]


def test_batch_supports_attribute_and_mapping_access():
    """Real BatchEncoding supports both; examples use both."""
    batch = _MockBatch(2)
    assert batch["input_ids"] == batch.input_ids
    assert batch["attention_mask"] == batch.attention_mask
    assert "input_ids" in batch
    assert sorted(batch.keys()) == ["attention_mask", "input_ids"]


def test_tokenizer_exposes_special_token_ids():
    """Fine-tuning examples append eos_token_id while building labels."""
    tokenizer = _MockTokenizer({})
    assert isinstance(tokenizer.eos_token_id, int)
    assert isinstance(tokenizer.pad_token_id, int)
    assert tokenizer.eos_token_id != tokenizer.pad_token_id


def test_decode_and_batch_decode_share_one_cursor():
    """Mixing the two must not replay or skip a fixture's responses."""
    tokenizer = _MockTokenizer({"sequence": ["A", "B", "C"]})
    tokenizer("p1")
    tokenizer("p2")
    tokenizer("p3")
    assert tokenizer.decode([0]) == "A"
    assert tokenizer.batch_decode([[0]]) == ["B"]
    assert tokenizer.decode([0]) == "C"


def test_model_training_helpers_are_chainable():
    model = _MockModel()
    assert model.half() is model
    assert model.eval() is model
    assert model.to("cpu") is model
    assert list(model.parameters()) == []


def test_unmodelled_attributes_raise():
    """A mock must fail loudly rather than fake an unsupported call."""
    tokenizer = _MockTokenizer({})
    model = _MockModel()

    for label, call in (
        ("tokenizer.save_pretrained", lambda: tokenizer.save_pretrained("/tmp/x")),
        ("tokenizer.pad", lambda: tokenizer.pad({"input_ids": [[0]]})),
        ("model.state_dict", lambda: model.state_dict()),
        ("model.config", lambda: model.config),
    ):
        try:
            call()
        except AttributeError:
            continue
        raise AssertionError("{} should raise AttributeError".format(label))

    try:
        _MockBatch(1)["bogus"]
    except KeyError:
        pass
    else:
        raise AssertionError("_MockBatch should raise KeyError for unknown keys")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("\n{} passed".format(len(tests)))


if __name__ == "__main__":
    main()
