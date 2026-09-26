# Tests for src/partner/answering.py's Anthropic SDK call.
# Covers the same four required model-response failure modes as
# test_reviewer.py (valid, malformed, refused, timed-out), plus
# generate_answer()'s empty-chunks guard. Uses a fake client (duck-typing
# anthropic.Anthropic's .messages.create) so tests run with no network
# access or real credentials. answer_question()'s retrieval-skip path is
# covered separately in tests/test_answer_question.py against real
# Postgres + real Ollama (free/local), since it needs a real retrieve()
# call, not a fake one.
from __future__ import annotations

from dataclasses import dataclass, field

import anthropic
import httpx
import pytest

from partner.answering import (
    AnswerMalformedError,
    AnswerRefusedError,
    AnswerTimedOutError,
    generate_answer,
)


@dataclass
class FakeToolUseBlock:
    input: dict
    type: str = "tool_use"
    name: str = "submit_grounded_answer"


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeMessage:
    content: list
    usage: FakeUsage
    stop_reason: str = "tool_use"


@dataclass
class FakeMessagesResource:
    response: FakeMessage | None = None
    responses: list[FakeMessage] | None = None
    exception: Exception | None = None
    call_count: int = field(default=0, init=False)
    last_kwargs: dict | None = field(default=None, init=False)

    def create(self, **kwargs: object) -> FakeMessage:
        self.last_kwargs = kwargs
        self.call_count += 1
        if self.exception is not None:
            raise self.exception
        if self.responses is not None:
            index = min(self.call_count - 1, len(self.responses) - 1)
            return self.responses[index]
        assert self.response is not None
        return self.response


@dataclass
class FakeClient:
    messages: FakeMessagesResource


SAMPLE_CHUNKS = [
    {
        "file_name": "BUILD_PLAN.md",
        "chunk_index": 2,
        "content": "Run `uv run pytest` before every commit.",
        "offset": 20,
        "score": 0.71,
    }
]

VALID_ANSWER_INPUT = {
    "can_answer": True,
    "answer": "Run `uv run pytest` before every commit.",
    "citations": [{"file_name": "BUILD_PLAN.md", "chunk_index": 2}],
    "rationale": "chunk_index=2 directly states the testing command.",
}


def test_valid_response_returns_an_answer_result() -> None:
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=VALID_ANSWER_INPUT)],
                usage=FakeUsage(input_tokens=150, output_tokens=40),
            )
        )
    )

    result = generate_answer("How do I run the tests?", SAMPLE_CHUNKS, fake_client)

    assert result.answer.can_answer is True
    assert result.input_tokens == 150
    assert result.output_tokens == 40
    # tool_choice must force the model onto our schema, not leave it optional
    assert fake_client.messages.last_kwargs["tool_choice"] == {
        "type": "tool",
        "name": "submit_grounded_answer",
    }


def test_malformed_tool_input_raises_malformed_error() -> None:
    malformed_input = {**VALID_ANSWER_INPUT, "citations": []}
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=malformed_input)],
                usage=FakeUsage(input_tokens=140, output_tokens=30),
            )
        )
    )

    with pytest.raises(AnswerMalformedError):
        generate_answer("How do I run the tests?", SAMPLE_CHUNKS, fake_client)


def test_malformed_output_is_retried_and_recovers() -> None:
    malformed_input = {**VALID_ANSWER_INPUT, "citations": []}
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            responses=[
                FakeMessage(
                    content=[FakeToolUseBlock(input=malformed_input)],
                    usage=FakeUsage(input_tokens=140, output_tokens=30),
                ),
                FakeMessage(
                    content=[FakeToolUseBlock(input=VALID_ANSWER_INPUT)],
                    usage=FakeUsage(input_tokens=145, output_tokens=35),
                ),
            ]
        )
    )

    result = generate_answer("How do I run the tests?", SAMPLE_CHUNKS, fake_client)

    assert result.answer.can_answer is True
    assert fake_client.messages.call_count == 2


def test_malformed_output_gives_up_after_max_retries() -> None:
    malformed_input = {**VALID_ANSWER_INPUT, "citations": []}
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=malformed_input)],
                usage=FakeUsage(input_tokens=140, output_tokens=30),
            )
        )
    )

    with pytest.raises(AnswerMalformedError):
        generate_answer(
            "How do I run the tests?",
            SAMPLE_CHUNKS,
            fake_client,
            max_malformed_retries=1,
        )

    # 1 initial attempt + 1 retry = 2 calls, then give up
    assert fake_client.messages.call_count == 2


def test_refusal_stop_reason_raises_refused_error() -> None:
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeTextBlock(text="I can't help with that request.")],
                usage=FakeUsage(input_tokens=120, output_tokens=10),
                stop_reason="refusal",
            )
        )
    )

    with pytest.raises(AnswerRefusedError):
        generate_answer("How do I run the tests?", SAMPLE_CHUNKS, fake_client)


def test_no_tool_call_raises_refused_error() -> None:
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeTextBlock(text="Let me think about this instead.")],
                usage=FakeUsage(input_tokens=120, output_tokens=15),
                stop_reason="end_turn",
            )
        )
    )

    with pytest.raises(AnswerRefusedError):
        generate_answer("How do I run the tests?", SAMPLE_CHUNKS, fake_client)


def test_api_timeout_raises_timed_out_error() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    fake_client = FakeClient(
        messages=FakeMessagesResource(exception=anthropic.APITimeoutError(request))
    )

    with pytest.raises(AnswerTimedOutError):
        generate_answer("How do I run the tests?", SAMPLE_CHUNKS, fake_client)


def test_empty_chunks_is_rejected_before_any_api_call() -> None:
    fake_client = FakeClient(messages=FakeMessagesResource())

    with pytest.raises(ValueError, match="non-empty"):
        generate_answer("How do I run the tests?", [], fake_client)

    assert fake_client.messages.last_kwargs is None
