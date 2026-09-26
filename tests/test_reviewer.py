# Tests for src/partner/reviewer.py's Anthropic SDK call.
# Covers the four required model-response failure modes (docs/TASKS.md
# "AI Checkpoint 1"): valid, malformed, refused, timed-out -- plus the
# status guard and API-key handling. Uses a fake client (duck-typing
# anthropic.Anthropic's .messages.create) so tests run with no network
# access or real credentials.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx
import pytest

from partner import config
from partner.models import Session
from partner.reviewer import (
    ReviewMalformedError,
    ReviewRefusedError,
    ReviewTimedOutError,
    build_client,
    generate_review_draft,
)


@dataclass
class FakeToolUseBlock:
    input: dict
    type: str = "tool_use"
    name: str = "submit_session_review"


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
    # If set, overrides `response` and returns one entry per call, in
    # order (repeating the last entry if create() is called more times
    # than there are responses) -- lets a test simulate a retry recovering.
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


def make_completed_session() -> Session:
    session = Session(
        goal="Build a calculator",
        project_path=Path("calculator"),
        constraints="Use Python standard library only",
        acceptance_criteria="All tests pass",
        evidence="python -m pytest: 14 passed",
        explanation_checkpoint="I understand the model, storage, and CLI.",
    )
    session.complete()
    return session


VALID_DRAFT_INPUT = {
    "acceptance_criteria_assessment": {
        "met": True,
        "rationale": "Evidence shows 14/14 tests passing.",
        "cited_fields": ["evidence"],
    },
    "constraint_compliance_assessment": {
        "violated": False,
        "rationale": "No third-party library is mentioned in evidence.",
        "cited_fields": ["constraints", "evidence"],
    },
    "strengths": [
        {
            "summary": "14 of 14 tests passed, matching the acceptance criteria.",
            "cited_fields": ["evidence", "acceptance_criteria"],
        }
    ],
    "risks": [],
    "follow_ups": [],
    "confidence": "high",
    "confidence_rationale": "Evidence directly supports the criteria.",
}


def test_valid_response_returns_a_review_result() -> None:
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=VALID_DRAFT_INPUT)],
                usage=FakeUsage(input_tokens=120, output_tokens=80),
            )
        )
    )

    result = generate_review_draft(make_completed_session(), fake_client)

    assert result.draft.confidence == "high"
    assert result.input_tokens == 120
    assert result.output_tokens == 80
    # tool_choice must force the model onto our schema, not leave it optional
    assert fake_client.messages.last_kwargs["tool_choice"] == {
        "type": "tool",
        "name": "submit_session_review",
    }


def test_malformed_tool_input_raises_malformed_error() -> None:
    malformed_input = {
        **VALID_DRAFT_INPUT,
        "strengths": [{"summary": "Looks fine.", "cited_fields": []}],
    }
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=malformed_input)],
                usage=FakeUsage(input_tokens=100, output_tokens=60),
            )
        )
    )

    with pytest.raises(ReviewMalformedError):
        generate_review_draft(make_completed_session(), fake_client)


def test_malformed_output_is_retried_and_recovers() -> None:
    malformed_input = {
        **VALID_DRAFT_INPUT,
        "strengths": [{"summary": "Looks fine.", "cited_fields": []}],
    }
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            responses=[
                FakeMessage(
                    content=[FakeToolUseBlock(input=malformed_input)],
                    usage=FakeUsage(input_tokens=100, output_tokens=60),
                ),
                FakeMessage(
                    content=[FakeToolUseBlock(input=VALID_DRAFT_INPUT)],
                    usage=FakeUsage(input_tokens=105, output_tokens=65),
                ),
            ]
        )
    )

    result = generate_review_draft(make_completed_session(), fake_client)

    assert result.draft.confidence == "high"
    assert fake_client.messages.call_count == 2


def test_malformed_output_gives_up_after_max_retries() -> None:
    malformed_input = {
        **VALID_DRAFT_INPUT,
        "strengths": [{"summary": "Looks fine.", "cited_fields": []}],
    }
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=malformed_input)],
                usage=FakeUsage(input_tokens=100, output_tokens=60),
            )
        )
    )

    with pytest.raises(ReviewMalformedError):
        generate_review_draft(
            make_completed_session(), fake_client, max_malformed_retries=1
        )

    # 1 initial attempt + 1 retry = 2 calls, then give up
    assert fake_client.messages.call_count == 2


def test_refusal_stop_reason_raises_refused_error() -> None:
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeTextBlock(text="I can't help with that request.")],
                usage=FakeUsage(input_tokens=90, output_tokens=10),
                stop_reason="refusal",
            )
        )
    )

    with pytest.raises(ReviewRefusedError):
        generate_review_draft(make_completed_session(), fake_client)


def test_no_tool_call_raises_refused_error() -> None:
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeTextBlock(text="Let me think about this instead.")],
                usage=FakeUsage(input_tokens=90, output_tokens=15),
                stop_reason="end_turn",
            )
        )
    )

    with pytest.raises(ReviewRefusedError):
        generate_review_draft(make_completed_session(), fake_client)


def test_api_timeout_raises_timed_out_error() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    fake_client = FakeClient(
        messages=FakeMessagesResource(exception=anthropic.APITimeoutError(request))
    )

    with pytest.raises(ReviewTimedOutError):
        generate_review_draft(make_completed_session(), fake_client)


def test_active_session_is_rejected_before_any_api_call() -> None:
    session = make_completed_session()
    session.status = session.status.__class__.ACTIVE
    fake_client = FakeClient(messages=FakeMessagesResource())

    with pytest.raises(ValueError, match="not 'completed'"):
        generate_review_draft(session, fake_client)

    assert fake_client.messages.last_kwargs is None


def test_build_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Must not depend on whether a real .env file happens to exist in the
    # repo -- isolate this test from load_dotenv_if_present entirely.
    # build_client() now lives in partner.config (shared with
    # answering.py), so that's the module whose load_dotenv_if_present
    # actually runs -- patching reviewer's copy of the *name* wouldn't
    # touch the call inside config.build_client()'s own function body.
    monkeypatch.setattr(config, "load_dotenv_if_present", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_client()


def test_build_client_succeeds_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")

    client = build_client()

    assert isinstance(client, anthropic.Anthropic)
