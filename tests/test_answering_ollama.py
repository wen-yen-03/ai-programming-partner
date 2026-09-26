# Tests for src/partner/answering_ollama.py's Ollama-backed answer
# generation. Mocks `requests.post` (no real model call -- gemma4:26b takes
# ~100s to cold-load, far too slow for the routine test suite) so these run
# fast and require no local Ollama instance. The real comparison run
# against Anthropic is documented in progress/DAILY_LOG.md instead.
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
import requests

import partner.chunking as chunking
import partner.vector_store as vector_store
from partner import DOCS_DIR, answering_ollama
from partner.answering_ollama import (
    AnswerMalformedError,
    AnswerTimedOutError,
    generate_answer,
)

folder_path = DOCS_DIR / "pytests" / "chunks"

SAMPLE_CHUNKS = [
    {
        "file_name": "BUILD_PLAN.md",
        "chunk_index": 2,
        "content": "Run `uv run pytest` before every commit.",
        "offset": 20,
        "score": 0.71,
    }
]

VALID_ANSWER_CONTENT = json.dumps(
    {
        "can_answer": True,
        "answer": "Run `uv run pytest` before every commit.",
        "citations": [{"file_name": "BUILD_PLAN.md", "chunk_index": 2}],
        "rationale": "chunk_index=2 directly states the testing command.",
    }
)

# The real failure mode observed on the first live run against gemma4:26b:
# valid JSON, can_answer=True, but citations silently omitted -- passes the
# raw JSON schema (citations has a Pydantic default), fails our
# model_validator (citations required when can_answer is True).
MISSING_CITATIONS_CONTENT = json.dumps(
    {
        "can_answer": True,
        "answer": "Run `uv run pytest` before every commit.",
        "rationale": "chunk_index=2 directly states the testing command.",
    }
)


@dataclass
class FakeResponse:
    _json: dict
    status_code: int = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json


_real_post = requests.post


@dataclass
class FakePost:
    """Intercepts only calls to the chat endpoint.

    retrieve() also calls requests.post (to Ollama's /api/embeddings, a
    different URL) as part of answer_question() -- a blanket patch of
    requests.post would silently break that real, still-needed call too.
    Anything not aimed at OLLAMA_CHAT_URL is passed through to the real
    requests.post untouched.
    """

    response: FakeResponse | None = None
    exception: Exception | None = None
    last_kwargs: dict | None = field(default=None, init=False)

    def __call__(self, url: str, **kwargs: object) -> FakeResponse:
        if url != answering_ollama.OLLAMA_CHAT_URL:
            return _real_post(url, **kwargs)
        self.last_kwargs = kwargs
        if self.exception is not None:
            raise self.exception
        assert self.response is not None
        return self.response


def make_ollama_response(content: str) -> FakeResponse:
    return FakeResponse(
        _json={
            "message": {"role": "assistant", "content": content},
            "eval_duration": 1_000_000,
        }
    )


def test_valid_response_returns_an_answer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_post = FakePost(response=make_ollama_response(VALID_ANSWER_CONTENT))
    monkeypatch.setattr(requests, "post", fake_post)

    result = generate_answer("How do I run the tests?", SAMPLE_CHUNKS)

    assert result.answer.can_answer is True
    assert result.eval_duration_ns == 1_000_000
    # format must be the real JSON schema, forcing constrained decoding --
    # not left as free-text generation
    assert "properties" in fake_post.last_kwargs["json"]["format"]


def test_missing_citations_raises_malformed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reproduces the real failure observed against gemma4:26b: valid JSON
    # shape, but violates the can_answer/citations consistency rule.
    fake_post = FakePost(response=make_ollama_response(MISSING_CITATIONS_CONTENT))
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(AnswerMalformedError):
        generate_answer("How do I run the tests?", SAMPLE_CHUNKS)


def test_non_json_content_raises_malformed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_post = FakePost(response=make_ollama_response("not valid json at all"))
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(AnswerMalformedError):
        generate_answer("How do I run the tests?", SAMPLE_CHUNKS)


def test_timeout_raises_timed_out_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_post = FakePost(exception=requests.exceptions.Timeout())
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(AnswerTimedOutError):
        generate_answer("How do I run the tests?", SAMPLE_CHUNKS)


def test_empty_chunks_is_rejected_before_any_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_post = FakePost(response=make_ollama_response(VALID_ANSWER_CONTENT))
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(ValueError, match="non-empty"):
        generate_answer("How do I run the tests?", [])

    assert fake_post.last_kwargs is None


def test_answer_question_skips_the_http_call_when_nothing_is_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real Postgres + real Ollama embeddings for retrieval (free/local,
    # same pattern as test_answer_question.py) -- only the chat/generation
    # call is faked here.
    chunks = chunking.chunk_file(folder_path / "TASKS.md", chunk_size=10)
    vector_store.build_index(chunks)
    fake_post = FakePost(response=make_ollama_response(VALID_ANSWER_CONTENT))
    monkeypatch.setattr(requests, "post", fake_post)

    result = answering_ollama.answer_question(
        "wheres the best pizza in town?", k=3, min_score=0.45
    )

    assert result.answer.can_answer is False
    assert fake_post.last_kwargs is None
