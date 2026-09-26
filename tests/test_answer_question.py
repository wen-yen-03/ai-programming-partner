# Tests for answer_question()'s orchestration of retrieve() + generate_answer().
# Real Postgres + real Ollama for retrieval (free/local, no mocking -- same
# pattern as test_retrieval.py), but a fake Anthropic client for the model
# call, so this costs no real API money while still proving the wiring.
from __future__ import annotations

from dataclasses import dataclass, field

import partner.chunking as chunking
import partner.vector_store as vector_store
from partner import DOCS_DIR
from partner.answering import answer_question

folder_path = DOCS_DIR / "pytests" / "chunks"


@dataclass
class FakeToolUseBlock:
    input: dict
    type: str = "tool_use"
    name: str = "submit_grounded_answer"


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
    response: FakeMessage
    call_count: int = field(default=0, init=False)

    def create(self, **kwargs: object) -> FakeMessage:
        self.call_count += 1
        return self.response


@dataclass
class FakeClient:
    messages: FakeMessagesResource


VALID_ANSWER_INPUT = {
    "can_answer": True,
    "answer": "See the tasks listed in TASKS.md.",
    "citations": [{"file_name": "TASKS.md", "chunk_index": 1}],
    "rationale": "The retrieved chunk lists completed tasks directly.",
}


def index_sample_docs() -> None:
    chunks = chunking.chunk_file(folder_path / "TASKS.md", chunk_size=10)
    vector_store.build_index(chunks)


def test_skips_the_model_call_when_nothing_is_retrieved() -> None:
    index_sample_docs()
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=VALID_ANSWER_INPUT)],
                usage=FakeUsage(input_tokens=100, output_tokens=20),
            )
        )
    )

    result = answer_question(
        "wheres the best pizza in town?", fake_client, k=3, min_score=0.45
    )

    assert result.answer.can_answer is False
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert fake_client.messages.call_count == 0


def test_calls_the_model_when_something_relevant_is_retrieved() -> None:
    index_sample_docs()
    fake_client = FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=VALID_ANSWER_INPUT)],
                usage=FakeUsage(input_tokens=100, output_tokens=20),
            )
        )
    )

    result = answer_question(
        "show me tasks I have completed", fake_client, k=3, min_score=0.45
    )

    assert fake_client.messages.call_count == 1
    assert result.answer.can_answer is True
    assert result.input_tokens == 100
