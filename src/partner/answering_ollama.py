"""Ollama-backed alternative to answering.py's generate_answer(), producing
the same GroundedAnswer (src/partner/answer.py) from a local model instead
of the Anthropic API. A deliberate experiment (2026-09-10): is a local
chat-capable model reliable enough at structured output to be a genuine
alternative to Anthropic for this feature? See progress/DAILY_LOG.md for
the real comparison run and its result.

Uses Ollama's `format` parameter (a JSON schema, enforced via constrained
decoding at the token level) instead of Anthropic's forced tool_choice --
a different mechanism aimed at the same goal: schema-conformant output.
Like answering.py, the JSON schema is only a *shape* guarantee, not a
semantic one: GroundedAnswer.model_validate_json() is still the real
enforcement point. Confirmed on the very first real run against
gemma4:26b: a can_answer=True response came back with `citations`
completely omitted -- valid per the raw JSON schema (citations has a
Pydantic default_factory=list, so it's optional there), but rejected by
GroundedAnswer's model_validator, which requires a non-empty citations
list whenever can_answer is True. The JSON-schema constraint stops the
model from producing malformed JSON; it can't stop it from omitting an
optional-looking field our business rule actually requires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from pydantic import ValidationError

from partner.answer import GroundedAnswer
from partner.retrieval import retrieve

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemma4:26b"
DEFAULT_TIMEOUT_SECONDS = 180.0  # local models can be slow to cold-load
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

NO_CHUNKS_RATIONALE = (
    "No retrieved chunk scored high enough for this query to be worth "
    "sending to the model."
)

_PROMPT_TEMPLATE = """\
Answer the question using ONLY the retrieved chunks below. Respond with a \
JSON object matching the required schema exactly, including a non-empty \
citations list whenever can_answer is true.

can_answer -- true only if the chunks actually contain enough information \
to answer the specific question asked. A chunk that is topically related \
but doesn't actually address the question means false.
answer -- the answer, grounded only in the chunks below. Must be null if \
can_answer is false.
citations -- which chunk(s) (file_name + chunk_index) the answer is \
grounded in. Required and non-empty if can_answer is true.
rationale -- one sentence explaining the can_answer judgment.

Never use outside knowledge. Never state anything the chunks don't \
directly support. If the chunks are only loosely related to the question, \
set can_answer to false rather than stretching a loosely related chunk \
into an answer.

Question: {question}

Retrieved chunks:
{chunks_block}
"""


class AnswerGenerationError(Exception):
    """Base class for Ollama answer-generation failures."""


class AnswerMalformedError(AnswerGenerationError):
    """Ollama's response was not valid JSON, or failed schema validation."""

    def __init__(self, message: str, *, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class AnswerTimedOutError(AnswerGenerationError):
    """The Ollama call did not complete within the configured timeout."""


@dataclass
class AnswerResult:
    answer: GroundedAnswer
    eval_duration_ns: int


def _format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[chunk_index={c['chunk_index']} file_name={c['file_name']}]\n{c['content']}"
        for c in chunks
    )


def _build_prompt(question: str, chunks: list[dict]) -> str:
    return _PROMPT_TEMPLATE.format(
        question=question, chunks_block=_format_chunks(chunks)
    )


def generate_answer(
    question: str,
    chunks: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> AnswerResult:
    """Generate a grounded answer from already-retrieved chunks via Ollama.

    Raises AnswerGenerationError (or a subclass) on any failure. No
    retry-on-malformed here (unlike answering.py's Anthropic path) --
    deliberately deferred until more real runs show whether malformed
    output here is a one-off glitch or a systemic reliability gap; see
    progress/DAILY_LOG.md.
    """
    if not chunks:
        raise ValueError("chunks must be non-empty -- call retrieve() first.")

    schema = GroundedAnswer.model_json_schema()
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": _build_prompt(question, chunks)}
                ],
                "format": schema,
                "stream": False,
            },
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise AnswerTimedOutError(
            f"Ollama answer generation timed out after {timeout}s"
        ) from exc
    response.raise_for_status()
    data = response.json()

    raw_content = data["message"]["content"]
    try:
        answer = GroundedAnswer.model_validate_json(raw_content)
    except ValidationError as exc:
        raise AnswerMalformedError(
            f"Ollama's answer failed schema validation: {exc}",
            raw_content=raw_content,
        ) from exc

    eval_duration_ns = data.get("eval_duration", 0)
    logger.info(
        "ollama grounded answer: model=%s eval_duration_ns=%s",
        model,
        eval_duration_ns,
    )
    return AnswerResult(answer=answer, eval_duration_ns=eval_duration_ns)


def answer_question(
    question: str,
    *,
    k: int = 3,
    min_score: float = 0.45,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> AnswerResult:
    """Retrieve chunks for `question`, then generate a grounded answer via Ollama."""
    chunks = retrieve(question, k=k, min_score=min_score)
    if not chunks:
        return AnswerResult(
            answer=GroundedAnswer(can_answer=False, rationale=NO_CHUNKS_RATIONALE),
            eval_duration_ns=0,
        )
    return generate_answer(question, chunks, model=model, timeout=timeout)
