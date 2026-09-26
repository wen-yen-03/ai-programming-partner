"""Raw Anthropic SDK call that turns a question + retrieved chunks into a
GroundedAnswer (src/partner/answer.py).

Same forced-tool-use pattern as reviewer.py's generate_review_draft(), and
the same four failure modes: valid, malformed, refused, timed-out.

Layered on top of retrieval.retrieve(): answer_question() is the
orchestrating entry point most callers want. It skips the model call
entirely -- no cost spent -- when retrieve() finds nothing above
min_score, since that's already a confident "nothing relevant" signal from
the retrieval layer and there's nothing to ground an answer in anyway.
That's a *different* not-found path than the one this module's schema
guards against: retrieve() returning [] means nothing was even topically
similar enough to send to the model; GroundedAnswer.can_answer == False
means chunks *were* retrieved but the model judged they don't actually
answer the specific question asked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import anthropic
from anthropic.types import ToolUseBlock
from pydantic import ValidationError

from partner.answer import GroundedAnswer
from partner.retrieval import retrieve

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheapest current Claude model
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
TOOL_NAME = "submit_grounded_answer"

NO_CHUNKS_RATIONALE = (
    "No retrieved chunk scored high enough for this query to be worth "
    "sending to the model."
)

_PROMPT_TEMPLATE = """\
Answer the question using ONLY the retrieved chunks below. Produce, in a \
single tool call:

1. can_answer -- true only if the chunks actually contain enough \
information to answer the specific question asked. A chunk that is \
topically related but doesn't actually address the question means false.
2. answer -- the answer, grounded only in the chunks below. Must be null \
if can_answer is false.
3. citations -- which chunk(s) (file_name + chunk_index) the answer is \
grounded in. Required if can_answer is true.
4. rationale -- one sentence explaining the can_answer judgment.

Never use outside knowledge. Never state anything the chunks don't \
directly support. If the chunks are only loosely related to the question, \
that is not enough to answer -- set can_answer to false rather than \
stretching a loosely related chunk into an answer.

Question: {question}

Retrieved chunks:
{chunks_block}
"""


class AnswerGenerationError(Exception):
    """Base class for answer-generation failures."""


class AnswerMalformedError(AnswerGenerationError):
    """The model's tool call did not validate against GroundedAnswer."""

    def __init__(
        self,
        message: str,
        *,
        raw_input: object,
        stop_reason: str | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        super().__init__(message)
        self.raw_input = raw_input
        self.stop_reason = stop_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class AnswerRefusedError(AnswerGenerationError):
    """The model declined to produce an answer, or made no tool call."""

    def __init__(
        self,
        message: str,
        *,
        stop_reason: str | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class AnswerTimedOutError(AnswerGenerationError):
    """The API call did not complete within the configured timeout."""


@dataclass
class AnswerResult:
    answer: GroundedAnswer
    input_tokens: int
    output_tokens: int


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
    client: anthropic.Anthropic,
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_malformed_retries: int = 1,
) -> AnswerResult:
    """Generate a grounded answer from already-retrieved chunks.

    Raises AnswerGenerationError (or a subclass) on any failure. Does not
    call retrieve() itself -- see answer_question() for the orchestrating
    function that also skips this call entirely when nothing was
    retrieved. Malformed output is retried up to max_malformed_retries
    times, same rationale as generate_review_draft(): an observed
    intermittent model glitch, not a consistent prompt/schema problem.
    """
    if not chunks:
        raise ValueError("chunks must be non-empty -- call retrieve() first.")

    attempt = 0
    while True:
        try:
            return _generate_once(
                question,
                chunks,
                client,
                model=model,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        except AnswerMalformedError:
            if attempt >= max_malformed_retries:
                raise
            attempt += 1
            logger.info(
                "retrying after malformed answer output (attempt %s of %s)",
                attempt,
                max_malformed_retries,
            )


def _generate_once(
    question: str,
    chunks: list[dict],
    client: anthropic.Anthropic,
    *,
    model: str,
    max_output_tokens: int,
    timeout: float,
) -> AnswerResult:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            timeout=timeout,
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": "Submit the grounded answer.",
                    "input_schema": GroundedAnswer.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": _build_prompt(question, chunks)}],
        )
    except anthropic.APITimeoutError as exc:
        raise AnswerTimedOutError(
            f"Answer generation timed out after {timeout}s"
        ) from exc

    logger.info(
        "grounded answer usage: model=%s input_tokens=%s output_tokens=%s",
        model,
        message.usage.input_tokens,
        message.usage.output_tokens,
    )

    usage_kwargs = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }

    if message.stop_reason == "refusal":
        raise AnswerRefusedError(
            "Model declined to produce an answer.",
            stop_reason=message.stop_reason,
            **usage_kwargs,
        )

    # getattr, not isinstance(block, ToolUseBlock): tests use plain
    # dataclasses that duck-type the real SDK's content blocks (no network
    # needed to test this module), so a real isinstance check against the
    # SDK's own class would reject legitimate fakes at runtime.
    found_block = next(
        (
            block
            for block in message.content
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == TOOL_NAME
        ),
        None,
    )
    # cast, not a type: ignore -- this documents *why* the checker can't
    # see it (duck typing, not a real type error) and still catches a
    # genuine mismatch if ToolUseBlock's fields ever change shape.
    tool_use_block = cast(ToolUseBlock, found_block) if found_block else None
    if tool_use_block is None:
        raise AnswerRefusedError(
            "Model did not produce an answer (no tool call present).",
            stop_reason=message.stop_reason,
            **usage_kwargs,
        )

    try:
        answer = GroundedAnswer.model_validate(tool_use_block.input)
    except ValidationError as exc:
        raise AnswerMalformedError(
            f"Model's answer failed schema validation: {exc}",
            raw_input=tool_use_block.input,
            stop_reason=message.stop_reason,
            **usage_kwargs,
        ) from exc

    return AnswerResult(
        answer=answer,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


def answer_question(
    question: str,
    client: anthropic.Anthropic,
    *,
    k: int = 3,
    min_score: float = 0.45,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_malformed_retries: int = 1,
) -> AnswerResult:
    """Retrieve chunks for `question`, then generate a grounded answer.

    Skips the model call entirely (0 tokens, 0 cost) when retrieve() finds
    nothing above min_score -- see this module's docstring for why that's
    a distinct not-found path from GroundedAnswer.can_answer == False.
    """
    chunks = retrieve(question, k=k, min_score=min_score)
    if not chunks:
        return AnswerResult(
            answer=GroundedAnswer(can_answer=False, rationale=NO_CHUNKS_RATIONALE),
            input_tokens=0,
            output_tokens=0,
        )
    return generate_answer(
        question,
        chunks,
        client,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_malformed_retries=max_malformed_retries,
    )
