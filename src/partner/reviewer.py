"""Raw Anthropic SDK call that turns a completed Session into a
SessionReviewDraft (src/partner/review.py).

Uses forced tool-use to get schema-conformant structured output: the
Pydantic schema is passed as a tool's input_schema and tool_choice pins the
model to that tool. The tool schema is only a hint to the model though, not
a guarantee, so SessionReviewDraft.model_validate() on the tool's input is
the real enforcement point -- this is also why a well-formed tool call can
still fail validation (e.g. a strength with no cited_fields).

Failure modes handled explicitly (docs/TASKS.md "AI Checkpoint 1"):
- valid: a tool_use call is present and its input validates ->
  SessionReviewDraft
- malformed: a tool_use call is present but its input fails Pydantic
  validation
- refused: the model declines (stop_reason == "refusal"), or produces no
  tool_use call at all
- timed-out: the SDK call exceeds the configured timeout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic
from pydantic import ValidationError

from partner.config import build_client  # noqa: F401 -- re-exported for callers
from partner.models import Session
from partner.review import SessionReviewDraft, assert_reviewable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # cheapest current Claude model
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TIMEOUT_SECONDS = 30.0
TOOL_NAME = "submit_session_review"

_PROMPT_TEMPLATE = """\
Review this completed programming session by producing ALL seven of the \
following, in this order, in a single tool call:

1. acceptance_criteria_assessment -- was the acceptance criteria met? \
(true / false / null if the record doesn't confirm or deny it)
2. constraint_compliance_assessment -- were the constraints violated? \
(true / false / null if the record doesn't confirm or deny it)
3. strengths -- specific, grounded positive findings (an empty list is fine)
4. risks -- specific, grounded problems or gaps (an empty list is fine)
5. follow_ups -- concrete next steps (an empty list is fine)
6. confidence -- your overall confidence: "low", "medium", or "high"
7. confidence_rationale -- one sentence explaining that confidence level

Do not stop after step 1 or 2. Fields 6 and 7 (confidence and \
confidence_rationale) are always required, even when strengths, risks, \
and follow_ups are all empty lists.

A met acceptance criterion does not mean constraints were followed, and a \
followed constraint does not mean the acceptance criteria was met -- \
assess them independently. For every strength, risk, and follow-up, cite \
which session field(s) it is grounded in; never state anything the record \
doesn't support. Citing a field is only valid if that field actually says \
the thing you're claiming -- not just because it's topically related. \
Never attribute a quality (e.g. "thoughtful", "robust", "backward \
compatible") to the work unless the record states it directly.

Check specifically for these three failure patterns before finalizing:
- A next_action that the evidence shows has already been completed -- \
flag this as a stale/redundant risk rather than treating it as \
legitimate remaining work.
- Evidence that is a bare, unproven assertion -- e.g. "X is working" or \
"tests pass" with no command run, no specific count, and no output \
shown -- must be flagged as an unverified claim, not treated as proof \
the acceptance criteria was met. Evidence with concrete detail (a \
command, a specific pass count like "14 passed", a request/response \
trace) is real support; a bare claim on the same topic is not, even \
though it looks similar at a glance.
- An explanation_checkpoint that merely restates the goal or evidence \
instead of describing how or why something works -- flag this as weak \
evidence of understanding rather than crediting it as a strength.

Goal: {goal}
Project path: {project_path}
Constraints: {constraints}
Acceptance criteria: {acceptance_criteria}
Evidence: {evidence}
Explanation checkpoint: {explanation_checkpoint}
Next action: {next_action}
"""


class ReviewGenerationError(Exception):
    """Base class for review-generation failures."""


class ReviewMalformedError(ReviewGenerationError):
    """The model's tool call did not validate against SessionReviewDraft."""

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


class ReviewRefusedError(ReviewGenerationError):
    """The model declined to produce a review, or made no tool call."""

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


class ReviewTimedOutError(ReviewGenerationError):
    """The API call did not complete within the configured timeout."""


@dataclass
class ReviewResult:
    draft: SessionReviewDraft
    input_tokens: int
    output_tokens: int


def _build_prompt(session: Session) -> str:
    return _PROMPT_TEMPLATE.format(
        goal=session.goal,
        project_path=session.project_path,
        constraints=session.constraints,
        acceptance_criteria=session.acceptance_criteria,
        evidence=session.evidence,
        explanation_checkpoint=session.explanation_checkpoint,
        next_action=session.next_action,
    )


def generate_review_draft(
    session: Session,
    client: anthropic.Anthropic,
    *,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_malformed_retries: int = 1,
) -> ReviewResult:
    """Generate a structured review draft for a completed session.

    Raises ReviewGenerationError (or a subclass) on any failure; never
    returns a partially-valid or auto-saved draft. The caller is
    responsible for presenting the draft for human approval before
    persisting it.

    Malformed output (a tool call that fails schema validation) is retried
    up to max_malformed_retries times -- observed in real eval runs to be
    an intermittent, non-deterministic model glitch (e.g. corrupted JSON
    mid-generation), not a consistent prompt/schema problem, so a retry
    with the same inputs has a real chance of succeeding. Refused and
    timed-out calls are not retried here -- those are either a deliberate
    model decision or already a slow-path failure, not a one-off glitch.
    """
    assert_reviewable(session)

    attempt = 0
    while True:
        try:
            return _generate_once(
                session,
                client,
                model=model,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        except ReviewMalformedError:
            if attempt >= max_malformed_retries:
                raise
            attempt += 1
            logger.info(
                "retrying after malformed output (attempt %s of %s)",
                attempt,
                max_malformed_retries,
            )


def _generate_once(
    session: Session,
    client: anthropic.Anthropic,
    *,
    model: str,
    max_output_tokens: int,
    timeout: float,
) -> ReviewResult:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            timeout=timeout,
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": "Submit the structured session review draft.",
                    "input_schema": SessionReviewDraft.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": _build_prompt(session)}],
        )
    except anthropic.APITimeoutError as exc:
        raise ReviewTimedOutError(
            f"Review generation timed out after {timeout}s"
        ) from exc

    logger.info(
        "session review usage: model=%s input_tokens=%s output_tokens=%s",
        model,
        message.usage.input_tokens,
        message.usage.output_tokens,
    )

    usage_kwargs = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }

    if message.stop_reason == "refusal":
        raise ReviewRefusedError(
            "Model declined to produce a review draft.",
            stop_reason=message.stop_reason,
            **usage_kwargs,
        )

    tool_use_block = next(
        (
            block
            for block in message.content
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == TOOL_NAME
        ),
        None,
    )
    if tool_use_block is None:
        raise ReviewRefusedError(
            "Model did not produce a review draft (no tool call present).",
            stop_reason=message.stop_reason,
            **usage_kwargs,
        )

    try:
        draft = SessionReviewDraft.model_validate(tool_use_block.input)
    except ValidationError as exc:
        raise ReviewMalformedError(
            f"Model's review draft failed schema validation: {exc}",
            raw_input=tool_use_block.input,
            stop_reason=message.stop_reason,
            **usage_kwargs,
        ) from exc

    return ReviewResult(
        draft=draft,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
