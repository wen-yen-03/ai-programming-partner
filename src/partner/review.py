"""Structured output schema for AI-assisted session review.

Design informed directly by five review-draft failure modes worked through
in docs/eval/session_review_cases.json:

- A missed constraint violation while acceptance criteria looked satisfied
  (conflating two independent axes) -> acceptance criteria and constraint
  compliance are separate required fields below, not folded into one pass/
  fail judgment.
- A claim that flatly contradicted the evidence field, and a plausible-
  sounding claim grounded in nothing at all -> every ReviewPoint must cite
  which Session field(s) it is grounded in.
- A correct draft that still missed an unaddressed acceptance-criteria
  clause -> "met" is a three-state field (True / False / None), where None
  means the record doesn't confirm or deny it, rather than a boolean that
  forces a guess.
- A review generated for a session that was never actually completed ->
  assert_reviewable() gates on status before a draft is produced at all.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from partner.models import Session, SessionStatus


class SessionField(StrEnum):
    GOAL = "goal"
    PROJECT_PATH = "project_path"
    CONSTRAINTS = "constraints"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    EVIDENCE = "evidence"
    EXPLANATION_CHECKPOINT = "explanation_checkpoint"
    NEXT_ACTION = "next_action"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewPoint(BaseModel):
    """One specific, checkable claim about the session."""

    summary: str = Field(min_length=1)
    cited_fields: list[SessionField] = Field(
        min_length=1,
        description=(
            "Session fields this claim is grounded in. A claim with no "
            "citation is not a valid review point."
        ),
    )


class AcceptanceCriteriaAssessment(BaseModel):
    met: bool | None = Field(
        description=(
            "True/False only when the record confirms or contradicts it. "
            "None means the record neither confirms nor denies it."
        )
    )
    rationale: str = Field(min_length=1)
    cited_fields: list[SessionField] = Field(min_length=1)


class ConstraintComplianceAssessment(BaseModel):
    violated: bool | None = Field(
        description=(
            "True/False only when the record confirms or contradicts "
            "compliance. None means constraints don't apply or nothing in "
            "the record speaks to them."
        )
    )
    rationale: str = Field(min_length=1)
    cited_fields: list[SessionField] = Field(min_length=1)


class SessionReviewDraft(BaseModel):
    """A structured review draft for one completed Session.

    Never auto-saved: the learner approves or edits this before it is
    persisted, same as the human-approval principle the rest of the
    project already enforces for session completion.
    """

    acceptance_criteria_assessment: AcceptanceCriteriaAssessment
    constraint_compliance_assessment: ConstraintComplianceAssessment
    strengths: list[ReviewPoint] = Field(default_factory=list)
    risks: list[ReviewPoint] = Field(default_factory=list)
    follow_ups: list[ReviewPoint] = Field(default_factory=list)
    confidence: Confidence
    confidence_rationale: str = Field(min_length=1)


def assert_reviewable(session: Session) -> None:
    """Raise if session is not eligible for AI-assisted review.

    The feature reviews completed sessions only (BUILD_PLAN.md: "Given a
    completed session record..."). An active session hasn't passed
    Session.complete()'s own evidence/explanation gate yet, so generating
    a review draft for it would review work that was never submitted for
    review in the first place.
    """
    if session.status != SessionStatus.COMPLETED:
        raise ValueError(
            f"Session status is {session.status!r}, not 'completed' — "
            "only completed sessions can be reviewed."
        )
