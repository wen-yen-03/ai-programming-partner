# Tests for src/partner/review.py's structured review-draft schema.
# Each test traces back to a specific failure mode worked through in
# docs/eval/session_review_cases.json (see review.py's module docstring).
from pathlib import Path

import pytest
from pydantic import ValidationError

from partner.models import Session
from partner.review import (
    AcceptanceCriteriaAssessment,
    Confidence,
    ConstraintComplianceAssessment,
    ReviewPoint,
    SessionField,
    SessionReviewDraft,
    assert_reviewable,
)


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


def make_review_point(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "summary": "14 of 14 tests passed, matching the acceptance criteria.",
        "cited_fields": [SessionField.EVIDENCE, SessionField.ACCEPTANCE_CRITERIA],
    }
    data.update(overrides)
    return data


def make_draft(**overrides: object) -> SessionReviewDraft:
    data: dict[str, object] = {
        "acceptance_criteria_assessment": AcceptanceCriteriaAssessment(
            met=True,
            rationale="Evidence shows 14/14 tests passing.",
            cited_fields=[SessionField.EVIDENCE],
        ),
        "constraint_compliance_assessment": ConstraintComplianceAssessment(
            violated=False,
            rationale="No third-party library is mentioned in evidence.",
            cited_fields=[SessionField.CONSTRAINTS, SessionField.EVIDENCE],
        ),
        "strengths": [ReviewPoint(**make_review_point())],
        "confidence": Confidence.HIGH,
        "confidence_rationale": "Evidence directly supports the criteria.",
    }
    data.update(overrides)
    return SessionReviewDraft(**data)


def test_review_point_requires_a_citation() -> None:
    with pytest.raises(ValidationError):
        ReviewPoint(**make_review_point(cited_fields=[]))


def test_review_point_requires_a_summary() -> None:
    with pytest.raises(ValidationError):
        ReviewPoint(**make_review_point(summary=""))


def test_review_point_accepts_a_grounded_claim() -> None:
    point = ReviewPoint(**make_review_point())

    assert point.cited_fields == [
        SessionField.EVIDENCE,
        SessionField.ACCEPTANCE_CRITERIA,
    ]


def test_acceptance_criteria_assessment_allows_undetermined() -> None:
    # Scenario D's missed clause: "met" must be able to say "not
    # confirmed or denied" instead of being forced to guess True/False.
    assessment = AcceptanceCriteriaAssessment(
        met=None,
        rationale="Evidence never confirms records remain queryable.",
        cited_fields=[SessionField.ACCEPTANCE_CRITERIA],
    )

    assert assessment.met is None


def test_constraint_compliance_assessment_requires_citation() -> None:
    with pytest.raises(ValidationError):
        ConstraintComplianceAssessment(
            violated=True,
            rationale="Used a third-party library.",
            cited_fields=[],
        )


def test_confidence_rejects_values_outside_the_enum() -> None:
    with pytest.raises(ValidationError):
        make_draft(confidence="very high")


def test_session_review_draft_builds_with_required_fields() -> None:
    draft = make_draft()

    assert draft.confidence is Confidence.HIGH
    assert draft.risks == []
    assert len(draft.strengths) == 1


def test_assert_reviewable_passes_for_completed_session() -> None:
    session = make_completed_session()

    assert_reviewable(session)  # must not raise


def test_assert_reviewable_rejects_active_session() -> None:
    # Scenario E: an active session must never reach review generation,
    # regardless of how accurate a draft about it might otherwise be.
    session = make_completed_session()
    session.status = session.status.__class__.ACTIVE

    with pytest.raises(ValueError, match="not 'completed'"):
        assert_reviewable(session)
