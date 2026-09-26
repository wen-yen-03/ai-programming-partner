# Tests for src/partner/models.py.
# Must cover at minimum (BUILD_PLAN.md acceptance criteria):
#  - a new session can be created with goal + acceptance criteria + status
#  - completion is blocked without a test/evidence note and explanation checkpoint
# Your turn to write these before or alongside the implementation.
from pathlib import Path

import pytest

from partner.models import (
    IncompleteSessionError,
    Session,
    SessionStatus,
)


def make_session() -> Session:
    return Session(
        goal="Build a calculator",
        project_path=Path("calculator"),
        constraints="Use Python standard library only",
        acceptance_criteria="The calculator passes its tests",
        next_action="Write the first test",
    )


def test_new_session_starts_active() -> None:
    session = make_session()

    assert session.status is SessionStatus.ACTIVE
    assert session.goal == "Build a calculator"


def test_completion_fails_without_explanation() -> None:
    session = make_session()
    session.explanation_checkpoint = "I can explain the design."

    with pytest.raises(IncompleteSessionError):
        session.complete()

    assert session.status is SessionStatus.ACTIVE


def test_completion_fails_without_evidence() -> None:
    session = make_session()
    session.evidence = "All tests pass."

    with pytest.raises(IncompleteSessionError):
        session.complete()

    assert session.status is SessionStatus.ACTIVE


def test_completion_fails_with_whitespace_only_evidence() -> None:
    session = make_session()
    session.evidence = "   "
    session.explanation_checkpoint = "I can explain the design."

    with pytest.raises(IncompleteSessionError):
        session.complete()

    assert session.status is SessionStatus.ACTIVE


def test_completion_succeeds_with_evidence_and_explanation() -> None:
    session = make_session()
    session.evidence = "All tests pass."
    session.explanation_checkpoint = (
        "I can explain the status transition and completion rule."
    )

    session.complete()

    assert session.status is SessionStatus.COMPLETED
    assert session.evidence == "All tests pass."
    assert session.explanation_checkpoint == "I can explain the status transition and completion rule."