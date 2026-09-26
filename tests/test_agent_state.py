"""Validation rules for the agent's state and decision models."""

import pytest
from pydantic import ValidationError

from partner.agent.state import ApprovalDecision, RouteDecision, initial_state


def test_initial_state_strips_question_and_sets_defaults() -> None:
    state = initial_state("  How does chunking work?  ")

    assert state["question"] == "How does chunking work?"
    assert state["query"] == "How does chunking work?"
    assert state["rewrites"] == 0
    assert state["chunks"] == []
    assert state["outcome"] == "pending"
    assert state["error"] is None


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_initial_state_rejects_empty_question(question: str) -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        initial_state(question)


def test_route_decision_requires_session_name_for_session_route() -> None:
    with pytest.raises(ValidationError, match="session_name is required"):
        RouteDecision(route="read_session_record", reason="asks about a session")


def test_route_decision_treats_blank_session_name_as_missing() -> None:
    with pytest.raises(ValidationError, match="session_name is required"):
        RouteDecision(route="read_session_record", reason="r", session_name="  ")


def test_route_decision_rejects_session_name_on_other_routes() -> None:
    with pytest.raises(ValidationError, match="session_name must be null"):
        RouteDecision(route="search_docs", reason="docs question", session_name="demo")


def test_route_decision_normalizes_empty_string_session_name_to_none() -> None:
    # gemma4 tends to emit "" instead of null for unused fields; that is not an error.
    decision = RouteDecision(route="search_docs", reason="docs question", session_name="")
    assert decision.session_name is None


def test_route_decision_rejects_unknown_route() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(route="delete_everything", reason="injected")


def test_approval_reject_requires_note() -> None:
    with pytest.raises(ValidationError, match="note is required"):
        ApprovalDecision(decision="reject")


def test_approval_approve_needs_no_note() -> None:
    assert ApprovalDecision(decision="approve").note is None
