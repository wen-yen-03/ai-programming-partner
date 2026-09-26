"""State and validated decision models for the grounded-answer agent.

AgentState is a TypedDict (LangGraph's native state type) and is NOT
validated at runtime -- type hints are not enforced. The rule in this
package: every model output passes one of the Pydantic models below before
it is written into state, so state only ever holds already-validated data.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

Route = Literal["search_docs", "read_session_record", "out_of_scope"]
Outcome = Literal[
    "pending", "approved", "rejected", "not_found", "out_of_scope", "error"
]
TERMINAL_OUTCOMES: frozenset[str] = frozenset(
    {"approved", "rejected", "not_found", "out_of_scope", "error"}
)
MAX_REWRITES = 1


class AgentState(TypedDict):
    question: str
    route: Route | None
    route_reason: str | None
    session_name: str | None
    query: str
    rewrites: int
    chunks: list[dict]
    session_record: dict | None
    draft: dict | None
    outcome: Outcome
    error: str | None
    reviewer_note: str | None


def initial_state(question: str) -> AgentState:
    """Build the starting state for one run. Rejects an empty question."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    return AgentState(
        question=cleaned,
        route=None,
        route_reason=None,
        session_name=None,
        query=cleaned,
        rewrites=0,
        chunks=[],
        session_record=None,
        draft=None,
        outcome="pending",
        error=None,
        reviewer_note=None,
    )


class RouteDecision(BaseModel):
    """The router's choice. session_name is present only for the session route."""

    route: Route
    reason: str = Field(min_length=1, description="One sentence explaining the choice.")
    session_name: str | None = Field(
        default=None,
        description=(
            "Name of the stored session, only when route is "
            "read_session_record; otherwise null."
        ),
    )

    @field_validator("session_name", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _session_name_matches_route(self) -> "RouteDecision":
        if self.route == "read_session_record" and self.session_name is None:
            raise ValueError("session_name is required when route is read_session_record")
        if self.route != "read_session_record" and self.session_name is not None:
            raise ValueError("session_name must be null unless route is read_session_record")
        return self


class ApprovalDecision(BaseModel):
    """What a human sends back when resuming the approval pause."""

    decision: Literal["approve", "reject"]
    note: str | None = None

    @model_validator(mode="after")
    def _reject_needs_note(self) -> "ApprovalDecision":
        if self.decision == "reject" and not (self.note and self.note.strip()):
            raise ValueError("note is required when decision is reject")
        return self
