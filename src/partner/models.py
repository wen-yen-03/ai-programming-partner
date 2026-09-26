# Domain model for a programming-partner Session: fields, status enum, and
# the state transitions described in docs/BUILD_PLAN.md milestone 1 and
# docs/sessions/2026-09-06-vertical-slice-1.md. Your turn to write this.
from enum import StrEnum
from dataclasses import dataclass
from pathlib import Path


class SessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"


class IncompleteSessionError(Exception):
    """Raised when a session is attempted to be completed without required fields."""

    pass


@dataclass
class Session:
    goal: str 
    project_path: Path
    constraints: str
    acceptance_criteria: str 
    status: SessionStatus = SessionStatus.ACTIVE
    evidence: str = ""
    explanation_checkpoint: str = ""
    next_action: str = ""

    def complete(self) -> None:
        if not self.evidence.strip():
            raise IncompleteSessionError("Evidence is required.")

        if not self.explanation_checkpoint.strip():
            raise IncompleteSessionError(
                "An explanation checkpoint is required."
            )

        self.status = SessionStatus.COMPLETED