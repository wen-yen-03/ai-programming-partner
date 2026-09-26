from pathlib import Path
from partner.storage import StorageJson
from partner.models import Session, SessionStatus
import json
import pytest

def make_storage(tmp_path) -> StorageJson:
    return StorageJson(file_path=tmp_path / "session.json")

def make_session() -> Session:
    return Session(
        goal="Build a calculator",
        project_path=Path("calculator"),
        constraints="Use Python standard library only",
        acceptance_criteria="The calculator passes its tests",
        status=SessionStatus.ACTIVE,
        evidence="",
        explanation_checkpoint="",
        next_action="Write the first test",
    )

def test_json_after_saving(tmp_path) -> None:
    session = make_session()
    session.status = SessionStatus.COMPLETED
    session.evidence = "All tests pass."
    session.explanation_checkpoint = "I can explain the status transition and completion rule."

    make_storage(tmp_path).save_session(session)

    with open(tmp_path / "session.json", "r") as f:
        json_check = json.load(f)

    assert json_check['status'] == SessionStatus.COMPLETED
    assert json_check['evidence'] == "All tests pass."
    assert json_check['explanation_checkpoint'] == "I can explain the status transition and completion rule."

def test_json_after_loading(tmp_path) -> None:
    file_path = tmp_path
    session = make_session()
    session.status = SessionStatus.COMPLETED
    session.evidence = "All tests pass."
    session.explanation_checkpoint = "I can explain the status transition and completion rule."

    make_storage(file_path).save_session(session)

    test_session = Session(
        goal="Build a calculator",
        project_path=Path("calculator"),
        constraints="Use Python standard library only",
        acceptance_criteria="The calculator passes its tests",
        status=SessionStatus.COMPLETED,
        evidence="All tests pass.",
        explanation_checkpoint="I can explain the status transition and completion rule.",
        next_action="Write the first test",
    )

    loaded_session = make_storage(file_path).load_session()

    assert loaded_session == test_session