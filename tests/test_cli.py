"""Behavior tests for the command-line session workflow."""

from dataclasses import dataclass

import pytest

import partner.cli as cli
from partner.cli import main
from partner.models import SessionStatus
from partner.storage import StorageJson

# --- Fakes for the `review`/`approve-review` commands -----------------------
# Duck-type anthropic.Anthropic's .messages.create() so no real API key or
# network access is needed to test the CLI wiring. Mirrors the fakes in
# tests/test_reviewer.py, which already cover the four SDK failure modes in
# depth -- these tests only need to prove the CLI's file behavior (pending
# vs. approved, never auto-saved).


@dataclass
class FakeToolUseBlock:
    input: dict
    type: str = "tool_use"
    name: str = "submit_session_review"


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeMessage:
    content: list
    usage: FakeUsage
    stop_reason: str = "tool_use"


@dataclass
class FakeMessagesResource:
    response: FakeMessage

    def create(self, **kwargs: object) -> FakeMessage:
        return self.response


@dataclass
class FakeClient:
    messages: FakeMessagesResource


VALID_DRAFT_INPUT = {
    "acceptance_criteria_assessment": {
        "met": True,
        "rationale": "Evidence shows POST and GET both return success.",
        "cited_fields": ["evidence"],
    },
    "constraint_compliance_assessment": {
        "violated": False,
        "rationale": "No third-party library is mentioned in evidence.",
        "cited_fields": ["constraints", "evidence"],
    },
    "strengths": [
        {
            "summary": "POST and GET both confirmed working.",
            "cited_fields": ["evidence", "acceptance_criteria"],
        }
    ],
    "risks": [],
    "follow_ups": [],
    "confidence": "high",
    "confidence_rationale": "Evidence directly supports the criteria.",
}


def make_fake_client(*, malformed: bool = False) -> FakeClient:
    draft_input = (
        {**VALID_DRAFT_INPUT, "strengths": [{"summary": "x", "cited_fields": []}]}
        if malformed
        else VALID_DRAFT_INPUT
    )
    return FakeClient(
        messages=FakeMessagesResource(
            response=FakeMessage(
                content=[FakeToolUseBlock(input=draft_input)],
                usage=FakeUsage(input_tokens=100, output_tokens=50),
            )
        )
    )


def complete_args(session_file):
    return [
        "complete",
        "--file", str(session_file),
        "--evidence", "POST and GET is working",
        "--explanation", "Testing POST and GET is working as expected.",
    ]


def create_args(session_file):
    return [
        "create",
        "--file", str(session_file),
        "--goal", "Build a CRUD application",
        "--project-path", "crud",
        "--constraints", "Full stack",
        "--acceptance-criteria", "Able to POST and GET values",
        "--next-action", "Plan the stack",
    ]


def test_create_writes_active_session(tmp_path):
    session_file = tmp_path / "practice.json"
    result = main(create_args(session_file))
    saved = StorageJson(session_file).load_session()

    assert result == 0
    assert session_file.exists()
    assert saved.goal == "Build a CRUD application"
    assert saved.status == SessionStatus.ACTIVE


def test_create_requires_all_session_fields(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "create",
            "--file", str(tmp_path / "practice.json"),
            "--project-path", "crud",
            "--constraints", "Full stack",
            "--acceptance-criteria", "Able to POST and GET values",
            "--next-action", "Plan the stack",
        ])


def test_show_reads_and_displays_saved_session(tmp_path, capsys):
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    result = main(["show", "--file", str(session_file)])
    output = capsys.readouterr().out

    assert result == 0
    assert "Goal: Build a CRUD application" in output
    assert "Status: active" in output
    assert "Next Action: Plan the stack" in output


def test_complete_rejects_missing_evidence(tmp_path, capsys):
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    result = main([
        "complete", "--file", str(session_file),
        "--evidence", "",
        "--explanation", "I understand the completion rule.",
    ])
    output = capsys.readouterr().out
    saved = StorageJson(session_file).load_session()

    assert result == 1
    assert "Evidence is required." in output
    assert saved.status == SessionStatus.ACTIVE


def test_complete_rejects_missing_explanation(tmp_path, capsys):
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    result = main([
        "complete", "--file", str(session_file),
        "--evidence", "POST and GET is working",
        "--explanation", "",
    ])
    output = capsys.readouterr().out
    saved = StorageJson(session_file).load_session()

    assert result == 1
    assert "An explanation checkpoint is required." in output
    assert saved.status == SessionStatus.ACTIVE


def test_complete_saves_completed_session(tmp_path):
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    result = main([
        "complete", "--file", str(session_file),
        "--evidence", "POST and GET is working",
        "--explanation", "Testing POST and GET is working as expected.",
    ])
    saved = StorageJson(session_file).load_session()

    assert result == 0
    assert saved.status == SessionStatus.COMPLETED
    assert saved.evidence == "POST and GET is working"


def test_complete_reports_completion_to_user(tmp_path, capsys):
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    main([
        "complete", "--file", str(session_file),
        "--evidence", "POST and GET is working",
        "--explanation", "Testing POST and GET is working as expected.",
    ])
    output = capsys.readouterr().out

    assert "Status: completed" in output
    assert "POST and GET is working" in output


# --- review / approve-review ------------------------------------------------


def test_review_rejects_non_completed_session(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_client", lambda: make_fake_client())
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))  # active, not completed

    result = main(["review", "--file", str(session_file)])

    assert result == 1
    assert not cli._pending_review_path(session_file).exists()


def test_review_writes_pending_file_but_not_approved(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_client", lambda: make_fake_client())
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    main(complete_args(session_file))

    result = main(["review", "--file", str(session_file)])

    assert result == 0
    # never auto-saved: pending exists, approved does not
    assert cli._pending_review_path(session_file).exists()
    assert not cli._approved_review_path(session_file).exists()


def test_review_prints_the_draft(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_client", lambda: make_fake_client())
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    main(complete_args(session_file))

    main(["review", "--file", str(session_file)])
    output = capsys.readouterr().out

    assert "Acceptance criteria met: True" in output
    assert "POST and GET both confirmed working." in output
    assert "Confidence: high" in output
    assert "not yet approved" in output


def test_review_reports_malformed_output_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_client", lambda: make_fake_client(malformed=True))
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    main(complete_args(session_file))

    result = main(["review", "--file", str(session_file)])

    assert result == 1
    assert not cli._pending_review_path(session_file).exists()
    assert not cli._approved_review_path(session_file).exists()


def test_approve_review_without_pending_file_errors(tmp_path, capsys):
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))

    result = main(["approve-review", "--file", str(session_file)])
    output = capsys.readouterr().out

    assert result == 1
    assert "no pending review" in output
    assert not cli._approved_review_path(session_file).exists()


def test_approve_review_promotes_pending_to_approved(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_client", lambda: make_fake_client())
    session_file = tmp_path / "practice.json"
    main(create_args(session_file))
    main(complete_args(session_file))
    main(["review", "--file", str(session_file)])

    result = main(["approve-review", "--file", str(session_file)])

    assert result == 0
    assert not cli._pending_review_path(session_file).exists()
    assert cli._approved_review_path(session_file).exists()
    approved_content = cli._approved_review_path(session_file).read_text()
    assert '"confidence":"high"' in approved_content.replace(" ", "").replace(
        "\n", ""
    )
