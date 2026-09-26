"""Command-line interface for the programming-partner session workflow.

Checkpoint 3 scaffold: implement the handlers using Session and StorageJson.
The CLI should remain a thin boundary; validation and completion rules belong
to the domain model, and JSON persistence belongs to storage.py.
"""

import argparse
from pathlib import Path
from partner.models import Session, IncompleteSessionError
from partner.reviewer import ReviewGenerationError, build_client, generate_review_draft
from partner.storage import StorageJson


DEFAULT_SESSION_FILE = Path("sessions/session.json")


def _pending_review_path(session_file: Path) -> Path:
    return session_file.with_suffix(".review.pending.json")


def _approved_review_path(session_file: Path) -> Path:
    return session_file.with_suffix(".review.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for create, show, and complete commands."""
    parser = argparse.ArgumentParser(prog="partner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create an active session")
    create.add_argument("--file", type=Path, default=DEFAULT_SESSION_FILE)
    create.add_argument("--goal", required=True)
    create.add_argument("--project-path", type=Path, required=True)
    create.add_argument("--constraints", required=True)
    create.add_argument("--acceptance-criteria", required=True)
    create.add_argument("--next-action", required=True)

    show = subparsers.add_parser("show", help="Display the saved session")
    show.add_argument("--file", type=Path, default=DEFAULT_SESSION_FILE)

    complete = subparsers.add_parser("complete", help="Attempt to complete a session")
    complete.add_argument("--file", type=Path, default=DEFAULT_SESSION_FILE)
    complete.add_argument("--evidence", required=True)
    complete.add_argument("--explanation", required=True)

    review = subparsers.add_parser(
        "review", help="Generate an AI review draft for a completed session"
    )
    review.add_argument("--file", type=Path, default=DEFAULT_SESSION_FILE)

    approve_review = subparsers.add_parser(
        "approve-review", help="Approve a pending review draft"
    )
    approve_review.add_argument("--file", type=Path, default=DEFAULT_SESSION_FILE)

    return parser

def handle_create(args: argparse.Namespace) -> int:
    session = Session(
        goal=args.goal,
        project_path=args.project_path,
        constraints=args.constraints,
        acceptance_criteria=args.acceptance_criteria,
        next_action=args.next_action,
    )

    storage = StorageJson(file_path=args.file)
    storage.save_session(session)
    print(f"Goal: {session.goal}")
    print(f"Project Path: {session.project_path}")
    print(f"Constraints: {session.constraints}")
    print(f"Acceptance Criteria: {session.acceptance_criteria}")
    print(f"Status: {session.status.value}")
    print(f"Evidence: {session.evidence}")
    print(f"Explanation Checkpoint: {session.explanation_checkpoint}")
    print(f"Next Action: {session.next_action}")
    return 0

def handle_show(args: argparse.Namespace) -> int:
    storage = StorageJson(file_path=args.file)
    session = storage.load_session()

    print(f"Goal: {session.goal}")
    print(f"Project Path: {session.project_path}")
    print(f"Constraints: {session.constraints}")
    print(f"Acceptance Criteria: {session.acceptance_criteria}")
    print(f"Status: {session.status.value}")
    print(f"Evidence: {session.evidence}")
    print(f"Explanation Checkpoint: {session.explanation_checkpoint}")
    print(f"Next Action: {session.next_action}")
    return 0

def handle_complete(args: argparse.Namespace) -> int:
    storage = StorageJson(file_path=args.file)
    session = storage.load_session()

    session.evidence = args.evidence
    session.explanation_checkpoint = args.explanation

    try:
        session.complete()
    except IncompleteSessionError as e:
        print(f"Error: {e}")
        return 1
    storage.save_session(session)
    print(f"Goal: {session.goal}")
    print(f"Project Path: {session.project_path}")
    print(f"Constraints: {session.constraints}")
    print(f"Acceptance Criteria: {session.acceptance_criteria}")
    print(f"Status: {session.status.value}")
    print(f"Evidence: {session.evidence}")
    print(f"Explanation Checkpoint: {session.explanation_checkpoint}")
    print(f"Next Action: {session.next_action}")

    return 0

def handle_review(args: argparse.Namespace) -> int:
    storage = StorageJson(file_path=args.file)
    session = storage.load_session()

    try:
        client = build_client()
        result = generate_review_draft(session, client)
    except (ReviewGenerationError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return 1

    draft = result.draft

    def render(points):
        if not points:
            print("  (none)")
            return
        for point in points:
            fields = ", ".join(f.value for f in point.cited_fields)
            print(f"  - {point.summary} [{fields}]")

    aca = draft.acceptance_criteria_assessment
    cca = draft.constraint_compliance_assessment
    print(f"Acceptance criteria met: {aca.met}")
    print(f"  {aca.rationale}")
    print(f"Constraint violated: {cca.violated}")
    print(f"  {cca.rationale}")
    print("Strengths:")
    render(draft.strengths)
    print("Risks:")
    render(draft.risks)
    print("Follow-ups:")
    render(draft.follow_ups)
    print(f"Confidence: {draft.confidence} — {draft.confidence_rationale}")
    print(f"Tokens: in={result.input_tokens} out={result.output_tokens}")

    pending_path = _pending_review_path(args.file)
    pending_path.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nDraft written to {pending_path} (not yet approved).")
    print(f"Run `partner approve-review --file {args.file}` to approve it.")
    return 0


def handle_approve_review(args: argparse.Namespace) -> int:
    pending_path = _pending_review_path(args.file)
    approved_path = _approved_review_path(args.file)

    if not pending_path.exists():
        print(f"Error: no pending review found at {pending_path}")
        return 1

    approved_path.write_text(pending_path.read_text(encoding="utf-8"), encoding="utf-8")
    pending_path.unlink()
    print(f"Approved. Review saved to {approved_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the selected command."""
    args = build_parser().parse_args(argv)
    handlers = {
        "create": handle_create,
        "show": handle_show,
        "complete": handle_complete,
        "review": handle_review,
        "approve-review": handle_approve_review,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
