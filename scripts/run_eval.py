"""Run the labeled evaluation cases (docs/eval/session_review_cases.json)
against the real Anthropic API and produce a human-reviewable report.

This does not auto-grade pass/fail -- matching a generative draft's
content against expected_signals is inherently fuzzy, and the point of
this feature is human review of AI output, not blind trust in it. The
report lays expected vs. actual side by side for a person to judge each
case, the same way the eval cases themselves were reviewed by hand.

Usage: python scripts/run_eval.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from partner.models import Session  # noqa: E402
from partner.review import ReviewPoint  # noqa: E402
from partner.reviewer import (  # noqa: E402
    DEFAULT_MODEL,
    ReviewGenerationError,
    ReviewMalformedError,
    ReviewRefusedError,
    build_client,
    generate_review_draft,
)

CASES_PATH = REPO_ROOT / "docs" / "eval" / "session_review_cases.json"
RESULTS_DIR = REPO_ROOT / "docs" / "eval" / "run_results"


def make_session(case_session: dict) -> Session:
    return Session(
        goal=case_session["goal"],
        project_path=Path(case_session["project_path"]),
        constraints=case_session["constraints"],
        acceptance_criteria=case_session["acceptance_criteria"],
        evidence=case_session.get("evidence", ""),
        explanation_checkpoint=case_session.get("explanation_checkpoint", ""),
        next_action=case_session.get("next_action", ""),
    )


def render_points_section(points: list[ReviewPoint]) -> list[str]:
    if not points:
        return ["  (none)"]
    rendered = []
    for point in points:
        fields = ", ".join(f.value for f in point.cited_fields)
        rendered.append(f"  - {point.summary} [{fields}]")
    return rendered


def run() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    client = build_client()

    total_input_tokens = 0
    total_output_tokens = 0
    lines: list[str] = [
        f"# Eval run — {datetime.now(timezone.utc).isoformat()}",
        f"Model: {DEFAULT_MODEL}\n",
    ]

    for case in cases:
        case_id = case["id"]
        session_data = case["session"]
        expected = case["expected_signals"]
        print(f"Running {case_id}...", file=sys.stderr)

        lines.append(f"## {case_id} ({case['category']})\n")

        if session_data.get("status") == "active":
            session = make_session(session_data)  # left ACTIVE on purpose
            try:
                generate_review_draft(session, client)
                lines.append(
                    "**FAIL — guard did not reject an active session.**\n"
                )
            except ValueError as exc:
                lines.append(f"PASS — guard rejected as expected: {exc}")
                lines.append(
                    "(no draft is generated for this case; the check is "
                    "whether the guard raised, which it did — there's "
                    "nothing to compare against expected_signals here)\n"
                )
            continue

        session = make_session(session_data)
        session.complete()

        try:
            result = generate_review_draft(session, client)
        except (ReviewMalformedError, ReviewRefusedError) as exc:
            total_input_tokens += exc.input_tokens
            total_output_tokens += exc.output_tokens
            lines.append(f"**ERROR ({type(exc).__name__})**: {exc}")
            lines.append(f"stop_reason: {exc.stop_reason}")
            if isinstance(exc, ReviewMalformedError):
                lines.append(f"raw tool input: {exc.raw_input!r}")
            lines.append("")
            continue
        except ReviewGenerationError as exc:
            lines.append(f"**ERROR ({type(exc).__name__})**: {exc}\n")
            continue

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        draft = result.draft

        lines.append(f"tokens: in={result.input_tokens} out={result.output_tokens}\n")
        lines.append("### Actual draft")
        lines.append(
            "- acceptance criteria met="
            f"{draft.acceptance_criteria_assessment.met}: "
            f"{draft.acceptance_criteria_assessment.rationale}"
        )
        lines.append(
            "- constraint violated="
            f"{draft.constraint_compliance_assessment.violated}: "
            f"{draft.constraint_compliance_assessment.rationale}"
        )
        lines.append("- strengths:")
        lines.extend(render_points_section(draft.strengths))
        lines.append("- risks:")
        lines.extend(render_points_section(draft.risks))
        lines.append("- follow_ups:")
        lines.extend(render_points_section(draft.follow_ups))
        lines.append(
            f"- confidence: {draft.confidence} — {draft.confidence_rationale}\n"
        )

        lines.append("### Expected signals (from the labeled eval case)")
        lines.append(f"- risks to catch: {expected['risks']}")
        lines.append(f"- strengths to catch: {expected['strengths']}")
        lines.append(f"- expected confidence: {expected['confidence']}\n")

    lines.append("---")
    lines.append(f"Total tokens: in={total_input_tokens} out={total_output_tokens}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{timestamp}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nReport written to {out_path}")
    print(f"Total tokens: in={total_input_tokens} out={total_output_tokens}")


if __name__ == "__main__":
    run()
