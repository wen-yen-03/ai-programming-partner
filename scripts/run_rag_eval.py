"""Run the labeled RAG evaluation cases (docs/eval/rag_cases.json) against
the local Ollama-backed grounded-answer feature and produce a
human-reviewable report.

Rebuilds the eval corpus index immediately before running, from
docs/eval/rag_corpus/ (uniquely-named files -- see docs/eval/rag_cases.json's
"corpus" field for why) rather than trusting whatever happens to already be
in document_chunks. This is a deliberate fix for the file_name-collision bug
documented in progress/DAILY_LOG.md's 2026-09-10 entries: pytest fixture
runs and real production runs share the same document_chunks table keyed
only on a bare file_name, so a stale or wrong snapshot can silently sit
there from an unrelated prior run. Re-indexing a known corpus immediately
before grading makes this script's results reproducible regardless of what
ran before it.

Like scripts/run_eval.py (the session-review feature's eval runner), this
does not auto-grade pass/fail against the labels -- matching generative
output to a label is inherently fuzzy, and the point is human review, not
blind trust. The report lays expected vs. actual side by side.

Usage: python scripts/run_rag_eval.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from partner.answering_ollama import AnswerGenerationError, answer_question  # noqa: E402
from partner.chunking import chunk_file  # noqa: E402
from partner.retrieval import retrieve  # noqa: E402
from partner.vector_store import build_index  # noqa: E402

CASES_PATH = REPO_ROOT / "docs" / "eval" / "rag_cases.json"
CORPUS_DIR = REPO_ROOT / "docs" / "eval" / "rag_corpus"
RESULTS_DIR = REPO_ROOT / "docs" / "eval" / "run_results"


def reindex_eval_corpus() -> None:
    all_chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        all_chunks.extend(chunk_file(path, chunk_size=20))
    build_index(all_chunks)
    print(f"Re-indexed {len(all_chunks)} chunks from {CORPUS_DIR}")


def run_case(case: dict) -> dict:
    question = case["question"]
    retrieved = retrieve(question, k=3, min_score=0.45)
    actual_files = sorted({r["file_name"] for r in retrieved})
    actual_chunks = sorted(
        f"{r['file_name']}#{r['chunk_index']}" for r in retrieved
    )

    try:
        result = answer_question(question, k=3, min_score=0.45)
        answer_summary = {
            "can_answer": result.answer.can_answer,
            "answer": result.answer.answer,
            "citations": [c.model_dump() for c in result.answer.citations],
            "rationale": result.answer.rationale,
        }
        error = None
    except AnswerGenerationError as exc:
        answer_summary = None
        error = f"{type(exc).__name__}: {exc}"

    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "expected_can_answer": case["expected_can_answer"],
        "expected_relevant_files": case["expected_relevant_files"],
        "notes": case["notes"],
        "actual_retrieved_files": actual_files,
        "actual_retrieved_chunks": actual_chunks,
        "answer": answer_summary,
        "error": error,
    }


def format_report(results: list[dict]) -> str:
    lines = [
        "# RAG evaluation run",
        "",
        f"Model: local Ollama (gemma4:26b, via src/partner/answering_ollama.py)",
        f"Corpus: docs/eval/rag_corpus/ (re-indexed immediately before this run)",
        "",
        "Not auto-graded -- matching generative output to a label is inherently",
        "fuzzy. This lays expected vs. actual side by side for human review, the",
        "same pattern as scripts/run_eval.py's session-review report.",
        "",
    ]
    for r in results:
        lines.append(f"## {r['id']} ({r['category']})")
        lines.append("")
        lines.append(f"**Question:** {r['question']}")
        lines.append("")
        lines.append(f"**Expected can_answer:** {r['expected_can_answer']}")
        lines.append(
            f"**Expected relevant files:** {r['expected_relevant_files'] or '(none)'}"
        )
        lines.append(f"**Notes:** {r['notes']}")
        lines.append("")
        lines.append(f"**Actually retrieved files:** {r['actual_retrieved_files']}")
        lines.append(f"**Actually retrieved chunks:** {r['actual_retrieved_chunks']}")
        lines.append("")
        if r["error"]:
            lines.append(f"**ERROR:** {r['error']}")
        else:
            a = r["answer"]
            lines.append(f"**Actual can_answer:** {a['can_answer']}")
            lines.append(f"**Actual answer:** {a['answer']}")
            lines.append(f"**Actual citations:** {a['citations']}")
            lines.append(f"**Actual rationale:** {a['rationale']}")
        lines.append("")
        lines.append("**Verdict (fill in after review):** ")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    reindex_eval_corpus()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question']}")
        results.append(run_case(case))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"rag_eval_{timestamp}.md"
    out_path.write_text(format_report(results), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
