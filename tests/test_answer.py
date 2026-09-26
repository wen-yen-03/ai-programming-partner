# Tests for src/partner/answer.py's structured grounded-answer schema.
# Each test traces back to a specific consistency rule from answer.py's
# module docstring: can_answer must agree with whether answer/citations
# are present, not be silently inconsistent.
import pytest
from pydantic import ValidationError

from partner.answer import Citation, GroundedAnswer


def make_citation(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {"file_name": "BUILD_PLAN.md", "chunk_index": 2}
    data.update(overrides)
    return data


def test_can_answer_true_requires_a_non_null_answer() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswer(
            can_answer=True,
            answer=None,
            citations=[Citation(**make_citation())],
            rationale="The chunk directly states the deployment steps.",
        )


def test_can_answer_true_requires_at_least_one_citation() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswer(
            can_answer=True,
            answer="Run `uv run pytest` before every commit.",
            citations=[],
            rationale="The chunk directly states the testing command.",
        )


def test_can_answer_false_forbids_a_non_null_answer() -> None:
    # The dangerous case this schema exists to prevent: the model saying
    # "I can't answer this" while still writing confident prose anyway.
    with pytest.raises(ValidationError):
        GroundedAnswer(
            can_answer=False,
            answer="The deployment strategy is to use Docker Compose.",
            citations=[],
            rationale="The chunks only cover testing, not deployment.",
        )


def test_valid_answer_with_citation_builds() -> None:
    result = GroundedAnswer(
        can_answer=True,
        answer="Run `uv run pytest` before every commit.",
        citations=[Citation(**make_citation())],
        rationale="chunk_index=2 directly states the testing command.",
    )

    assert result.can_answer is True
    assert result.answer is not None
    assert result.citations == [Citation(**make_citation())]


def test_valid_abstain_with_no_answer_builds() -> None:
    result = GroundedAnswer(
        can_answer=False,
        answer=None,
        citations=[],
        rationale="The chunks only cover testing, not deployment.",
    )

    assert result.can_answer is False
    assert result.answer is None


def test_rationale_is_required() -> None:
    with pytest.raises(ValidationError):
        GroundedAnswer(
            can_answer=False,
            answer=None,
            citations=[],
            rationale="",
        )


def test_citation_requires_a_file_name() -> None:
    with pytest.raises(ValidationError):
        Citation(**make_citation(file_name=""))
