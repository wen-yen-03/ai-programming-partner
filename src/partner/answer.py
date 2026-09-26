"""Structured output schema for a document-grounded answer (RAG).

Design goal: make it structurally hard for the model to confidently answer
a question its retrieved chunks don't actually address. Forced tool-use
(see answering.py) gives the model no graceful way to leave a plain
`answer: str` field blank or partially refuse per-question -- that creates
pressure to fabricate a plausible-sounding answer from loosely-related
chunks when the honest response is "I don't know." review.py hit the same
failure shape in a different feature (a bare `met: bool` field forced a
guess when the record didn't actually confirm or deny something) and fixed
it the same way this schema does: make "I can't answer this" a required,
explicit, first-class field rather than something implicit in whatever
text the model happens to produce.

`can_answer` must be set explicitly before `answer` is written, and the
three fields' consistency (answer required + non-empty citations when
can_answer is True; answer must be null when can_answer is False) is
enforced at the Pydantic level -- an inconsistent tool call fails schema
validation rather than being silently trusted by a caller who only reads
one of the fields.
"""

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """One retrieved chunk the answer is grounded in."""

    file_name: str = Field(min_length=1)
    chunk_index: int


class GroundedAnswer(BaseModel):
    """An answer to a question, grounded only in retrieved document chunks."""

    can_answer: bool = Field(
        description=(
            "True only if the retrieved chunks actually contain enough "
            "information to answer the specific question asked -- not "
            "just topically related to it."
        )
    )
    answer: str | None = Field(
        default=None,
        description=(
            "The answer, grounded only in the retrieved chunks. Must be "
            "null when can_answer is False."
        ),
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Retrieved chunks the answer is grounded in. Required "
            "(non-empty) when can_answer is True."
        ),
    )
    rationale: str = Field(
        min_length=1, description="Why can_answer is True or False."
    )

    @model_validator(mode="after")
    def _check_answer_consistency(self) -> "GroundedAnswer":
        if self.can_answer:
            if not self.answer:
                raise ValueError("answer is required when can_answer is True")
            if not self.citations:
                raise ValueError("citations are required when can_answer is True")
        elif self.answer:
            raise ValueError("answer must be null when can_answer is False")
        return self
