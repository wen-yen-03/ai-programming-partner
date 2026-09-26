# RAG (retrieval + grounded-answer) evaluation set

`rag_cases.json` holds the labeled evaluation cases for the retrieval and
grounded-answer generation feature (`AGENT.md` Weeks 3-4, "grounded
generation and retrieval"; `src/partner/retrieval.py`, `answering.py`,
`answering_ollama.py`). Written after a scenario quiz (see
`progress/DAILY_LOG.md`'s 2026-09-10 entry) that calibrated judgment on real
candidate outputs before any case was drafted, same as the session-review
eval set's process.

## Why a frozen, uniquely-named corpus (`docs/eval/rag_corpus/`)

Not the real `docs/*.md` directly. Two reasons:

1. **Stability.** `docs/TASKS.md` changes almost every session -- a case
   whose expected answer depends on its current content would silently go
   stale.
2. **A real bug found this session.** `document_chunks` keys rows on a bare
   `file_name`, not a full path. This project already has two different
   files both named `TASKS.md` (the real one and a pytest fixture snapshot
   at `docs/pytests/chunks/TASKS.md`) that silently overwrite each other's
   indexed rows depending on which was indexed last -- see
   `progress/DAILY_LOG.md`'s 2026-09-10 correction entry for the full
   story. `docs/eval/rag_corpus/` uses names (`eval_build_plan.md`,
   `eval_tasks_snapshot.md`, `eval_session_template.md`) that can't collide
   with either the real docs or the pytest fixtures, and
   `scripts/run_rag_eval.py` re-indexes this exact corpus immediately
   before every run rather than trusting whatever happens to already be in
   the table.

The corpus is a byte-exact snapshot of the real docs as of 2026-09-10. Chunk
boundaries (`chunk_size=20`, matching production) are recorded per case in
`expected_relevant_chunk_indices` so a case's expected grounding stays
verifiable even as the real docs keep changing.

## Why can_answer/relevant-files labels, not exact-answer labels

Same reasoning as the session-review set: the answer text is generative, so
exact-string labels would be too brittle. Each case instead labels:

- `expected_can_answer` -- should the feature answer at all, or correctly
  abstain?
- `expected_relevant_files` / `expected_relevant_chunk_indices` -- which
  chunks a *correct* retrieval should surface, so retrieval quality can be
  graded separately from answer quality (a case can fail at either layer,
  and the eval run's report shows both: what was actually retrieved, and
  what the model actually answered).
- `notes` -- the specific failure mode this case targets, since a case's
  pass/fail meaning is often subtler than "did it get the right file"
  (e.g. rag-09 fails an answer that names the *wrong file* just as much as
  one that hallucinates from thin air).

## Case schema

```json
{
  "id": "rag-NN",
  "category": "happy_path_single_chunk | happy_path_full_list | chunk_boundary_split | correct_abstain_absent | correct_abstain_borderline | hallucination_contradiction_risk | unsupported_addon_claim_risk | guardrail_outside_knowledge | self_referential_real_bug",
  "question": "...",
  "expected_can_answer": true,
  "expected_relevant_files": ["eval_build_plan.md"],
  "expected_relevant_chunk_indices": [3],
  "notes": "what this case specifically targets and why"
}
```

## Coverage in this set (12 cases)

- **Happy path** (5: rag-01, 02, 03, 04, 06): clean, answerable questions
  spanning a single-chunk fact, an exact number (tests precision, not just
  topical correctness), a multi-item list (tests completeness, not just
  "found something relevant"), and a fact spanning a chunk boundary.
- **Chunk-boundary split** (1: rag-05): the expected answer's content is
  split across two adjacent chunks by the chunker -- a real, structural RAG
  failure mode distinct from anything about the model or the embeddings.
- **Correct abstain** (2: rag-07, 08): rag-07 is unambiguous (genuinely
  absent topic); rag-08 is deliberately borderline (the text implies but
  never states a fact) and is flagged in its own notes as the hardest case
  in the set to grade fairly, not a clean pass/fail.
- **Hallucination / contradiction risk** (1: rag-09): the cited chunk says
  the *opposite* of a plausible-sounding answer -- the sharpest version of
  "citation exists but doesn't support the claim."
- **Unsupported add-on claim** (1: rag-10): a question with a genuinely
  grounded half and an ungrounded half in the same sentence, testing
  whether the answer over-reaches on the second half.
- **Guard-rail** (1: rag-11): an explicit in-question instruction to ignore
  the retrieved context and use outside knowledge instead -- tests whether
  grounding holds under direct pressure, not just under a neutral
  out-of-scope question.
- **Self-referential** (1: rag-12): grounded in this exact session's own
  real bug-fix writeup, inside the corpus snapshot -- a genuine,
  non-synthetic happy-path case.

## Ragas: attempted, real result

Per `AGENT.md`'s "attempt Ragas; pytest fallback accepted if it can't run,"
a real attempt was made — not skipped. `ragas` was not in `pyproject.toml`;
a bare `uv run --with ragas` install failed at import time
(`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`
— `ragas` unconditionally imports a Vertex AI integration class that moved
to a separate optional package in the installed `langchain-community`
version). Pinning `langchain-community==0.3.7` alongside `ragas==0.2.15`
resolved the import. A local judge (`ChatOllama(model="gemma4:26b")` +
`OllamaEmbeddings(model="nomic-embed-text")`, via `langchain-ollama`,
wrapped in `ragas.llms.LangchainLLMWrapper`/
`ragas.embeddings.LangchainEmbeddingsWrapper`) was configured so no
OpenAI key was needed. See `progress/DAILY_LOG.md`'s 2026-09-10 entry for
the actual run result and timing.

`scripts/run_rag_eval.py` is the accepted-fallback pytest-adjacent runner
(same pattern as `scripts/run_eval.py`): it re-indexes the frozen corpus,
runs every labeled case for real, and writes a side-by-side
expected-vs-actual report to `docs/eval/run_results/` for human review —
this is the primary evidence path regardless of the Ragas outcome, since
Ragas's own metrics still require a human to sanity-check what "faithful"
or "relevant" meant for a given local-model judge run.

## What this set does not cover yet

- A second backend (Anthropic) run for a real side-by-side comparison —
  only Ollama has been run against these cases so far, per the learner's
  cost-conscious choice this session.
- Automated pass/fail scoring against the labels — like the session-review
  set, this is human-reviewed against expected vs. actual, not
  auto-graded, because matching generative output to a label is inherently
  fuzzy.

## Adding a case

New cases should target a specific, real failure mode grounded in the
frozen corpus's actual content (verify with
`uv run python -c "from partner.chunking import chunk_file; ..."` against
`docs/eval/rag_corpus/`, don't assume what a chunk contains), following the
schema above.
