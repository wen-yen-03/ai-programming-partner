# RAG retrieval/answer failure-mode report

Graded run: `docs/eval/run_results/rag_eval_2026-09-11T025935Z.md` (12/12
labeled cases from `docs/eval/rag_cases.json`, local Ollama `gemma4:26b` via
`src/partner/answering_ollama.py`, real Postgres + real Ollama retrieval,
no mocking). Graded by hand against the labels, same as the session-review
feature's eval process -- matching generative output to a label is
inherently fuzzy, so this is human review of real output, not an
auto-scored pass/fail.

## Summary

| Case | Category | Verdict |
|---|---|---|
| rag-01 | happy_path_single_chunk | Pass |
| rag-02 | happy_path_single_chunk | Pass (exact number, not vague) |
| rag-03 | happy_path_full_list | Pass (all 5 items, none dropped) |
| rag-04 | happy_path_single_chunk | **Fail — temporal/status misread** |
| rag-05 | chunk_boundary_split | **Fail — retrieval incompleteness, predicted and confirmed** |
| rag-06 | happy_path_single_chunk | Pass |
| rag-07 | correct_abstain_absent | Pass |
| rag-08 | correct_abstain_borderline | Pass (correctly hedges on genuinely ambiguous evidence) |
| rag-09 | hallucination_contradiction_risk | Pass — **and the original label was wrong, not the model** (corrected in `rag_cases.json`) |
| rag-10 | unsupported_addon_claim_risk | **Schema limitation, not a clean pass/fail** |
| rag-11 | guardrail_outside_knowledge | Pass — held under direct adversarial pressure |
| rag-12 | self_referential_real_bug | Pass |

**8 clean passes, 2 real defects, 1 case-design bug caught and fixed, 1
architectural limitation surfaced.** Weighted more toward finding real
issues than toward "everything works," matching the session-review eval
set's own stated bias (`docs/eval/README.md`).

## Real defect 1: temporal/status misreading (rag-04)

Question: *"What database does this project use for storing chunk
embeddings?"* The retrieved chunks (`eval_tasks_snapshot.md` #6 and #7)
both carry a completed `[x]` checkbox: #6 documents the original local
JSON store (an earlier, since-superseded step), #7 documents the
PostgreSQL + pgvector migration, also marked done. The model's answer:

> "The project currently stores chunk metadata and embeddings locally
> using JSON (deliberately not a real vector DB yet), but it is designed
> to migrate storage to PostgreSQL + pgvector."

This is wrong: both citations are correct, the model didn't fabricate
anything, but it read a checklist of two *sequential, both-completed*
steps as *one still-pending plan*, describing a finished migration as a
future intention. Not a hallucination in the usual sense (no invented
facts) and not a retrieval miss (the right chunks were retrieved) — a
**reasoning failure about temporal/status language in the source text
itself**. A model-only fix (e.g. explicitly prompting "state whether a
`[x]` item is a *finished* fact, not a plan") is plausible; a schema fix
(a `superseded_by` or `as_of` field) is also plausible but adds real
complexity for one case. Left open, not fixed, in this session.

## Real defect 2: chunk-boundary-split retrieval gap (rag-05)

Question: *"What is the acceptance criteria for the first vertical
slice?"* The real answer is a 6-item bulleted list, split by the chunker:
item 1 is the last line of chunk 3, items 2-6 are all of chunk 4. Chunk 4
never made it into the top-3 retrieved results (crowded out by unrelated
chunks that happened to share vocabulary like "session"). The model
correctly answered with item 1 only, correctly cited chunk 3, and did not
hallucinate anything — but the answer is **structurally incomplete because
retrieval never gave it the rest of the list**.

This was a predicted failure mode (see `rag_cases.json`'s notes on this
case) and the real run confirmed it exactly. It's a genuine argument for
either a larger `k` for list-shaped questions, chunk overlap (so a
boundary split doesn't fully separate related content), or a
larger/semantically-aware chunk size — none implemented this session; this
is evidence to act on next, not a fix.

## Case-design bug caught by the real run (rag-09)

The original rag-09 label expected `can_answer: false` for a question the
docs actually *can* answer (accurately, in the negative: "deployment
automation is explicitly out of scope"). The label was wrong because it
was internally inconsistent with `GroundedAnswer`'s own schema —
`answer` must be `null` whenever `can_answer` is `False`, so there was
never a way to express "correctly explain what's NOT true" as an abstain
in the first place. The model's real answer was accurate, well-cited, and
correctly graded a pass once the label itself was fixed. Recorded directly
in `rag_cases.json`'s notes rather than silently corrected, the same way
Checkpoint 1's false "miss" verdicts were corrected in the open rather
than quietly edited away.

## Architectural limitation surfaced (rag-10)

Question: *"What Python testing framework should I use for this kind of
project, and why is it better than the alternatives?"* — a compound
question with one groundable half (pytest is named) and one ungroundable
half (no comparison to alternatives exists in the corpus). The model
abstained on the *entire* question (`can_answer: false`) rather than
answering the groundable half and declining only the rest.

This is arguably the safest available behavior given the schema: `can_answer`
is a single boolean for the whole question, with no way to express partial
groundedness. It avoided fabricating the ungrounded half at the cost of not
delivering the grounded half at all. Whether that tradeoff is acceptable is
a real product decision (a stricter feature might prefer exactly this
conservative behavior; a more helpful one might want partial answers with
per-sentence grounding) — not resolved here, but the schema genuinely
cannot express the alternative today even if a future decision wanted it.

## What held up well

- **The abstain design works under real adversarial pressure** (rag-11):
  an explicit in-question instruction to ignore retrieved context and use
  outside knowledge did not break the grounding.
- **Borderline evidence was handled honestly** (rag-08): the model
  correctly noticed a `logger.info(...)` mention without inferring an
  unstated library name from it.
- **List completeness held on a single-chunk list** (rag-03): all 5 items
  reported, none dropped, when the whole list fit in one chunk (contrast
  with rag-05, where the list was split across chunks and failed).
- **Precision, not just topical correctness** (rag-02): an exact number
  (16) reported correctly, not a vaguer approximation.

## What this report does not cover

- A real side-by-side against the Anthropic backend — only Ollama has been
  run against this dataset (learner's cost-conscious choice this session).
  `src/partner/answering.py` is built and unit-tested but has never been
  run against real questions.
- Automated regression scoring — this is a one-time human-graded run, not
  a CI-integrated pass/fail gate.
- Ragas metrics — attempted (see `docs/eval/RAG_README.md`), but every
  metric computation failed with an internal `ragas.executor` async
  incompatibility (`RuntimeError: Timeout should be used inside a task`),
  most likely a `ragas==0.2.15` vs. Python 3.14 compatibility gap. This
  human-graded report is the accepted fallback evidence per `AGENT.md`.
