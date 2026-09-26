# Initial Tasks

- [x] Confirm the first vertical slice and technical stack. (Python + pytest,
      stdlib-only CLI; see docs/sessions/2026-09-06-vertical-slice-1.md)
- [x] Define the session state model and explain the transition rules.
      (`src/partner/models.py`: `Session`, `SessionStatus`, `complete()` gate)
- [x] Implement session creation through the CLI.
- [x] Implement completion gating.
- [x] Add tests for the state transitions and rejection cases.
      (`tests/test_models.py`)
- [x] Add human-readable JSON storage. (`src/partner/storage.py`,
      `tests/test_storage.py`)
- [x] Implement `show` and `complete` CLI commands.
- [x] Demonstrate the complete flow end-to-end. (14 tests passing; see
      `docs/sessions/2026-09-07-checkpoint-4-cli.md`)
- [ ] Run one real learner session and record the results.
- [x] Update the parent `progress/CURRENT.md` with the selected next action.

Note: this checklist was out of sync with actual progress — items above were
completed as of the 2026-09-07 checkpoint-4 session but left unchecked.
Corrected 2026-09-08.

## AI Checkpoint 1 (Weeks 1–2): session-review feature + evaluation loop

Per `docs/BUILD_PLAN.md`'s "Planned AI feature" section and
`progress/CURRENT.md`'s next action.

- [x] Draft 10–20 labeled evaluation cases for the session-review feature
      (`docs/eval/session_review_cases.json` + `docs/eval/README.md`; 16
      cases spanning happy-path, vague evidence, constraint violations,
      internal contradictions, an omission-based security risk, ambiguous
      criteria, mixed signal, and two guard-rail edge cases).
- [x] Define the Pydantic schema for the structured review draft (strengths,
      risks, follow-ups, confidence, per-point field citation).
      (`src/partner/review.py`, `tests/test_review.py`, 9 tests passing.
      Directly encodes the 5 failure modes from the review-draft quiz: a
      `SessionField` citation required on every `ReviewPoint`; separate
      mandatory `acceptance_criteria_assessment` and
      `constraint_compliance_assessment` fields, each `bool | None` so an
      unaddressed clause isn't forced into a guess; and `assert_reviewable()`
      gating on `status == "completed"` before any draft is generated.)
- [x] Implement the raw Anthropic SDK call producing that structured output.
      (`src/partner/reviewer.py`: `generate_review_draft()` forces a tool
      call matching `SessionReviewDraft.model_json_schema()` via
      `tool_choice`, then validates the tool input with Pydantic — the
      real enforcement point, since a tool schema is only a hint to the
      model, not a guarantee.)
- [x] Add tests for the four required failure modes: valid, malformed,
      refused, timed-out. (`tests/test_reviewer.py`, fake-client based, no
      network/credentials needed; 10 tests passing — the four required
      modes plus the status guard, API-key handling, and retry-on-malformed
      (recovers / gives up after `max_malformed_retries`). Retry-on-
      malformed was added after a real eval run hit an intermittent
      corrupted-JSON glitch on Haiku (docs/eval/run_results/
      2026-09-08T221219Z.md, case-06) — `ReviewMalformedError` correctly
      caught it rather than returning bad data, but there was no automatic
      second attempt yet. Re-ran the full 16-case suite after adding the
      retry: 16/16 clean, zero errors
      (docs/eval/run_results/2026-09-08T223140Z.md).)
- [x] Confirm the draft is never auto-saved — the learner approves/edits it
      before persistence. (`src/partner/cli.py`: `review` writes only to
      `<file>.review.pending.json`; a separate, explicit `approve-review`
      command promotes it to `<file>.review.json`. Proven by a real CLI
      test, not just construction: `test_review_writes_pending_file_but_not_approved`
      asserts the approved file does not exist after `review` alone.
      6 new CLI tests in `tests/test_cli.py`, all using a fake Anthropic
      client — no real API key spent building or testing this command.)
- [x] Log token/cost usage per call; cap max output tokens.
      (`ReviewResult.input_tokens`/`output_tokens`, `logger.info(...)` in
      `generate_review_draft()`; `max_output_tokens` defaults to 2048 and
      is a parameter.)
- [x] Run the feature against the labeled eval cases and record pass/fail
      per case. (`docs/eval/run_results/2026-09-08T220525Z.md`, Haiku.
      Learner manually graded all 16 cases against labels — see
      `docs/eval/session_review_cases.json` — and corrected two false
      "miss" verdicts to reveal 11/15 solid, 4 real issues found: case-02/
      case-13 treated vague evidence as verified fact; case-06 missed a
      goal-restatement explanation and hallucinated an unsupported claim;
      case-11 mischaracterized a stale next_action as legitimate future
      work. Fixed the prompt to explicitly check for stale next_actions
      and goal-restatement explanations, and to require citations to
      actually support (not just relate to) a claim; case-06 and case-11
      re-verified fixed on re-run. Follow-up: added the same "unproven bare
      assertion" check to the prompt for case-02/case-13's vague-evidence
      issue; re-verified fixed on re-run
      (docs/eval/run_results/2026-09-08T221219Z.md — both now correctly
      flag the bare claim and land on confidence "low", matching labels).
      Final state: 16/16 cases clean, all four identified real issues
      fixed and re-verified against the live API.)
- [x] Update `progress/CURRENT.md` and `progress/DAILY_LOG.md` with the
      resulting evidence. Verified 2026-09-08: full suite (39 tests, all
      passing, `python -m pytest`) confirms the `review`/`approve-review`
      CLI wiring and its 6 tests are present and green, closing the item
      above (this checkbox itself had been left unchecked after the CLI
      work landed).

## Weeks 3–4 Checkpoint: grounded generation and retrieval

Per `AGENT.md`'s "Weeks 3–4: Grounded generation and retrieval" section.

- [x] Chunk local hardcoded docs (`src/partner/chunking.py`).
- [x] Embed chunks via local Ollama `nomic-embed-text`
      (`src/partner/embeddings.py`, direct `requests.post`).
- [x] Store chunk metadata + embeddings locally (`src/partner/vector_store.py`,
      JSON — deliberately not a real vector DB yet, per the local-first-then
      -pgvector sequencing).
- [x] Retrieve via naive top-k cosine similarity (`src/partner/retrieval.py`).
- [x] Citations (file name + computed line range) and an explicit
      "not found" behavior below a calibrated `min_score` threshold.
- [x] Introduce LangChain only for retrieval orchestration. The
      `CustomOllamaEmbeddings` LangChain `Embeddings` subclass
      (`tests/test_lc_embeddings.py`, added 2026-09-09) is now the real
      embeddings path: `src/partner/embeddings.py`, `vector_store.py`, and
      `retrieval.py` all use it. `tests/test_retrieval.py` still passes
      unchanged through the new interface. `tests/test_lc_embeddings.py`
      now imports `CustomOllamaEmbeddings` from `partner.embeddings`
      rather than defining its own copy. Full suite 41/41 passing
      (`uv run pytest -q`, verified 2026-09-10).
- [x] Migrate storage to PostgreSQL + pgvector once the local retrieval
      design is proven. `document_chunks` table (file_name, chunk_index,
      content, rag_offset, embedding vector(768)) with a UNIQUE(file_name,
      chunk_index) constraint and an `hnsw (embedding vector_cosine_ops)`
      index, in the `ai-partner-pgvector` Docker container. `build_index()`
      does a per-file transactional delete-then-insert (not truncate-all,
      not plain upsert — avoids orphaning rows if a file's chunk count
      shrinks); `retrieve()` runs `ORDER BY embedding <=> %(qvec)s LIMIT k`
      directly in Postgres instead of loading everything into Python.
      `numpy` dropped (no longer needed once Postgres does the similarity
      math). `.env` loader deduplicated into `src/partner/config.py`,
      shared with `reviewer.py`. Full suite 41/41 passing
      (`uv run pytest -q`, verified 2026-09-10) including a real-Postgres
      + real-Ollama test (`tests/test_retrieval.py::test_retrieve_after_indexing`,
      no mocking); real end-to-end run via `python -m partner.chunking`
      verified against actual rows in the container.
- [x] Implement grounded answer generation: given a question and its
      retrieved chunks, call the Anthropic SDK to produce a cited answer
      (reuse `reviewer.py`'s structured-output pattern), with an explicit
      no-answer path when `retrieve()` returns `[]`. Split out because
      `AGENT.md`'s Weeks 3–4 section is titled "grounded generation and
      retrieval" and asks for an "answer failure-mode report," but nothing
      built so far actually generates an answer from retrieved chunks —
      decided 2026-09-10 (learner chose "add generation, then evaluate
      both" over a retrieval-only fallback).
      `src/partner/answer.py` (`GroundedAnswer`/`Citation` schema, a
      `can_answer` field required before `answer` can be written, enforced
      via a Pydantic `model_validator` so an inconsistent tool call is
      caught as malformed rather than silently trusted) +
      `src/partner/answering.py` (`generate_answer()`/`answer_question()`,
      same forced-tool-use + four-failure-mode pattern as `reviewer.py`).
      Also, as a deliberate free experiment: `src/partner/answering_ollama.py`,
      a parallel implementation backed by local Ollama (`gemma4:26b`) using
      structured-output JSON-schema constraints instead of forced
      tool-use. Real result: the JSON-schema constraint stops malformed
      JSON but not a *semantically* incomplete response — the very first
      live run returned `can_answer: true` with `citations` entirely
      omitted (schema-valid, since `citations` has a Pydantic default;
      caught by the `model_validator`, not the schema). A second real-run
      write-up initially described a specific retrieval miss against a
      quote that, on checking, does not actually exist in any indexed
      document — corrected in `progress/DAILY_LOG.md`. The real finding
      underneath it: `document_chunks` keys on a bare `file_name`, and
      this project has two different files both named `TASKS.md` (this
      real file and a frozen pytest fixture at
      `docs/pytests/chunks/TASKS.md`), so whichever gets indexed last
      silently overwrites the other's rows — a real, still-open bug. See
      `progress/DAILY_LOG.md`'s 2026-09-10 entries for the full trail,
      correction included. 64/64 tests passing.
- [x] Build a labeled evaluation dataset and a retrieval/answer
      failure-mode report, covering both retrieval quality (right chunks
      retrieved for a query) and answer quality (grounded, cited, no
      hallucination, correctly abstains when nothing relevant was
      retrieved). Attempt Ragas; a local pytest-driven evaluation suite is
      the accepted fallback if Ragas can't run.
      A scenario quiz (5 real candidate outputs, free-text judgment) ran
      before any case was written, same pattern as Checkpoint 1 — see
      `progress/DAILY_LOG.md`'s 2026-09-10 entry for the graded quiz.
      12 labeled cases (`docs/eval/rag_cases.json`) grounded in a frozen,
      uniquely-named corpus (`docs/eval/rag_corpus/`) built specifically
      to dodge the file_name-collision bug found earlier this session.
      Ragas genuinely attempted: `ragas==0.2.15` + `langchain-community==0.3.7`
      (a version pin fix for an otherwise-broken import) ran end-to-end
      against a local Ollama judge (no OpenAI key), but every metric
      computation failed with `RuntimeError: Timeout should be used inside
      a task` (`nan` for every score) — a real `ragas.executor` async
      incompatibility, most likely Python 3.14-related. `scripts/run_rag_eval.py`
      is the accepted-fallback runner: re-indexes the frozen corpus fresh
      every run (so the collision bug can't corrupt results), runs all 12
      cases for real against `src/partner/answering_ollama.py`
      (`docs/eval/run_results/rag_eval_2026-09-11T025935Z.md`). Hand-graded
      result: 8/12 clean passes, 2 real defects found (a temporal/status
      misread — completed-checklist language read as a future plan despite
      correct citations; and a predicted-and-confirmed chunk-boundary
      retrieval gap — a 6-item list split across chunks only partially
      retrieved), 1 case-design bug caught by the real run and corrected in
      the open (an expected-abstain label that was internally inconsistent
      with the schema itself), and 1 genuine architectural limitation
      surfaced (a compound question with a partly-groundable, partly-ungroundable
      half — the schema's all-or-nothing `can_answer` boolean can't express
      partial grounding). Full writeup: `docs/eval/RAG_FAILURE_MODE_REPORT.md`.
- [x] Update `progress/CURRENT.md` and `progress/DAILY_LOG.md` with the
      resulting evidence.
