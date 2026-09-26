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
