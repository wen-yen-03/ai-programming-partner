# Session-review evaluation set

`session_review_cases.json` holds the labeled evaluation cases for the
AI-assisted session review feature (`docs/BUILD_PLAN.md`, "Planned AI
feature"; `AGENT.md` Weeks 1–2). This is step 1 of AI Checkpoint 1 — written
before any SDK/model code, so the feature has a concrete target to be built
and tested against.

## Why criteria-based labels, not exact-output labels

The feature's output (strengths/risks/follow-ups/confidence, Pydantic
schema) is generative — the same correct review can be phrased many ways.
Exact-string expected outputs would be too brittle to use as pass/fail
labels. Instead, each case lists the **signals a correct review must
surface**: specific facts, contradictions, or omissions that exist in the
input `Session` record. Scoring (once the SDK feature exists) checks whether
the generated draft's risks/strengths/follow-ups cover these signals
(keyword or semantic match), not whether the wording matches.

## Case schema

```json
{
  "id": "case-NN-short-name",
  "category": "happy-path | vague-evidence | constraint-violation | ...",
  "session": { /* the exact fields Session accepts, per src/partner/models.py */ },
  "expected_signals": {
    "strengths": ["..."],
    "risks": ["..."],
    "follow_ups": ["..."],
    "confidence": "high | medium | low | n/a",
    "confidence_rationale": "why that confidence level, not another",
    "required_citations": ["which Session fields a correct review must ground its points in"]
  },
  "notes": "what this case is specifically testing and why it's in the set"
}
```

`session` fields always match `src/partner/models.py`'s `Session` dataclass
(`goal`, `project_path`, `constraints`, `acceptance_criteria`, `status`,
`evidence`, `explanation_checkpoint`, `next_action`).

## Coverage in this set (16 cases)

- **Happy path** (2): evidence genuinely supports the acceptance criteria,
  including the two real records from `sessions/demo.json` and
  `sessions/practice.json`.
- **Vague/unverifiable evidence** (4): claims without proof, pass-count-only
  evidence that doesn't confirm specifically named required cases, and
  evidence too thin for the size of the change.
- **Constraint violations** (2): an explicit technology-restriction
  violation and a scope-creep violation.
- **Internal contradictions** (3): evidence that documents its own test
  failures against a "completed" status, a stale/already-done next_action,
  and a next_action that duplicates completed work.
- **Omission-based risk** (1): a security-relevant property (token expiry)
  that's neither confirmed nor denied — "silence is a risk," harder than an
  explicit contradiction.
- **Input-quality problems** (1): acceptance criteria too vague to
  measure — a problem with the record itself, not just the evidence.
- **Mixed signal** (1): a strong explanation paired with thin test count,
  to check the review holds both in tension instead of anchoring on one.
- **Edge cases** (2): an empty-but-valid field (`constraints`) that
  shouldn't be treated as a manufactured risk, and a `status: "active"`
  record that should be rejected by the feature's input guard rather than
  reviewed at all.

This deliberately weights toward failure modes over clean passes (2 clean
vs. 14 with at least one real issue), because the point of this feature is
catching what a learner's self-report misses — a reviewer that only ever
says "looks good" isn't useful evidence for the portfolio story.

## What this set does not cover yet

- SDK-response failure modes (malformed JSON, refused, timed-out) — those
  are mocked-API tests against the SDK call itself, not `Session`-content
  eval cases, and are tracked separately in `docs/TASKS.md`.
- A scoring harness that actually runs the feature against these cases and
  checks `required_citations`/signal coverage — next step after the schema
  and SDK call exist.

## Adding a case

New cases should name a specific, real failure mode (not a rephrasing of an
existing one) and follow the schema above. Prefer grounding `session`
content in situations plausible for this project or a small app, per the
existing cases.
