# Build Plan

## Product goal

Create a local-first programming partner that supports AI-assisted development while making the learner’s reasoning, review, and evidence visible.

## MVP boundary

The first version should be useful without paid APIs or autonomous execution. It should demonstrate the software-engineering skills expected around AI products before adding LLM calls.

### MVP capabilities

- Create a session with a goal, repository/project path, constraints, and acceptance criteria.
- Display a checklist for inspect → plan → build → test → review → explain.
- Store a human-readable session record.
- Capture changed files, test commands/results, open questions, and next action.
- Provide a review prompt that asks the learner to explain the implementation and tradeoffs.

## Role-aligned evidence

This project should demonstrate: clean Python, a usable CLI/API contract, persistence, validation, tests, error handling, human approval, readable audit records, and clear product reasoning. Later extensions should demonstrate an LLM API, structured output, retrieval with citations, evaluation, observability, and a constrained tool workflow.

## Planned AI feature (Weeks 1–2 target)

**AI-assisted session review.** Given a completed session record (goal,
changes made, tests/evidence, explanation-checkpoint answers), an LLM call
produces a structured review draft — a Pydantic schema of strengths, risks,
suggested follow-ups, confidence, and a citation back to which session field
informed each point. Stack: raw Anthropic or OpenAI SDK, no orchestration
framework (that's introduced later, in Weeks 7–10, once retrieval/agent
complexity justifies it — see `AGENT.md`).

Originally scoped for Week 5–6; moved up to Weeks 1–2 as part of the
AI-first curriculum resequencing agreed 2026-09-06/08 (see
`progress/CURRENT.md` and `resources/ROLE_ALIGNMENT.md`) so the AI feature
and its evaluation loop come before RAG and agent work rather than after.

Rules:

- The draft is never auto-saved. The learner approves or edits it before
  it's persisted, same as the human-approval principle the rest of this
  project already enforces for completion.
- Cap max output tokens and log token/cost usage per call.
- API key comes from an environment variable, never committed or logged —
  consistent with the "no credentials in the project" rule above.
- By end of Week 6, this feature must be reachable by someone else: a small
  hosted API/CLI demo, or a well-documented local Docker Compose walkthrough
  with a recorded demo if hosting isn't practical yet.

### Explicitly out of scope for MVP

- Automatic commits, deployment, or application submission
- Unrestricted code execution
- Storage of API keys or private credentials
- Claiming that generated code is correct without tests or review
- Replacing the parent AI Engineer Coach progress system

## Acceptance criteria for the first vertical slice

- A new session can be created from a clear text form or command.
- The session records goal, acceptance criteria, status, and next action.
- The workflow cannot be marked complete without a test/evidence note and learner explanation checkpoint.
- The record remains readable outside the application.
- At least one automated test covers session creation and completion gating.
- The slice can be demonstrated end-to-end without manually editing internal objects.

## Suggested technical direction

Choose the simplest stack that supports fast learning and testing. A small Python application with typed models and pytest is a suitable default, but confirm the choice against the learner’s current study goal before implementation.

## Milestones

1. Write the domain model and session-state transitions.
2. Add tests for valid transitions and completion gates.
3. Add persistence to a human-readable local format.
4. Add a minimal CLI or web interface.
5. Add a review/evidence export.
6. Evaluate the workflow with one real programming task.
