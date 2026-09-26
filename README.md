# AI Programming Partner

An AI-assisted programming partner for the AI Engineer preparation workspace. It helps the learner plan, implement, test, debug, review, and explain software while keeping the learner in control of decisions and authorship.

## Purpose

The partner may actively help with programming. It can inspect the project, propose designs, write or revise code, run tests, explain errors, and review changes. It must keep the work understandable, evidence-based, and suitable for interview discussion.

## Operating principles

- Be transparent about what the AI changed, why, and what remains uncertain.
- Use small, reviewable increments instead of silently rewriting the project.
- Ask the learner to understand or verify important decisions before calling work complete.
- Prefer tests, examples, and observable evidence over confident claims.
- Keep credentials, private data, and confidential source material out of the project.
- Record meaningful progress in the parent workspace progress files.

## Planned first version

1. Session brief: goal, constraints, acceptance criteria, and learner confidence.
2. Project inspection: relevant files, existing behavior, tests, and risks.
3. Build loop: plan, implement, test, review, explain.
4. Learning checkpoint: questions the learner should answer without copying.
5. Evidence record: changed files, test results, limitations, and next action.

## Suggested first build

Start with a small local CLI that accepts a programming task, stores a session brief, and produces a reviewable implementation checklist. This is deliberately a software-engineering foundation for later AI features. Add model integrations only after the workflow, tests, and safety boundaries are working.

## Project status

- Stage: First vertical slice, revised to match applied AI engineering work
- Owner: Wen Yen
- Stack: Python 3.12+, stdlib-only CLI, pytest (see `docs/BUILD_PLAN.md`)
- AI assistance: explicitly allowed for implementation, debugging, testing, and review
- Current next action: learner implements the model and tests, then wires readable JSON storage and the `create`, `show`, and `complete` CLI commands.
