# Programming Partner Session

## Goal

Build the first vertical slice: a usable local CLI for tracking an
AI-assisted programming session. Its `Session` model must enforce that a
session cannot be marked complete without evidence and an explanation
checkpoint.

## Project and scope

`ai-programming-partner`, using Python 3.11+ and pytest. The first slice uses
the standard library, `argparse`, readable JSON, no network calls, and no
credentials. The work is staged as model, storage, CLI, and end-to-end
checkpoints.

## Constraints

- No autonomous execution, deployment, or unrestricted shell access.
- No API keys or private data stored in the project.
- Session records remain human-readable outside the application.
- The model remains independent of file operations.

## Checkpoint status

- Checkpoint 1 — model and transition tests: complete
- Checkpoint 2 — JSON storage and round-trip tests: complete
- Checkpoint 3 — CLI create/show/complete flow: next

## Acceptance criteria for the complete vertical slice

- [ ] A new session can be created from a CLI command.
- [ ] The session record stores goal, acceptance criteria, status, and next action.
- [ ] Completion is blocked without evidence and an explanation checkpoint.
- [ ] The stored record is readable as ordinary JSON.
- [ ] Automated tests cover creation, storage, loading, and completion gating.
- [ ] The complete flow works from a fresh shell without manually editing internal objects.

## Changes made

Implemented the `SessionStatus` enum, `Session` dataclass, completion-gating
transition, and `IncompleteSessionError` in `src/partner/models.py`.

Implemented JSON storage in `src/partner/storage.py`. Storage writes a
replaceable JSON snapshot and reconstructs a `Session` when loading. The model
does not call storage or monitor the file.

Added five model tests and two storage tests. Storage tests use pytest's
temporary path so they do not modify project data.

## Tests and evidence

Command:

```text
python -m pytest
```

Result:

```text
7 passed in 0.04s
```

The tests cover session creation, completion rejection, successful completion,
JSON readability, and Session save/load round-trip behavior.

## Explanation checkpoint

The model is a dataclass because it stores related session data and avoids
manually writing common methods such as `__init__`, `__repr__`, and `__eq__`.

`SessionStatus` is an enum because the allowed values are limited and known.
The enum values are fixed, while a session's current status can change from
`ACTIVE` to `COMPLETED`.

Completion is a domain rule. The session remains `ACTIVE` when completion is
rejected because incomplete work is not the same as failed work. Completion is
allowed only when both evidence and an explanation checkpoint contain
meaningful text.

The storage tradeoff is a replaceable JSON snapshot rather than multiple JSON
objects appended to one file. This keeps the main record readable with the
standard JSON parser. A future audit log could append separate JSONL events.

The current implementation does not yet provide a CLI or validate every field
when a session is created. Those are later improvements.

## Open questions

None for Checkpoints 1 and 2.

## Next action

Begin Checkpoint 3: implement the CLI `create`, `show`, and `complete`
commands. The CLI should coordinate the model and storage without moving
persistence logic into the model.

