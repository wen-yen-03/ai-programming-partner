# Archived: FastAPI checkpoint stub (2026-09-07)

`src/partner/api.py` (empty) and `tests/test_api.py` (a 12-line unwired
FastAPI stub) were created on 2026-09-07 as the start of a planned "API
checkpoint with FastAPI and Pydantic request/response models." Later the
same day, the curriculum was corrected away from that generic FastAPI-CRUD
detour toward the AI-first sequence (see `progress/DAILY_LOG.md`'s
2026-09-07 "Curriculum update: skill toolkit and adjacent-role scope"
entry). The two files were never removed, `fastapi` was never added to
`pyproject.toml`, and by 2026-09-10 `tests/test_api.py` was breaking
`pytest` collection for the whole suite since `fastapi` isn't installed.

Moved here (`.bak`) instead of deleted, since this project has no git
history to recover them from if they turn out to matter later. Restore by
moving them back to `src/partner/api.py` and `tests/test_api.py` and
adding `fastapi` to `pyproject.toml` if FastAPI work is ever picked back
up — Weeks 5–6's MCP tool or a later portfolio demo could plausibly want
a real API layer.
