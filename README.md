# AI Programming Partner

A review-first AI programming partner. It gives cited RAG answers over the project's docs, writes structured AI reviews of programming sessions that need human approval before they're saved, and is backed by labeled evals that report failures honestly.

**Python · Anthropic SDK · Ollama (local models) · PostgreSQL + pgvector · Pydantic · pytest**

The rule behind the whole design: **AI output is a draft until a person approves it, and every claim needs evidence.** Model output is validated against strict schemas. "I can't answer this" is an explicit, required field, not a gap left in the text. Every feature was checked against hand-labeled eval cases before it counted as done.

## What it does

| Feature | How it works | Key files |
|---|---|---|
| **Session tracking** | CLI to create, show and complete a programming session. A session can't be marked complete without test evidence and a written explanation | `models.py`, `storage.py`, `cli.py` |
| **AI session review** | Claude (Haiku 4.5) reviews a completed session and returns strengths, risks, follow-ups and a confidence level. Each point cites the session field it's based on. The draft goes to a `.pending` file and is saved only after an explicit `approve-review` | `review.py`, `reviewer.py` |
| **Grounded answers (RAG)** | Docs are chunked, embedded locally with `nomic-embed-text` and stored in pgvector. Retrieval has a calibrated relevance cutoff. The answer model must set `can_answer` before it can write an answer, and must cite chunks when it does | `chunking.py`, `embeddings.py`, `vector_store.py`, `retrieval.py`, `answer.py` |
| **Two answer backends** | Anthropic forced tool-use (`answering.py`), or local `gemma4:26b` with Ollama's JSON-schema constrained decoding (`answering_ollama.py`). Both produce the same `GroundedAnswer` schema | `answering*.py` |

## Evaluation results

Each result below was graded by hand against labeled cases. Output from generative models can't be graded fairly by exact string match, so the reports put expected and actual output side by side.

**AI session review: 16 labeled cases, real Anthropic API** ([cases](docs/eval/session_review_cases.json), [final run](docs/eval/run_results/2026-09-08T223140Z.md))
- The first graded run found **4 real issues**:
  - it treated vague evidence as verified fact (2 cases)
  - it made an unsupported claim
  - it accepted a stale "next action" as real future work
- Each was fixed in the prompt and re-verified on the live API. The final run was 16/16 clean.
- A real run also returned intermittently corrupted JSON. The schema validation caught it, and a bounded retry-on-malformed was added in response.

**RAG: 12 labeled cases, local `gemma4:26b`, real pgvector retrieval** ([report](docs/eval/RAG_FAILURE_MODE_REPORT.md))
- **8 clean passes**, including abstaining correctly when the answer wasn't in the docs and resisting a direct push to use outside knowledge.
- **2 real defects, documented and not yet fixed:**
  - The model read two completed checklist steps as a plan still pending (a failure to reason about time and status).
  - A known retrieval gap when an answer is split across chunk boundaries. This one was predicted before the run and then confirmed.
- **1 labeling mistake**, caught by the real run and corrected in the open.
- **1 schema limitation:** compound questions can't be partially answered.
- [Ragas](https://docs.ragas.io/) was set up and run against a local judge model, but every metric returned `NaN` because of an async incompatibility in its executor. The fallback is a scripted, reproducible runner ([`scripts/run_rag_eval.py`](scripts/run_rag_eval.py)).

**Tests:** 64 passing (`uv run pytest`). Most of them fake the model client, so the full suite needs no API key. A few integration tests use real Postgres and Ollama.

## Design decisions

- **Explicit "can't answer."** A bare `answer: str` field pushes a model to make something up from loosely related chunks. `GroundedAnswer` requires `can_answer` and enforces its rules in a Pydantic validator: an answer and non-empty citations when `true`, and no answer when `false`.
- **The schema guarantees shape, and validation guarantees meaning.** On the first real run, Ollama's constrained decoding produced valid JSON that silently left out `citations`. Only the Pydantic validator caught it, which is why validation, not the JSON schema, is where the rules are enforced.
- **Retrieval cutoff calibrated on real data.** A clearly unrelated query ("best pizza recipe") scored about 0.39, while real questions scored 0.60–0.71. The cutoff is `min_score=0.45`, below which the system says "no relevant document" instead of returning weak matches.
- **Cosine similarity with an HNSW index.** Cosine matches the original hand-built similarity search. HNSW needs no training step on an empty table, unlike IVFFlat.
- **Human approval is structural.** The review command physically can't write the approved file; only `approve-review` can.
- **Retry only what can be retried.** Malformed output is retried a bounded number of times. Timeouts and refusals surface as typed errors.

## Known limitations

- `document_chunks` is keyed on a bare `file_name`. Two different files with the same name (for example `docs/TASKS.md` and a test fixture `TASKS.md`) overwrite each other's rows. The eval corpus uses unique names to avoid this; the real fix (keying on the full path) is still open.
- The Anthropic RAG backend is unit-tested but hasn't been evaluated against the real API. The RAG evals ran on the local model.
- There's no CI yet.

## In progress: an agent workflow (Weeks 5–6)

A LangGraph agent that routes each question (search the docs, read a session record through a read-only MCP tool, or decline as out of scope), drafts a grounded answer, and **pauses for human approval** using `interrupt()` with a Postgres checkpointer. It retries once with a rewritten query, and every run ends in a named outcome. It will ship with permission and prompt-injection tests and a Docker Compose demo.
[Design spec](docs/specs/2026-09-25-weeks-5-6-agent-design.md) · [Checkpoint 1 plan](docs/plans/2026-09-25-weeks-5-6-checkpoint-1.md)

## Running it locally

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, [Ollama](https://ollama.com/).

```bash
uv sync --extra dev
cp .env.example .env                 # fill in DATABASE_URL (and ANTHROPIC_API_KEY for review)

# Postgres + pgvector
docker run -d --name ai-partner-pgvector -p 5432:5432 \
  -e POSTGRES_USER=partner -e POSTGRES_PASSWORD=change-me -e POSTGRES_DB=ai_programming_partner \
  -v ai_partner_pgvector_data:/var/lib/postgresql/data pgvector/pgvector:pg16
docker exec -i ai-partner-pgvector psql -U partner -d ai_programming_partner < sql/schema.sql

# Local models
ollama pull nomic-embed-text
ollama pull gemma4:26b

uv run pytest                        # full suite
uv run python -m partner.chunking    # index docs/*.md into pgvector
uv run python scripts/run_rag_eval.py
```

Session workflow:

```bash
uv run python -m partner.cli create --file sessions/demo.json --goal "Build a calculator" \
  --project-path calculator --constraints "stdlib only" \
  --acceptance-criteria "All tests pass" --next-action "Write the first test"
uv run python -m partner.cli complete --file sessions/demo.json \
  --evidence "pytest: 14 passed" --explanation "..."
uv run python -m partner.cli review --file sessions/demo.json          # draft -> .review.pending.json
uv run python -m partner.cli approve-review --file sessions/demo.json  # human approval -> .review.json
```

## How this was built

This is a learning project, built in staged checkpoints: design first, then labeled evals before trusting any model output, then an explanation of each concept before a stage counts as closed. AI assistance (Claude Code) was used for implementation, debugging and review. Design decisions and eval grading were made and checked by me, and the reasoning is in [`docs/TASKS.md`](docs/TASKS.md) and the eval reports.

— Wen Yen
