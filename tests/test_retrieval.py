import partner.chunking as chunking
import partner.vector_store as vector_store
import partner.retrieval as retrieval
from partner import DOCS_DIR

folder_path = DOCS_DIR / "pytests" / "chunks"

def test_retrieve_after_indexing() -> None:
    # Real Postgres + real Ollama, no mocking -- both are free/local, so this
    # proves the pgvector round-trip actually works rather than just the SQL
    # strings being well-formed.
    chunks = chunking.chunk_file(folder_path / "TASKS.md", chunk_size=10)

    vector_store.build_index(chunks)

    assert retrieval.retrieve("show me tasks I have completed", k=3)
    assert retrieval.retrieve("wheres the best pizza in town?", k=3, min_score=0.45) == []
    assert retrieval.retrieve("wheres the best pizza in town?", k=3, min_score=0.30)

