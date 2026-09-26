# pgvector-backed top-k retrieval (docs/BUILD_PLAN.md Weeks 3-4 RAG
# checkpoint). Postgres does the similarity search directly (hnsw index on
# document_chunks.embedding) instead of loading every record and computing
# cosine similarity in Python.
from pgvector import Vector

from partner.embeddings import CustomOllamaEmbeddings
from partner.vector_store import get_connection


def retrieve(query: str, k: int = 3, min_score: float = 0.0) -> list[dict]:
    """Return the top-k chunks most similar to `query`, each with a `score`.

    `min_score` is a similarity threshold (1.0 = identical, higher is
    better), matching the old JSON-store behavior. pgvector's `<=>` operator
    returns cosine *distance* (1 - similarity), so it's converted once to a
    max-distance filter before querying.
    """
    ollama_embedding = CustomOllamaEmbeddings()
    query_vector = Vector(ollama_embedding.embed_query(query))
    max_distance = 1 - min_score

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT file_name, chunk_index, content, rag_offset AS "offset",
                   1 - (embedding <=> %(qvec)s) AS score
            FROM document_chunks
            WHERE (embedding <=> %(qvec)s) <= %(max_distance)s
            ORDER BY embedding <=> %(qvec)s
            LIMIT %(k)s
            """,
            {"qvec": query_vector, "max_distance": max_distance, "k": k},
        )
        return cur.fetchall()
