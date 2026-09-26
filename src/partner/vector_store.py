# PostgreSQL + pgvector-backed store for the RAG checkpoint (docs/BUILD_PLAN.md
# Weeks 3-4). Replaces the earlier local JSON file store once the retrieval
# design was proven locally, per the curriculum's local-first-then-pgvector
# sequencing. Schema: document_chunks(file_name, chunk_index, content,
# rag_offset, embedding vector(768)), unique on (file_name, chunk_index),
# hnsw index on embedding using vector_cosine_ops.
import os

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow, dict_row

from partner.config import load_dotenv_if_present
from partner.embeddings import CustomOllamaEmbeddings


def get_connection() -> psycopg.Connection[DictRow]:
    """Open a DATABASE_URL connection with pgvector + dict rows wired up.

    Used as a context manager (`with get_connection() as conn:`): psycopg
    commits on clean exit, rolls back on exception, and closes either way.
    """
    load_dotenv_if_present()
    # psycopg.Connection[DictRow].connect(), not the psycopg.connect() module-
    # level shortcut -- the shortcut (and the classmethod called unparameterized)
    # is typed as always returning Connection[TupleRow], regardless of what
    # row_factory is actually passed. Explicitly parameterizing the class is
    # what makes the checker track dict_row's real row type end to end.
    conn = psycopg.Connection[DictRow].connect(
        os.environ["DATABASE_URL"], row_factory=dict_row
    )
    register_vector(conn)
    return conn


def build_index(chunks) -> None:
    """(Re)index the given chunks.

    For each distinct file_name present in `chunks`, deletes that file's
    existing rows and inserts the new set, all in one transaction -- a
    per-file transactional replace. This avoids two cheaper-but-wrong
    alternatives: truncating the whole table (re-embeds every unrelated
    document and leaves a window where nothing is retrievable) and a plain
    upsert (leaves orphaned rows behind if a file shrinks to fewer chunks
    than it used to have).
    """
    ollama_embedding = CustomOllamaEmbeddings()
    file_names = {c.file_name for c in chunks}

    with get_connection() as conn, conn.cursor() as cur:
        for file_name in file_names:
            cur.execute(
                "DELETE FROM document_chunks WHERE file_name = %s",
                (file_name,),
            )
        for c in chunks:
            embedding = Vector(ollama_embedding.embed_query(c.content))
            cur.execute(
                """
                INSERT INTO document_chunks
                    (file_name, chunk_index, content, rag_offset, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (c.file_name, c.chunk_index, c.content, c.offset, embedding),
            )
