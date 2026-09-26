-- Schema for the RAG chunk store (src/partner/vector_store.py, retrieval.py).
-- Apply once to a fresh pgvector database:
--   docker exec -i ai-partner-pgvector psql -U partner -d ai_programming_partner < sql/schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id          bigserial PRIMARY KEY,
    file_name   varchar(100),
    chunk_index integer,
    content     text,
    rag_offset  bigint,
    embedding   vector(768),              -- nomic-embed-text dimensions
    CONSTRAINT uq_document_chunks_file_chunk UNIQUE (file_name, chunk_index)
);

-- Cosine distance (<=>) matches how retrieval scores similarity.
-- hnsw needs no training step on an empty table (unlike ivfflat).
CREATE INDEX IF NOT EXISTS idx_document_chunks
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
