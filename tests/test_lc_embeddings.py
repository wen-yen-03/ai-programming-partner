import partner.chunking as chunking
import requests
from partner import DOCS_DIR, DATA_DIR
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from typing import List
from dataclasses import asdict
from partner.embeddings import CustomOllamaEmbeddings

folder_path = DOCS_DIR / "pytests" / "chunks"
embeddings_path = DATA_DIR / "pytests" / "embeddings" / "embeddings.json"

def test_langchain_core_embeddings_ollama() -> None:
    chunks = chunking.chunk_file(folder_path / "TASKS.md", chunk_size=10)
    embeddings = CustomOllamaEmbeddings()
    records = [{**asdict(c), "embedding": embeddings.embed_query(c.content)} for c in chunks]

    assert records