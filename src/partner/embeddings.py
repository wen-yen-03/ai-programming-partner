# Local embeddings via Ollama (docs/BUILD_PLAN.md Weeks 3-4 RAG checkpoint).
# Requires `ollama serve` running locally with the model pulled, e.g.
# `ollama pull nomic-embed-text`.
import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from typing import List

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

class CustomOllamaEmbeddings(BaseModel, Embeddings):
    model: str = Field(default="nomic-embed-text")
    base_url: str = Field(default="http://localhost:11434")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of search documents."""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]