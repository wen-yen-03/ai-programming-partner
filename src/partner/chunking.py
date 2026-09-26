import io
import sys
from itertools import batched
from pathlib import Path
from pydantic.dataclasses import dataclass
import partner.vector_store as vector_store
import partner.retrieval as retrieval
from partner import DOCS_DIR


@dataclass
class Chunk:
    file_name: str
    chunk_index: int
    content: str
    offset: int

def chunk_file(file_path: Path, chunk_size: int = 20) -> list[Chunk]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    for i, chunk in enumerate(batched(lines, chunk_size)):
        chunk_content = "".join(chunk)
        offset = i * chunk_size
        chunks.append(Chunk(file_name=file_path.name, chunk_index=i + 1, content=chunk_content, offset=offset))

    return chunks   

def main():
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    print(f"Processing folder: {DOCS_DIR}")

    all_chunks = []
    for file_path in DOCS_DIR.glob("*.md"):
        chunks = chunk_file(file_path)
        all_chunks.extend(chunks)

    for chunk in all_chunks:
        print(f"File: {chunk.file_name}, Chunk Index: {chunk.chunk_index}, Offset: {chunk.offset}")
        print("content omitted for brevity")
        print("-" * 40)

    vector_store.build_index(all_chunks)

    results = retrieval.retrieve("show me my project plan to become an AI engineer", k=3, min_score=0.45)
    if not results:
        print("No relevant document found for this query.")
    else:
        for r in results:
            print(f"[{r['score']:.4f}] {r['file_name']}, lines {r['offset']+1}-{r['offset']+20} (chunk {r['chunk_index']}): {r['content'][:200]}")

if __name__ == "__main__":
    main()