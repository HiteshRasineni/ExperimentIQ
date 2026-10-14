"""Document ingestion and retrieval modules for ExperimentIQ.

Handles document parsing, chunking, embedding, and
semantic retrieval for unstructured content.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


class Document:
    """Represents a single document in the retrieval system."""

    def __init__(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: str = "document",
        title: str = "",
    ):
        self.id = id
        self.text = text
        self.metadata = metadata or {}
        self.type = doc_type
        self.title = title
        self.hash = hashlib.sha256(text.encode()).hexdigest()[:16]


class SimpleChunker:
    """Splits documents into chunks for retrieval."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, title: str = "") -> List[Dict[str, str]]:
        """Split text into chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append({
                "title": title or "Untitled",
                "chunk": chunk,
            })
            start = end - self.overlap

        return chunks


class EmbeddingGenerator:
    """Generates embeddings for text using a sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._loaded = True
        except ImportError:
            self._loaded = False
            self.model = None

    def generate(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        if not self._loaded:
            # Return a simple hash-based embedding as fallback
            return self._hash_embedding(text)
        return self.model.encode(text).tolist()

    def _hash_embedding(self, text: str) -> List[float]:
        """Generate a deterministic pseudo-embedding using hashing."""
        import random
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return [((hash_val >> i) & 1) * 0.1 for i in range(384)]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Utility function to chunk text into smaller pieces."""
    chunks = []
    if len(text) <= chunk_size:
        return [text]

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
