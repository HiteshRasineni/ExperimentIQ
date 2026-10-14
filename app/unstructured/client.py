"""Qdrant vector database client wrapper.

Provides a clean interface to Qdrant for storing and
retrieving unstructured documents (logs, configs, notes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing import Any, Dict, List, Optional


class QdrantClientWrapper:
    """Wrapper around Qdrant client with in-memory fallback for local dev."""

    def __init__(self, url: Optional[str] = None, port: int = 6333, collection_name: str = "experimentiq"):
        self.collection_name = collection_name
        self._docs: List[Dict[str, Any]] = []
        self._use_qdrant = False
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._Distance = Distance
            self._VectorParams = VectorParams
            host = url or "localhost"
            self.client = QdrantClient(host=host, port=port, timeout=3)
            self.client.get_collections()
            self._use_qdrant = True
            self._ensure_collection()
        except Exception:
            self.client = None
            self._use_qdrant = False

    def _ensure_collection(self):
        if not self._use_qdrant:
            return
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._VectorParams(size=384, distance=self._Distance.COSINE),
            )

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Add documents to the Qdrant collection (or in-memory fallback)."""
        if not documents:
            return 0
        if not self._use_qdrant:
            self._docs.extend(documents)
            return len(documents)
        from qdrant_client.models import PointStruct
        import uuid
        try:
            from .retrieve import EmbeddingGenerator
            _emb = EmbeddingGenerator()
        except Exception:
            _emb = None
        points = []
        for doc in documents:
            vec = doc.get("vector")
            if vec is None and _emb is not None:
                try:
                    vec = _emb.generate(str(doc.get("text", "")))
                except Exception:
                    vec = None
            if vec is None:
                continue
            pid = doc.get("id") or str(uuid.uuid4())
            try:
                uuid.UUID(str(pid))
            except Exception:
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pid)))
            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {}),
                        "type": doc.get("type", "document"),
                        "title": doc.get("title", ""),
                    },
                )
            )
        if not points:
            return 0
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query for similar documents to the given text."""
        if not self._use_qdrant:
            # keyword-overlap fallback over in-memory docs
            q = query_text.lower().split()
            scored = []
            for d in self._docs:
                txt = str(d.get("text", "")).lower()
                score = sum(1 for w in q if w in txt) / max(1, len(q))
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            out = []
            for score, d in scored[:top_k]:
                out.append({
                    "id": str(d.get("id", "")),
                    "text": d.get("text", ""),
                    "score": float(score),
                    "metadata": d.get("metadata", {}),
                    "type": d.get("type", "document"),
                    "title": d.get("title", ""),
                })
            return out
        from .retrieve import EmbeddingGenerator
        try:
            qvec = EmbeddingGenerator().generate(query_text)
        except Exception:
            qvec = None
        if not qvec:
            return []
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
        results = []
        pts = getattr(hits, "points", hits) or []
        for hit in pts:
            payload = hit.payload or {}
            results.append({
                "id": str(hit.id),
                "text": payload.get("text", ""),
                "score": float(hit.score or 0),
                "metadata": payload.get("metadata", {}),
                "type": payload.get("type", "document"),
                "title": payload.get("title", ""),
            })
        return results

    def delete_by_id(self, doc_id: str):
        """Delete a document by its ID."""
        if not self._use_qdrant:
            self._docs = [d for d in self._docs if str(d.get("id")) != str(doc_id)]
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[doc_id],
        )

    def count(self) -> int:
        """Get the number of documents in the collection."""
        if not self._use_qdrant:
            return len(self._docs)
        return self.client.count(collection_name=self.collection_name).count
