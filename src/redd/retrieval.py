from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalMatch:
    item_id: str
    score: float


@dataclass
class RetrievalIndex:
    ids: list[str]
    embeddings: np.ndarray
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)
    backend: str = "auto"
    _faiss_index: Any | None = field(default=None, init=False, repr=False)

    def search(self, query_embedding: Sequence[float], top_k: int = 10) -> list[RetrievalMatch]:
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        if self.backend in {"auto", "faiss"}:
            matches = self._search_faiss(query, query_norm, top_k)
            if matches is not None:
                return matches
            if self.backend == "faiss":
                raise RuntimeError("FAISS retrieval backend requested but `faiss-cpu` is not installed.")

        docs = self.embeddings
        doc_norms = np.linalg.norm(docs, axis=1)
        denom = doc_norms * query_norm
        safe_scores = np.divide(docs @ query, denom, out=np.zeros_like(doc_norms), where=denom != 0)

        ranked_indices = np.argsort(-safe_scores)[:top_k]
        return [
            RetrievalMatch(item_id=self.ids[index], score=float(safe_scores[index]))
            for index in ranked_indices
        ]

    def _search_faiss(
        self,
        query: np.ndarray,
        query_norm: float,
        top_k: int,
    ) -> list[RetrievalMatch] | None:
        try:
            import faiss
        except ModuleNotFoundError:
            return None

        if self._faiss_index is None:
            docs = np.array(self.embeddings, dtype=np.float32, copy=True)
            doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
            np.divide(docs, doc_norms, out=docs, where=doc_norms != 0)
            index = faiss.IndexFlatIP(docs.shape[1])
            index.add(docs)
            self._faiss_index = index

        normalized_query = (query / query_norm).reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss_index.search(normalized_query, min(top_k, len(self.ids)))
        return [
            RetrievalMatch(item_id=self.ids[int(index)], score=float(score))
            for score, index in zip(scores[0], indices[0])
            if int(index) >= 0
        ]

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stored_metadata = dict(self.metadata)
        stored_metadata.setdefault("backend", self.backend)
        np.savez_compressed(
            output_path,
            ids=np.array(self.ids),
            embeddings=self.embeddings,
            model=np.array([self.model]),
            metadata=np.array([json.dumps(stored_metadata)]),
        )
        return output_path


def build_retrieval_index(
    embeddings: Mapping[str, Sequence[float]],
    *,
    model: str,
    metadata: Mapping[str, Any] | None = None,
    backend: str = "auto",
) -> RetrievalIndex:
    if backend not in {"auto", "numpy", "faiss"}:
        raise ValueError("Retrieval backend must be one of: auto, numpy, faiss")
    ids = list(embeddings.keys())
    matrix = np.array([embeddings[item_id] for item_id in ids], dtype=np.float32)
    return RetrievalIndex(
        ids=ids,
        embeddings=matrix,
        model=model,
        metadata=dict(metadata or {}),
        backend=backend,
    )


def load_retrieval_index(path: str | Path) -> RetrievalIndex:
    archive = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(archive["metadata"][0]))
    return RetrievalIndex(
        ids=[str(item_id) for item_id in archive["ids"].tolist()],
        embeddings=np.array(archive["embeddings"], dtype=np.float32),
        model=str(archive["model"][0]),
        metadata=metadata,
        backend=str(metadata.get("backend", "auto")),
    )


__all__ = [
    "RetrievalIndex",
    "RetrievalMatch",
    "build_retrieval_index",
    "load_retrieval_index",
]
