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

    def search(self, query_embedding: Sequence[float], top_k: int = 10) -> list[RetrievalMatch]:
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        docs = self.embeddings
        doc_norms = np.linalg.norm(docs, axis=1)
        denom = doc_norms * query_norm
        safe_scores = np.divide(docs @ query, denom, out=np.zeros_like(doc_norms), where=denom != 0)

        ranked_indices = np.argsort(-safe_scores)[:top_k]
        return [
            RetrievalMatch(item_id=self.ids[index], score=float(safe_scores[index]))
            for index in ranked_indices
        ]

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            ids=np.array(self.ids),
            embeddings=self.embeddings,
            model=np.array([self.model]),
            metadata=np.array([json.dumps(self.metadata)]),
        )
        return output_path


def build_retrieval_index(
    embeddings: Mapping[str, Sequence[float]],
    *,
    model: str,
    metadata: Mapping[str, Any] | None = None,
) -> RetrievalIndex:
    ids = list(embeddings.keys())
    matrix = np.array([embeddings[item_id] for item_id in ids], dtype=np.float32)
    return RetrievalIndex(
        ids=ids,
        embeddings=matrix,
        model=model,
        metadata=dict(metadata or {}),
    )


def load_retrieval_index(path: str | Path) -> RetrievalIndex:
    archive = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(archive["metadata"][0]))
    return RetrievalIndex(
        ids=[str(item_id) for item_id in archive["ids"].tolist()],
        embeddings=np.array(archive["embeddings"], dtype=np.float32),
        model=str(archive["model"][0]),
        metadata=metadata,
    )


__all__ = [
    "RetrievalIndex",
    "RetrievalMatch",
    "build_retrieval_index",
    "load_retrieval_index",
]
