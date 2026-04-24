"""Embedding-driven document clustering helpers."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    KMeans = None  # type: ignore[assignment]

from redd.exceptions import RuntimeDependencyError
from .providers import get_embedding_provider


def llm_embeddings(
    string: str,
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
) -> list[float]:
    """Generate a single embedding vector through the shared embedding provider."""
    embedder = get_embedding_provider(
        model=model,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
    )
    return embedder.embed_single(string)


class Vectorizer:
    def fit_transform(self, documents: Sequence[str]):
        raise NotImplementedError

    def transform(self, documents: Sequence[str]):
        raise NotImplementedError


class DocVectorizer(Vectorizer):
    def __init__(
        self,
        embedder: Callable[[str], list[float]] | None = None,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.embedder = embedder or (
            lambda text: llm_embeddings(
                text,
                model=model,
                api_key=api_key,
                provider=provider,
                base_url=base_url,
            )
        )

    def fit_transform(self, documents: Sequence[str]) -> list[list[float]]:
        return [self.embedder(document) for document in documents]

    def transform(self, documents: Sequence[str]) -> list[list[float]]:
        return self.fit_transform(documents)


class Clusterer:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def fit(self, embeddings: Sequence[Sequence[float]]):
        raise NotImplementedError

    def predict(self, embeddings: Sequence[Sequence[float]]):
        raise NotImplementedError


class ClustererKMeans(Clusterer):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters)
        if KMeans is None:
            raise RuntimeDependencyError(
                "Document clustering requires `scikit-learn`. "
                "Install the optional clustering dependencies before using redd.embedding.DocumentClustering."
            )
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit_predict(self, embeddings: Sequence[Sequence[float]]):
        if not isinstance(embeddings, (list, tuple)):
            raise ValueError("Embeddings must be a list or 2D array-like structure.")
        return self.model.fit_predict(embeddings)


ClustererGPT = ClustererKMeans


class DocumentClustering:
    def __init__(
        self,
        documents: Sequence[str],
        n_clusters: int,
        vectorizer: DocVectorizer | None = None,
        clusterer: ClustererKMeans | None = None,
    ):
        self.documents = list(documents)
        self.vectorizer = vectorizer or DocVectorizer()
        self.clusterer = clusterer or ClustererKMeans(n_clusters)
        self.vectors: list[list[float]] = []
        self.cluster_ids: list[int] = []

    def cluster(self):
        self.vectors = self.vectorizer.fit_transform(self.documents)
        self.cluster_ids = list(self.clusterer.fit_predict(self.vectors))
        return self.cluster_ids

    def get_clustered_documents(self):
        clustered_documents: dict[int, list[str]] = {}
        for cluster_id, document in zip(self.cluster_ids, self.documents):
            clustered_documents.setdefault(int(cluster_id), []).append(document)
        return clustered_documents
