from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import EmbeddingProviderBase
from .providers import detect_provider, get_embedding_provider

if TYPE_CHECKING:
    from redd.core.data_loader import DataLoaderBase


class EmbeddingManager:
    """Persist document and query embeddings in a SQLite cache."""

    DOC_EMBEDDINGS_TABLE = "doc_embeddings"
    QUERY_EMBEDDINGS_TABLE = "query_embeddings"

    def __init__(
        self,
        storage_path: str | Path | None = None,
        *,
        loader: "DataLoaderBase | Any" | None = None,
        dataset_db_path: str | Path | None = None,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        auto_save: bool = True,
        **provider_kwargs: Any,
    ) -> None:
        self.storage_path = self._resolve_storage_path(
            storage_path=storage_path,
            loader=loader,
            dataset_db_path=dataset_db_path,
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.api_key = api_key
        self.auto_save = auto_save
        self.provider_kwargs = provider_kwargs
        self.provider_name = str(provider_kwargs.get("provider") or detect_provider(model)).lower()
        self._provider: EmbeddingProviderBase | None = None

        self._init_tables()

    @staticmethod
    def _resolve_storage_path(
        *,
        storage_path: str | Path | None,
        loader: "DataLoaderBase | Any" | None,
        dataset_db_path: str | Path | None,
    ) -> Path:
        if storage_path is not None:
            return Path(storage_path).expanduser().resolve()

        if dataset_db_path is not None:
            anchor = Path(dataset_db_path).expanduser()
        elif loader is not None and getattr(loader, "data_root", None) is not None:
            anchor = Path(loader.data_root).expanduser()
        else:
            raise ValueError(
                "EmbeddingManager requires `storage_path=`, `dataset_db_path=`, or `loader=`."
            )

        if anchor.is_dir():
            return (anchor / "embeddings.sqlite3").resolve()

        if anchor.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            return anchor.with_name(f"{anchor.stem}.embeddings.sqlite3").resolve()

        return (anchor / "embeddings.sqlite3").resolve()

    @property
    def provider(self) -> EmbeddingProviderBase:
        if self._provider is None:
            self._provider = get_embedding_provider(
                model=self.model,
                api_key=self.api_key,
                provider=self.provider_name,
                **self.provider_kwargs,
            )
        return self._provider

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.storage_path)

    def _init_tables(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.DOC_EMBEDDINGS_TABLE} (
                doc_id TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                PRIMARY KEY (doc_id, model, provider)
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.QUERY_EMBEDDINGS_TABLE} (
                query_id TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                query_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                PRIMARY KEY (query_id, model, provider)
            )
            """
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> bytes:
        return np.array(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(raw: bytes) -> list[float]:
        return np.frombuffer(raw, dtype=np.float32).tolist()

    @staticmethod
    def _load_doc_text(loader: "DataLoaderBase | Any", doc_id: str) -> str:
        if hasattr(loader, "get_doc"):
            doc_text, _resolved_doc_id, _metadata = loader.get_doc(doc_id)
            return str(doc_text)
        if hasattr(loader, "get_doc_text"):
            return str(loader.get_doc_text(doc_id))
        raise ValueError("Loader must provide `get_doc()` or `get_doc_text()`.")

    def load_doc_embeddings(self, doc_ids: list[str] | None = None) -> dict[str, list[float]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        if doc_ids is None:
            cursor.execute(
                f"""
                SELECT doc_id, embedding
                FROM {self.DOC_EMBEDDINGS_TABLE}
                WHERE model = ? AND provider = ?
                """,
                (self.model, self.provider_name),
            )
        else:
            placeholders = ",".join("?" * len(doc_ids))
            cursor.execute(
                f"""
                SELECT doc_id, embedding
                FROM {self.DOC_EMBEDDINGS_TABLE}
                WHERE model = ? AND provider = ? AND doc_id IN ({placeholders})
                """,
                [self.model, self.provider_name, *doc_ids],
            )
        rows = cursor.fetchall()
        conn.close()
        return {str(doc_id): self._deserialize_embedding(raw) for doc_id, raw in rows}

    def save_doc_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        if not embeddings:
            return
        conn = self._get_connection()
        cursor = conn.cursor()
        payload = [
            (
                str(doc_id),
                self.model,
                self.provider_name,
                self._serialize_embedding(embedding),
                len(embedding),
            )
            for doc_id, embedding in embeddings.items()
        ]
        cursor.executemany(
            f"""
            INSERT OR REPLACE INTO {self.DOC_EMBEDDINGS_TABLE}
            (doc_id, model, provider, embedding, embedding_dim)
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
        conn.close()

    def get_doc_embeddings(
        self,
        loader: "DataLoaderBase | Any",
        doc_ids: list[str] | None = None,
        *,
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        target_doc_ids = [str(doc_id) for doc_id in (doc_ids or loader.doc_ids)]
        cached = self.load_doc_embeddings(target_doc_ids)
        missing_doc_ids = [doc_id for doc_id in target_doc_ids if doc_id not in cached]

        if missing_doc_ids:
            generated: dict[str, list[float]] = {}
            for start in range(0, len(missing_doc_ids), batch_size):
                batch_ids = missing_doc_ids[start : start + batch_size]
                texts = [self._load_doc_text(loader, doc_id) for doc_id in batch_ids]
                embeddings = self.provider.embed_batch(texts, batch_size=len(batch_ids))
                for doc_id, embedding in zip(batch_ids, embeddings):
                    generated[doc_id] = embedding

            if self.auto_save:
                self.save_doc_embeddings(generated)
            cached.update(generated)

        return {doc_id: cached[doc_id] for doc_id in target_doc_ids if doc_id in cached}

    def load_query_embeddings(self, query_ids: list[str] | None = None) -> dict[str, list[float]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        if query_ids is None:
            cursor.execute(
                f"""
                SELECT query_id, embedding
                FROM {self.QUERY_EMBEDDINGS_TABLE}
                WHERE model = ? AND provider = ?
                """,
                (self.model, self.provider_name),
            )
        else:
            placeholders = ",".join("?" * len(query_ids))
            cursor.execute(
                f"""
                SELECT query_id, embedding
                FROM {self.QUERY_EMBEDDINGS_TABLE}
                WHERE model = ? AND provider = ? AND query_id IN ({placeholders})
                """,
                [self.model, self.provider_name, *query_ids],
            )
        rows = cursor.fetchall()
        conn.close()
        return {str(query_id): self._deserialize_embedding(raw) for query_id, raw in rows}

    def save_query_embeddings(self, query_embeddings: dict[str, tuple[str, list[float]]]) -> None:
        if not query_embeddings:
            return
        conn = self._get_connection()
        cursor = conn.cursor()
        payload = [
            (
                str(query_id),
                self.model,
                self.provider_name,
                query_text,
                self._serialize_embedding(embedding),
                len(embedding),
            )
            for query_id, (query_text, embedding) in query_embeddings.items()
        ]
        cursor.executemany(
            f"""
            INSERT OR REPLACE INTO {self.QUERY_EMBEDDINGS_TABLE}
            (query_id, model, provider, query_text, embedding, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
        conn.close()

    def get_query_embeddings(
        self,
        loader: "DataLoaderBase",
        query_ids: list[str] | None = None,
        *,
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        query_dict = loader.load_query_dict()
        target_query_ids = [str(query_id) for query_id in (query_ids or list(query_dict.keys()))]
        cached = self.load_query_embeddings(target_query_ids)
        missing_query_ids = [query_id for query_id in target_query_ids if query_id not in cached]

        if missing_query_ids:
            generated: dict[str, tuple[str, list[float]]] = {}
            for start in range(0, len(missing_query_ids), batch_size):
                batch_ids = missing_query_ids[start : start + batch_size]
                texts = [str(query_dict[query_id]["query"]) for query_id in batch_ids]
                embeddings = self.provider.embed_batch(texts, batch_size=len(batch_ids))
                for query_id, query_text, embedding in zip(batch_ids, texts, embeddings):
                    generated[query_id] = (query_text, embedding)

            if self.auto_save:
                self.save_query_embeddings(generated)
            cached.update({query_id: embedding for query_id, (_query_text, embedding) in generated.items()})

        return {query_id: cached[query_id] for query_id in target_query_ids if query_id in cached}

    def get_query_embedding(self, query_id: str, query_text: str) -> list[float]:
        cached = self.load_query_embeddings([str(query_id)])
        if str(query_id) in cached:
            return cached[str(query_id)]

        embedding = self.provider.embed_single(query_text)
        if self.auto_save:
            self.save_query_embeddings({str(query_id): (query_text, embedding)})
        return embedding

    def embed_single(self, text: str) -> list[float]:
        return self.provider.embed_single(text)

    @staticmethod
    def cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
        return EmbeddingProviderBase.cosine_similarity(emb1, emb2)
