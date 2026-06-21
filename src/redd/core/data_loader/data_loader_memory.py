"""In-memory data loader for programmatic extraction requests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .data_loader_basic import DataLoaderBase
from .structured import (
    coerce_json_payload,
    identity_name_map,
    normalize_query_records,
    query_to_legacy,
    schema_for_query,
    schema_to_legacy_or_passthrough,
)

__all__ = ["InMemoryDataLoader"]


class InMemoryDataLoader(DataLoaderBase):
    """Load extraction documents from Python dictionaries."""

    def __init__(
        self,
        data_root: str | Path = ".",
        *,
        documents: list[Mapping[str, Any]],
        schema_json: Any,
        query_json: Any | None = None,
        doc_id_column: str = "doc_id",
        doc_text_column: str = "doc_text",
        dataset_id: str = "memory",
        **_: Any,
    ) -> None:
        super().__init__(data_root)
        self.dataset_id = dataset_id
        self.doc_id_column = doc_id_column
        self.doc_text_column = doc_text_column
        self._schema_contract = coerce_json_payload(schema_json, default={})
        self._schema_general = schema_to_legacy_or_passthrough(self._schema_contract)
        self._query_records = normalize_query_records(
            query_json,
            schema_general=self._schema_general,
            dataset_id=dataset_id,
        )
        self._query_dict = {
            str(record["query_id"]): query_to_legacy(record, self._schema_general)
            for record in self._query_records
        }
        self._documents = self._normalize_documents(documents)
        self._doc_ids = list(self._documents)

    @property
    def num_docs(self) -> int:
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> List[str]:
        return list(self._doc_ids)

    def iter_docs(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        for doc_id in self._doc_ids:
            yield self.get_doc(doc_id)

    def get_doc(self, doc_id: str) -> Tuple[str, str, Dict[str, Any]]:
        item = self._documents.get(str(doc_id))
        if item is None:
            return ("", str(doc_id), {})
        return (item["doc_text"], str(doc_id), dict(item["metadata"]))

    def get_doc_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        doc_text, resolved_doc_id, metadata = self.get_doc(doc_id)
        if not doc_text and not metadata:
            return None
        return {
            "doc": doc_text,
            "fn": metadata.get("source_table") or metadata.get("table_name") or resolved_doc_id,
            "doc_id": resolved_doc_id,
            "parent_doc_id": metadata.get("parent_doc_id"),
            "chunk_index": metadata.get("chunk_index"),
            "source_row_id": metadata.get("source_row_id"),
            "data_records": [],
            "data": {},
        }

    @property
    def num_queries(self) -> int:
        return len(self._query_dict)

    @property
    def query_ids(self) -> List[str]:
        return list(self._query_dict)

    def load_query_dict(self) -> Dict[str, Any]:
        return dict(self._query_dict)

    def get_query_info(self, qid: str | int) -> Optional[Dict[str, Any]]:
        return self._query_dict.get(str(qid))

    def load_schema_general(self) -> List[Dict[str, Any]]:
        return list(self._schema_general)

    def load_schema_query(self, qid: str | int) -> List[Dict[str, Any]]:
        return schema_for_query(self._schema_general, self._query_dict.get(str(qid)))

    def load_name_map(self, query_id: str | int | None = None, **_: Any) -> Dict[str, Any]:
        del query_id
        return identity_name_map(self._schema_general)

    def _normalize_documents(
        self,
        documents: list[Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        for document in documents:
            doc_id = str(document.get(self.doc_id_column) or "").strip()
            if not doc_id:
                raise ValueError("InMemoryDataLoader documents must include non-empty doc_id.")
            if doc_id in normalized:
                raise ValueError(f"InMemoryDataLoader duplicate doc_id: {doc_id}")
            doc_text = document.get(self.doc_text_column)
            metadata = {
                key: value
                for key, value in document.items()
                if key not in {self.doc_id_column, self.doc_text_column}
            }
            metadata.setdefault("source", "memory")
            normalized[doc_id] = {
                "doc_text": "" if doc_text is None else str(doc_text),
                "metadata": metadata,
            }
        return normalized
