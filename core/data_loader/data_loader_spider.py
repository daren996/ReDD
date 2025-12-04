# data_loader/data_loader_spider.py
""" 
Concrete implementation of :class:`DataLoaderBase` for Spider-style 
experimental datasets. 

Folder layout expected by default
---------------------------------
```
<data_root>/
    doc_dict.json          # mapping doc_id -> tuple(doc, fn, raw_id)
    doc_indo.json          # mapping doc_id -> dict{doc, fn, data}
    queries.json       # mapping qid -> dict{query, attributes, sql}
    schema_general.json
    schema_query_<qid>.json
```
If your filenames differ, pass a custom *filemap* dict when instantiating.

The loader is eager - small JSON files are read fully into memory at
construction time. If memory is a concern you can switch to a streaming
implementation later (e.g. by inheriting from ``DataLoaderBase`` and
only loading what you need).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from .data_loader_basic import DataLoaderBase

__all__ = ["DataLoaderSpider"]


class DataLoaderSpider(DataLoaderBase):
    """Dataset loader for the canonical *doc_dict.json* format."""
    # TODO: give unique id/name to each document

    DEFAULT_FILEMAP = {
        "doc_dict": "doc_dict.json",
        "doc_info": "doc_info.json",
        "queries": "queries.json",
        "schema_general": "schema_general.json",
        "schema_query": "schema_query_{qid}.json",
    }

    def __init__(
        self, 
        data_root: str | Path, 
        *, 
        filemap: Dict[str, str] | None = None,
        encoding: str = "utf-8"
    ):
        super().__init__(data_root)
        self._filemap = {**self.DEFAULT_FILEMAP, **(filemap or {})}
        self._encoding = encoding

        # eager read
        self._doc_dict = self._read_json(self._path("doc_dict"), self._encoding)
        self._doc_info = self._read_json(self._path("doc_info"), self._encoding)
        self._query_dict = self._read_json(self._path("queries"), self._encoding)
        self._schema_general = self._read_json(self._path("schema_general"), self._encoding)

        # pre-compute cheap metadata
        self._doc_ids = list(self._doc_dict.keys())

    # ---------------- document access ---------------------------------

    @property
    def num_docs(self) -> int:  # noqa: D401 - property fine
        return len(self._doc_dict)

    @property
    def doc_ids(self) -> List[str]:
        return self._doc_ids

    def iter_docs(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        for doc_id in self._doc_dict:
            doc_text, table_name, raw_id = self._doc_dict[doc_id]
            metadata = {
                "table_name": table_name,
                "raw_id": raw_id,
            }
            yield (doc_text, doc_id, metadata)

    def get_doc(self, doc_id: str) -> Tuple[str, str, Dict[str, Any]]:
        doc_text, table_name, raw_id = self._doc_dict[doc_id]
        metadata = {
            "table_name": table_name,
            "raw_id": raw_id,
        }
        return (doc_text, doc_id, metadata)

    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self._doc_info:
            return None
        return self._doc_info[doc_id]

    # ---------------- query / schema access ---------------------------

    @property
    def num_queries(self) -> int:
        return len(self._query_dict)

    @property
    def query_ids(self) -> List[str]:
        """List of all query IDs."""
        return list(self._query_dict.keys())

    def load_query_dict(self) -> Dict[str, Any]:
        return self._query_dict

    def load_schema_general(self) -> List[Dict[str, Any]]:
        return self._schema_general

    def load_schema_query(self, qid: str | int) -> List[Dict[str, Any]]:
        path = self._path("schema_query", qid=qid)
        return self._read_json(path, self._encoding)

    # ---------------- hot-reload support -------------------------------

    def refresh(self) -> None:
        """Re-read JSON files from disk (hot-reload support)."""
        self._doc_dict = self._read_json(self._path("doc_dict"), self._encoding)
        self._doc_info = self._read_json(self._path("doc_info"), self._encoding)
        self._query_dict = self._read_json(self._path("queries"), self._encoding)
        self._schema_general = self._read_json(self._path("schema_general"), self._encoding)
        self._doc_ids = list(self._doc_dict.keys())

