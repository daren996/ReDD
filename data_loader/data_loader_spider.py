# data_loader/data_loader_spider.py
""" 
Concrete implementation of :class:`BaseDatasetLoader` for Spider-style 
experimental datasets. 

Folder layout expected by default
---------------------------------
```
<dataset_root>/
    doc_dict.json          # mapping doc_id -> tuple(doc, fn, raw_id)
    doc_indo.json          # mapping doc_id -> dict{doc, fn, data}
    queries_drc.json       # mapping qid -> dict{query, attributes, sql}
    schema_general.json
    schema_query_<qid>.json
```
If your filenames differ, pass a custom *filemap* dict when instantiating.

The loader is eager - small JSON files are read fully into memory at
construction time. If memory is a concern you can switch to a streaming
implementation later (e.g. by inheriting from ``BaseDatasetLoader`` and
only loading what you need).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from .data_loader_basic import BaseDatasetLoader

__all__ = ["SpiderDatasetLoader"]


class SpiderDatasetLoader(BaseDatasetLoader):
    """Dataset loader for the canonical *doc_dict.json* format."""

    DEFAULT_FILEMAP = {
        "doc_dict": "doc_dict.json",
        "doc_info": "doc_info.json",
        "queries": "queries_drc.json",
        "schema_general": "schema_general.json",
        "schema_query": "schema_query_{qid}.json",
    }

    def __init__(self, dataset_root: str | Path, *, filemap: Dict[str, str] | None = None):
        super().__init__(dataset_root)
        self._filemap = {**self.DEFAULT_FILEMAP, **(filemap or {})}

        # eager read
        self._doc_dict = self._read_json(self._path("doc_dict"))
        self._doc_info = self._read_json(self._path("doc_info"))
        self._query_dict = self._read_json(self._path("queries"))
        self._schema_general = self._read_json(self._path("schema_general"))

        # pre-compute cheap metadata
        self._doc_ids = list(self._doc_dict.keys())

    def _path(self, key: str, **fmt) -> Path:
        """Return *absolute* path for logical resource *key*."""
        pattern = self._filemap[key]
        path = self.dataset_root / pattern.format(**fmt)
        return path

    # ---------------- document access ---------------------------------

    @property
    def num_docs(self) -> int:  # noqa: D401 - property fine
        return len(self._doc_dict)

    @property
    def doc_ids(self) -> List[str]:
        return self._doc_ids

    def iter_docs(self) -> Iterator[Tuple[str, str, str]]:
        for doc_id in self._doc_dict:
            yield self._doc_dict[doc_id]

    def get_doc(self, doc_id: str) -> Tuple[str, str, str]:
        return self._doc_dict[doc_id]

    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self._doc_info:
            return None
        return self._doc_info[doc_id]

    # ---------------- query / schema access ---------------------------

    @property
    def num_queries(self) -> int:
        return len(self._query_dict)

    def load_query_dict(self) -> Dict[str, Any]:
        return self._query_dict

    def load_schema_general(self) -> List[Dict[str, Any]]:
        return self._schema_general

    def load_schema_query(self, qid: str | int) -> List[Dict[str, Any]]:
        path = self._path("schema_query", qid=qid)
        return self._read_json(path)

    # ---------- helpers --------------------------------------------------

    @staticmethod
    def _read_json(path: Path):
        if not path.exists():
            # raise FileNotFoundError(path)
            return {}
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    # ---------------- hot-reload support -------------------------------

    def refresh(self) -> None:
        """Re-read JSON files from disk (hot-reload support)."""
        self._doc_dict = self._read_json(self._path("doc_dict"))
        self._doc_info = self._read_json(self._path("doc_info"))
        self._query_dict = self._read_json(self._path("queries"))
        self._schema_general = self._read_json(self._path("schema_general"))
        self._doc_ids = list(self._doc_dict.keys())

