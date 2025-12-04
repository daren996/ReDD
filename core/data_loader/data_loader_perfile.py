# data_loader/data_loader_perfile.py
""" 
Concrete implementation of :class:`DataLoaderBase` for datasets where each
document is stored in a separate file.

This loader reads documents directly from individual files (not chunked).
For chunked documents stored in SQLite, use :class:`DataLoaderSQLite`.

Folder layout
-------------
```
<data_root>/
    path_to_docs/                  # folder containing document files
        doc1.txt
        doc2.txt
        ...
    queries.json                # mapping qid -> dict{query, attributes, sql}
    schema_general.json         # general schema
    schema_query_<qid>.json     # query-specific schemas
    doc_info.json               # optional: mapping doc_id -> dict{doc, data, metadata}
```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from .data_loader_basic import DataLoaderBase

__all__ = ["DataLoaderPerFile"]


class DataLoaderPerFile(DataLoaderBase):
    """Dataset loader for per-file document storage (un-chunked documents)."""

    DEFAULT_FILEMAP = {
        "documents": "documents",           # folder containing document files
        "doc_info": "doc_info.json",        # optional doc info
        "queries": "queries.json",
        "schema_general": "schema_general.json",
        "schema_query": "schema_query_{qid}.json",
    }

    def __init__(
        self, 
        data_root: str | Path, 
        *, 
        filemap: Dict[str, str] | None = None,
        file_extension: str = ".txt",
        encoding: str = "utf-8"
    ):
        """
        Initialize the per-file dataset loader.
        
        Args:
            data_root: Root directory of the dataset
            filemap: Custom file mapping (optional)
            file_extension: File extension for document files (default: .txt)
            encoding: File encoding (default: utf-8)
        """
        super().__init__(data_root)
        self._filemap = {**self.DEFAULT_FILEMAP, **(filemap or {})}
        self._file_extension = file_extension
        self._encoding = encoding

        # Initialize file mode
        self._init_file_mode()

        # Load common data
        self._doc_info = self._read_json(self._path("doc_info"), self._encoding)
        self._query_dict = self._read_json(self._path("queries"), self._encoding)
        self._schema_general = self._read_json(self._path("schema_general"), self._encoding)

    def _init_file_mode(self):
        """Initialize for reading from individual files."""
        doc_folder = self._path("documents")
        if not doc_folder.exists():
            logging.warning(f"[{self.__class__.__name__}:_init_file_mode] Document folder not found: {doc_folder}")
            self._doc_files = {}
            self._doc_ids = []
            return

        # Scan for document files
        self._doc_files = {}
        pattern = f"*{self._file_extension}"
        for file_path in sorted(doc_folder.glob(pattern)):
            # Use filename (without extension) as doc_id
            doc_id = file_path.stem
            self._doc_files[doc_id] = file_path

        self._doc_ids = list(self._doc_files.keys())
        logging.info(f"[{self.__class__.__name__}:_init_file_mode] Found {len(self._doc_ids)} documents in {doc_folder}")

    # ---------------- document access ---------------------------------

    @property
    def num_docs(self) -> int:
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> List[str]:
        return self._doc_ids

    def iter_docs(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Yield (doc_text, doc_id, metadata_dict) tuples."""
        for doc_id in self._doc_ids:
            yield self.get_doc(doc_id)

    def get_doc(self, doc_id: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Return (doc_text, doc_id, metadata_dict) for the given doc_id.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple of (doc_text, doc_id, metadata_dict)
        """
        if doc_id not in self._doc_files:
            logging.warning(f"[{self.__class__.__name__}:get_doc] Document {doc_id} not found")
            return ("", doc_id, {})

        file_path = self._doc_files[doc_id]
        try:
            with open(file_path, "r", encoding=self._encoding) as f:
                doc_text = f.read()
            
            # Build metadata dictionary
            metadata = {
                "source_file": file_path.name,
                "table_name": file_path.stem,
                "file_path": str(file_path),
                "encoding": self._encoding,
            }
            return (doc_text, doc_id, metadata)
        
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:get_doc] Error reading file {file_path}: {e}")
            return ("", doc_id, {"error": str(e)})

    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Return document info for doc_id.
        
        If doc_info.json doesn't have the entry, construct basic info from file.
        """
        # First try doc_info.json
        if doc_id in self._doc_info:
            return self._doc_info[doc_id]

        # Construct basic info from file
        doc_text, doc_id_ret, metadata = self.get_doc(doc_id)
        result = {
            "doc": doc_text,
            "doc_id": doc_id,
            "data": {},
        }
        result.update(metadata)
        return result

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
        """Re-read data from disk (hot-reload support)."""
        self._init_file_mode()
        self._doc_info = self._read_json(self._path("doc_info"), self._encoding)
        self._query_dict = self._read_json(self._path("queries"), self._encoding)
        self._schema_general = self._read_json(self._path("schema_general"), self._encoding)
