# data_loader/data_loader_basic.py
"""Base *interface* for dataset loaders.

This file contains **no concrete I/O logic** - it only specifies *what* a
loader must provide so that the rest of the pipeline can work with any
back-end implementation (JSON eager read, streaming folder read, HTML
scraping, etc.).

Expected artefacts
------------------

**Documents** - the iterator exposes tuples ``(<doc_text>, <table_name>, 
                <doc_id>)`` in the order of <doc_id>.

**Queries**  - dict keyed by <qid> (in str format):
    {
        "<qid>": {
            "query": "...",
            "attributes": [...],
            "sql": "..."
        },
        ...
    }

**Schemas** - two granularities:
    *General*: list of tables, each in the following format:
        {
            "Schema Name": "<schema_name>",
            "Attributes": [
                {
                    "Attribute Name": "<attribute_name>", 
                    "Description": "<attribute_description>",
                },
                ...
            ]
        }
    *Per-query*: subset or projection relevant to a single query

**Document Info** - each <doc_id> resolves to a record shaped like:
    {
        "doc": "<doc_text>",
        "fn":  "<table_name>",
        "data": {"attr": "value", ...}
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Iterator, Tuple

# ---------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------

class BaseDatasetLoader(ABC):
    """ A minimal contract every dataset reader must fulfil.
    Implementations may choose in-memory loading or lazy/streaming loading;
    The interface has been designed so that both are possible. """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(self.dataset_root)

    # ---------- document access ------------------------------------------

    # ----------------------------- docs ----------------------------------

    @property
    @abstractmethod
    def num_docs(self) -> int:
        """Total number of docs (cheap to compute)."""

    @property
    @abstractmethod
    def doc_ids(self) -> List[str]:
        """List of all doc ids (doc_id) in stable order."""

    @abstractmethod
    def iter_docs(self) -> Iterator[Tuple[str, str, str]]:
        """Yield ``[raw_doc_text, table_name, raw_doc_id]`` - streaming allowed; 
        must not assume all docs fit in memory. """

    @abstractmethod
    def get_doc(self, doc_id: str) -> Tuple[str, str, str]:
        """Return document for <doc_id>. May read from disk lazily."""

    @abstractmethod
    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        """Return document info for <doc_id> as a dict. May read from disk lazily."""

    # ----------------------------- queries & schema ----------------------

    @property
    @abstractmethod
    def num_queries(self) -> int:
        """Total number of queries (cheap to compute)."""

    @abstractmethod
    def load_query_dict(self) -> Dict[str, Any]:
        """Return the *entire* query mapping (see format above)."""

    @abstractmethod
    def load_schema_general(self) -> List[Dict[str, Any]]:
        """Dataset-wide table list with attribute metadata."""

    @abstractmethod
    def load_schema_query(self, qid: str | int) -> List[Dict[str, Any]]:
        """Query-specific schema (often a subset of the general one)."""

    # ---------- helpers --------------------------------------------------

    # ---------- syntactic sugar ------------------------------------------

    def __iter__(self):
        """By default iterate over documents (stream-friendly)."""
        return self.iter_docs()

    def __len__(self):
        return self.num_docs

    def __repr__(self):
        return f"<{self.__class__.__name__} root='{self.dataset_root}'>"
    
    def __str__(self):
        return f"<{self.__class__.__name__} root='{self.dataset_root}'>"
