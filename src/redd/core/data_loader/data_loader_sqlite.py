# data_loader/data_loader_sqlite.py
""" 
Concrete implementation of :class:`DataLoaderBase` for datasets stored
in SQLite database (typically chunked documents).

This loader reads documents from a SQLite database containing chunked documents.
For reading original un-chunked files, use :class:`DataLoaderPerFile`.

Folder layout
-------------
```
<data_root>/
    documents.db                # SQLite database with chunked documents
    queries.json                # mapping qid -> dict{query, attributes, sql}
    schema_general.json         # general schema
    schema_query_<qid>.json     # query-specific schemas
    doc_info.json               # optional: mapping doc_id -> dict{doc, fn, data}
```

SQLite database schema
----------------------
Table: documents
    - doc_id TEXT PRIMARY KEY       # unique document id (e.g., "doc-1-1", "doc-1-2")
    - doc_text TEXT                 # document content
    - source_file TEXT              # source file name
    - parent_doc_id TEXT            # original doc id before chunking (e.g., "doc-1")
    - chunk_index INTEGER           # chunk sequence number (0-based)
    - created_at TIMESTAMP          # creation timestamp
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Optional

from .data_loader_basic import DataLoaderBase

__all__ = ["DataLoaderSQLite"]


class DataLoaderSQLite(DataLoaderBase):
    """Dataset loader for SQLite database (chunked documents)."""

    DEFAULT_FILEMAP = {
        "sqlite_db": "documents.db",        # SQLite database for chunked docs
        "doc_info": "doc_info.json",        # optional doc info
        "queries": "queries.json",
        "schema_general": "schema_general.json",
        "schema_query": "schema_query_{qid}.json",
    }

    def __init__(
        self, 
        data_root: str | Path, 
        *, 
        filemap: Dict[str, str] | None = None
    ):
        """
        Initialize the SQLite dataset loader.
        
        Args:
            data_root: Root directory of the dataset
            filemap: Custom file mapping (optional)
        """
        super().__init__(data_root)
        self._filemap = {**self.DEFAULT_FILEMAP, **(filemap or {})}

        # Initialize SQLite mode
        self._init_sqlite_mode()

        # Load common data
        self._doc_info = self._read_json(self._path("doc_info"), self._encoding)
        self._query_dict = self._read_json(self._path("queries"), self._encoding)
        self._schema_general = self._read_json(self._path("schema_general"), self._encoding)

    def _init_sqlite_mode(self):
        """Initialize for reading from SQLite database."""
        db_path = self._path("sqlite_db")
        if not db_path.exists():
            logging.warning(
                f"[{self.__class__.__name__}:_init_sqlite_mode] "
                f"SQLite database not found: {db_path}"
            )
            self._db_path = None
            self._doc_ids = []
            return

        self._db_path = db_path
        
        # Load doc_ids from database
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT doc_id FROM documents ORDER BY doc_id")
                self._doc_ids = [row[0] for row in cursor.fetchall()]
            
            logging.info(
                f"[{self.__class__.__name__}:_init_sqlite_mode] "
                f"Found {len(self._doc_ids)} documents in SQLite database"
            )
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:_init_sqlite_mode] "
                f"Error reading from SQLite database: {e}"
            )
            self._doc_ids = []

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
        if self._db_path is None:
            logging.warning(
                f"[{self.__class__.__name__}:get_doc] "
                f"SQLite database not initialized"
            )
            return ("", doc_id, {})

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT doc_text, source_file, parent_doc_id, chunk_index FROM documents WHERE doc_id = ?",
                    (doc_id,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    logging.warning(
                        f"[{self.__class__.__name__}:get_doc] "
                        f"Document {doc_id} not found in database"
                    )
                    return ("", doc_id, {})
                
                doc_text, source_file, parent_doc_id, chunk_index = row
                
                # Build metadata dictionary
                metadata = {
                    "source_file": source_file,
                    "table_name": Path(source_file).stem if source_file else doc_id,
                    "parent_doc_id": parent_doc_id,
                    "chunk_index": chunk_index,
                    "is_chunked": parent_doc_id is not None,
                }
                return (doc_text, doc_id, metadata)
        
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:get_doc] "
                f"Error reading from database: {e}"
            )
            return ("", doc_id, {"error": str(e)})

    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Return document info for doc_id.
        
        If doc_info.json doesn't have the entry, construct info from database metadata.
        """
        # First try doc_info.json
        if doc_id in self._doc_info:
            return self._doc_info[doc_id]

        # Construct from database
        if self._db_path:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT doc_text, source_file, parent_doc_id, chunk_index 
                        FROM documents 
                        WHERE doc_id = ?
                        """,
                        (doc_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        doc_text, source_file, parent_doc_id, chunk_index = row
                        return {
                            "doc": doc_text,
                            "fn": source_file or doc_id,
                            "doc_id": doc_id,
                            "parent_doc_id": parent_doc_id,
                            "chunk_index": chunk_index,
                            "data": {}  # No ground truth data by default
                        }
            except sqlite3.Error as e:
                logging.error(
                    f"[{self.__class__.__name__}:get_doc_info] "
                    f"Error reading from database: {e}"
                )

        return None

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

    # ---------------- SQLite-specific methods -------------------------

    def get_doc_chunks(self, parent_doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a parent document.
        
        Args:
            parent_doc_id: Original document ID before chunking
            
        Returns:
            List of chunk dictionaries with keys: doc_id, doc_text, chunk_index
        """
        if self._db_path is None:
            logging.warning(
                f"[{self.__class__.__name__}:get_doc_chunks] "
                f"SQLite database not initialized"
            )
            return []

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT doc_id, doc_text, chunk_index 
                    FROM documents 
                    WHERE parent_doc_id = ? 
                    ORDER BY chunk_index
                    """,
                    (parent_doc_id,)
                )
                
                chunks = []
                for row in cursor.fetchall():
                    chunks.append({
                        "doc_id": row[0],
                        "doc_text": row[1],
                        "chunk_index": row[2]
                    })
                
                return chunks
        
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:get_doc_chunks] "
                f"Error reading chunks from database: {e}"
            )
            return []

    def get_parent_docs(self) -> List[str]:
        """
        Get list of unique parent document IDs.
        
        Returns:
            List of parent document IDs
        """
        if self._db_path is None:
            logging.warning(
                f"[{self.__class__.__name__}:get_parent_docs] "
                f"SQLite database not initialized"
            )
            return []

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT parent_doc_id FROM documents WHERE parent_doc_id IS NOT NULL ORDER BY parent_doc_id"
                )
                return [row[0] for row in cursor.fetchall()]
        
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:get_parent_docs] "
                f"Error reading parent docs from database: {e}"
            )
            return []

    def get_source_files(self) -> List[str]:
        """
        Get list of unique source file names.
        
        Returns:
            List of source file names
        """
        if self._db_path is None:
            logging.warning(
                f"[{self.__class__.__name__}:get_source_files] "
                f"SQLite database not initialized"
            )
            return []

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT DISTINCT source_file FROM documents WHERE source_file IS NOT NULL ORDER BY source_file"
                )
                return [row[0] for row in cursor.fetchall()]
        
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:get_source_files] "
                f"Error reading source files from database: {e}"
            )
            return []

    # ---------------- hot-reload support -------------------------------

    def refresh(self) -> None:
        """Re-read data from database (hot-reload support)."""
        self._init_sqlite_mode()
        self._doc_info = self._read_json(self._path("doc_info"), self._encoding)
        self._query_dict = self._read_json(self._path("queries"), self._encoding)
        self._schema_general = self._read_json(self._path("schema_general"), self._encoding)

    # ---------------- utility methods ---------------------------------

    @staticmethod
    def create_sqlite_db(db_path: str | Path, overwrite: bool = False) -> None:
        """
        Create a new SQLite database with the documents table schema.
        
        Args:
            db_path: Path to the SQLite database file
            overwrite: If True, drop existing table; if False, raise error if exists
        """
        db_path = Path(db_path)
        
        if db_path.exists() and not overwrite:
            raise FileExistsError(
                f"Database already exists: {db_path}. Set overwrite=True to recreate."
            )

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Drop table if overwrite
                if overwrite:
                    cursor.execute("DROP TABLE IF EXISTS documents")
                
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT PRIMARY KEY,
                        doc_text TEXT NOT NULL,
                        source_file TEXT,
                        parent_doc_id TEXT,
                        chunk_index INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_parent_doc_id ON documents(parent_doc_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_source_file ON documents(source_file)"
                )
                
                conn.commit()
            
            logging.info(
                f"[DataLoaderSQLite:create_sqlite_db] "
                f"Created SQLite database at {db_path}"
            )
        
        except sqlite3.Error as e:
            logging.error(
                f"[DataLoaderSQLite:create_sqlite_db] "
                f"Error creating database: {e}"
            )
            raise

    @staticmethod
    def insert_document(
        db_path: str | Path,
        doc_id: str,
        doc_text: str,
        source_file: str,
        parent_doc_id: Optional[str] = None,
        chunk_index: Optional[int] = None
    ) -> None:
        """
        Insert a document into the SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
            doc_id: Unique document identifier
            doc_text: Document content
            source_file: Source file name
            parent_doc_id: Parent document ID (for chunks)
            chunk_index: Chunk sequence number (for chunks)
        """
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO documents (doc_id, doc_text, source_file, parent_doc_id, chunk_index)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (doc_id, doc_text, source_file, parent_doc_id, chunk_index)
                )
                conn.commit()
        
        except sqlite3.Error as e:
            logging.error(
                f"[DataLoaderSQLite:insert_document] "
                f"Error inserting document {doc_id}: {e}"
            )
            raise

