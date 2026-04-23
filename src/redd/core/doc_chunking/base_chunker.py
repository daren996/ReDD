"""
Base Chunker

Abstract base class for all document chunking algorithms.
Provides common functionality for loading data from various sources
and writing chunked results to SQLite database files.
"""

from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..data_loader import create_data_loader, DataLoaderBase


class BaseChunker(ABC):
    """
    Abstract base class for document chunking algorithms.
    
    Provides:
    - Loading documents from various sources via DataLoader (JSON, SQLite, etc.)
    - Writing chunked documents to a SQLite database (unified output format)
    - Mapping preservation between original docs and chunked docs
    
    Subclasses must implement:
    - chunk_document(): Split a single document into chunks
    - chunker_name: Property returning the chunker identifier for output naming
    
    Config keys used:
    - data_main: Base data directory (e.g., "dataset/fda_sqlite/")
    - exp_dataset_task_list: List of task folder paths to process (e.g., ["no_chunk"])
    - out_main: Output directory (optional, defaults to data_main)
    - data_loader_type: Type of data loader (e.g., "sqlite", "standard")
    """
    
    def __init__(self, config: Dict[str, Any], loader: Optional[DataLoaderBase] = None):
        self.config = config
        self.loader = loader
        
        # Extract common config values
        self.data_main = config.get("data_main", "dataset/")
        self.out_main = config.get("out_main", self.data_main)
        self.loader_type = config.get("data_loader_type")
        
        # data_loader_type is required only if loader is not provided
        if self.loader is None and self.loader_type is None:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] data_loader_type must be specified in config "
                f"when loader is not provided"
            )
        
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"Initialized with data_main={self.data_main}")
    
    @property
    @abstractmethod
    def chunker_name(self) -> str:
        """
        Return the chunker identifier used for output file naming.
        
        Example: "fixed_size_512" -> output task folder: "fixed_size_512/"
        """
        pass
    
    @abstractmethod
    def chunk_document(
        self, 
        doc_text: str, 
        doc_id: str, 
        metadata: Dict[str, Any]
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Split a single document into chunks.
        
        Args:
            doc_text: The document text to chunk
            doc_id: Original document ID
            metadata: Original document metadata
            
        Returns:
            List of tuples: (chunk_text, chunk_id, chunk_metadata)
            - chunk_id should be unique and traceable to original doc
            - chunk_metadata should include 'parent_doc_id' and 'chunk_index'
        """
        pass
    
    def __call__(self, dataset_task_list: Optional[List[str]] = None):
        """
        Main entry point for chunking.
        
        Args:
            dataset_task_list: Optional list of dataset/task folders to process.
                              If None, uses exp_dataset_task_list from config.
        """
        if dataset_task_list is None:
            dataset_task_list = self.config.get("exp_dataset_task_list", [])
        
        if not dataset_task_list:
            logging.error(f"[{self.__class__.__name__}:__call__] "
                         f"No dataset_task_list provided and exp_dataset_task_list not in config")
            raise ValueError("No dataset_task_list provided")
        
        logging.info(f"[{self.__class__.__name__}:__call__] "
                    f"Processing {len(dataset_task_list)} dataset(s)")
        
        for data_path in dataset_task_list:
            self._process_dataset(data_path)
    
    def _process_dataset(self, data_path: str):
        """
        Process a single dataset.
        
        Args:
            data_path: Relative path to the task folder (e.g., "no_chunk" or "fortune/default_task")
        """
        # Build full path
        full_data_path = Path(self.data_main) / data_path
        
        logging.info(f"[{self.__class__.__name__}:_process_dataset] "
                    f"Processing: {full_data_path}")
        
        # Create data loader
        self.loader = create_data_loader(
            full_data_path,
            loader_type=self.loader_type,
        )
        
        logging.info(f"[{self.__class__.__name__}:_process_dataset] "
                    f"Created {self.loader.__class__.__name__} for {full_data_path}")
        
        # Determine output path — write to a sibling task folder
        data_root = self.loader.data_root
        output_db_name = self.config.get("output_db_name", f"{self.chunker_name}.db")
        if not output_db_name.endswith(".db"):
            output_db_name = f"{output_db_name}.db"
        
        # Strip .db suffix to get the output task folder name
        output_task_name = output_db_name.removesuffix(".db")
        output_task_dir = data_root.parent / output_task_name
        output_task_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_task_dir / "documents.db"
        
        # Copy schema.db and queries.json from source task folder
        import shutil
        for fname in ("schema.db", "queries.json"):
            src = data_root / fname
            dst = output_task_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))
                logging.info(f"[{self.__class__.__name__}:_process_dataset] "
                            f"Copied {fname} to {output_task_dir}")
        
        # Run chunking
        self._run_chunking(output_path)
    
    def _run_chunking(
        self, 
        output_path: Path,
        overwrite: bool = True,
    ) -> Path:
        """
        Run the chunking process on all documents.
        
        Args:
            output_path: Path to output SQLite database
            overwrite: If True, overwrite existing output file.
            
        Returns:
            Path to the created output database
        """
        if output_path.exists():
            if overwrite:
                logging.warning(f"[{self.__class__.__name__}:_run_chunking] "
                              f"Overwriting existing file: {output_path}")
                output_path.unlink()
            else:
                raise FileExistsError(
                    f"[{self.__class__.__name__}:_run_chunking] Output file already exists: {output_path}. "
                    f"Use overwrite=True to replace."
                )
        
        logging.info(f"[{self.__class__.__name__}:_run_chunking] "
                    f"Starting chunking process -> {output_path}")
        
        # Create new SQLite database with standard schema
        conn = sqlite3.connect(str(output_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Create standard tables (documents + mapping only)
            self._create_output_schema(cursor)
            
            # Track statistics
            total_docs = 0
            total_chunks = 0
            
            # Collect all original mappings for reference (if source supports it)
            original_mappings = self._load_original_mappings()
            
            # Process each document
            for doc_text, doc_id, metadata in self.loader.iter_docs():
                total_docs += 1
                
                # Apply chunking
                chunks = self.chunk_document(doc_text, doc_id, metadata)
                
                # Determine match_type for chunks:
                # - If only 1 chunk (document wasn't split), preserve original match_type
                # - If multiple chunks (document was split), use "partial"
                is_chunked = len(chunks) > 1
                
                for chunk_text, chunk_id, chunk_metadata in chunks:
                    total_chunks += 1
                    
                    # Insert chunk as new document
                    # Mark is_chunked=1 if document was split into multiple chunks
                    self._insert_document(cursor, chunk_id, chunk_text, chunk_metadata, is_chunked=is_chunked)
                    
                    # Preserve mappings from original document
                    if doc_id in original_mappings:
                        for mapping in original_mappings[doc_id]:
                            # If document was split into multiple chunks, 
                            # each chunk only contains partial information
                            if is_chunked:
                                chunk_match_type = "partial"
                            else:
                                chunk_match_type = mapping.get("match_type", "full")
                            
                            self._insert_mapping(
                                cursor, 
                                chunk_id,
                                mapping["table_name"],
                                mapping["row_id"],
                                chunk_match_type
                            )
            
            conn.commit()
            
            logging.info(f"[{self.__class__.__name__}:_run_chunking] "
                        f"Chunking complete: {total_docs} docs -> {total_chunks} chunks")
            logging.info(f"[{self.__class__.__name__}:_run_chunking] "
                        f"Output saved to: {output_path}")
            
        except Exception as e:
            conn.rollback()
            logging.error(f"[{self.__class__.__name__}:_run_chunking] "
                         f"Error during chunking: {e}")
            # Clean up partial output
            if output_path.exists():
                output_path.unlink()
            raise
        finally:
            conn.close()
        
        return output_path
    
    def _create_output_schema(self, cursor: sqlite3.Cursor) -> None:
        """Create the standard SQLite output schema (documents + mapping only)."""
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_text TEXT,
                source_file TEXT,
                parent_doc_id TEXT,
                chunk_index INTEGER,
                is_chunked INTEGER DEFAULT 0
            )
        """)
        
        # Mapping table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mapping (
                table_name TEXT NOT NULL,
                row_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                match_type TEXT DEFAULT 'full',
                PRIMARY KEY (table_name, row_id, doc_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mapping_table_row ON mapping(table_name, row_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mapping_doc ON mapping(doc_id)")
    
    def _load_original_mappings(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all mappings from the source database.
        
        Returns:
            Dict mapping doc_id -> list of mapping records
        """
        mappings: Dict[str, List[Dict[str, Any]]] = {}
        
        if self.loader is None:
            return mappings
        
        try:
            cursor = self.loader._input_conn.cursor()
            cursor.execute("SELECT doc_id, table_name, row_id, match_type FROM mapping")
            
            for row in cursor.fetchall():
                doc_id = row[0]
                if doc_id not in mappings:
                    mappings[doc_id] = []
                mappings[doc_id].append({
                    "table_name": row[1],
                    "row_id": row[2],
                    "match_type": row[3]
                })
                
        except sqlite3.OperationalError as e:
            logging.warning(f"[{self.__class__.__name__}:_load_original_mappings] "
                          f"Failed to load mappings: {e}")
        
        return mappings
    
    def _insert_document(
        self, 
        cursor: sqlite3.Cursor, 
        doc_id: str, 
        doc_text: str,
        metadata: Dict[str, Any],
        is_chunked: bool = False
    ) -> None:
        """Insert a document into the output database.
        
        Args:
            cursor: SQLite cursor
            doc_id: Document ID
            doc_text: Document text content
            metadata: Document metadata
            is_chunked: Whether this document was created from chunking a larger document
        """
        cursor.execute(
            """
            INSERT INTO documents (doc_id, doc_text, source_file, parent_doc_id, chunk_index, is_chunked)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                doc_text,
                metadata.get("source_file"),
                metadata.get("parent_doc_id"),
                metadata.get("chunk_index"),
                1 if is_chunked else 0
            )
        )
    
    def _insert_mapping(
        self,
        cursor: sqlite3.Cursor,
        doc_id: str,
        table_name: str,
        row_id: str,
        match_type: str = "full"
    ) -> None:
        """Insert a mapping record into the output database."""
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO mapping (doc_id, table_name, row_id, match_type)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, table_name, row_id, match_type)
            )
        except sqlite3.IntegrityError:
            # Mapping already exists, skip
            pass
    
    def preview(self, data_path: Optional[str] = None, num_docs: int = 3) -> List[Dict[str, Any]]:
        """
        Preview chunking results without writing to database.
        
        Args:
            data_path: Relative path to the data file (not needed if loader was provided)
            num_docs: Number of documents to preview
            
        Returns:
            List of preview results, each containing original doc info and chunks
        """
        # Use existing loader if available, otherwise create one
        if self.loader is not None:
            loader = self.loader
        elif data_path is not None:
            full_data_path = Path(self.data_main) / data_path
            loader = create_data_loader(
                full_data_path,
                loader_type=self.loader_type,
            )
        else:
            raise ValueError(
                f"[{self.__class__.__name__}:preview] data_path is required when loader is not provided"
            )
        
        results = []
        count = 0
        
        for doc_text, doc_id, metadata in loader.iter_docs():
            if count >= num_docs:
                break
                
            chunks = self.chunk_document(doc_text, doc_id, metadata)
            
            results.append({
                "original_doc_id": doc_id,
                "original_length": len(doc_text),
                "num_chunks": len(chunks),
                "chunks": [
                    {
                        "chunk_id": cid,
                        "chunk_length": len(ctxt),
                        "preview": ctxt[:200] + "..." if len(ctxt) > 200 else ctxt
                    }
                    for ctxt, cid, _ in chunks
                ]
            })
            count += 1
        
        return results
    
    def __str__(self) -> str:
        """String representation of the chunker."""
        return f"{self.__class__.__name__}(chunker_name={self.chunker_name})"
