# data_loader/data_loader_cuad.py
"""
DataLoader for CUAD_v1 dataset stored in SQLite format.

This loader reads the CUAD_v1 contract dataset from a SQLite database created
by the prep_cuadv1.py script. The database contains:
- Contracts: Contract metadata and file paths
- Categories: Category definitions (attributes to extract)
- Contract_Answers: Ground truth answers for each contract-category pair
- Queries: Query definitions with attributes to extract

Each document represents a complete contract with its full text.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Optional

from .data_loader_sqlite import DataLoaderSQLite

__all__ = ["DataLoaderCUAD"]


class DataLoaderCUAD(DataLoaderSQLite):
    """Dataset loader for CUAD_v1 contract dataset."""

    DEFAULT_FILEMAP = {
        "sqlite_db": "CUAD_v1.db",
    }

    def __init__(
        self, 
        data_root: str | Path, 
        *, 
        filemap: Dict[str, str] | None = None
    ):
        """
        Initialize the CUAD dataset loader.
        
        Args:
            data_root: Root directory of the CUAD dataset (should contain CUAD_v1.db)
            filemap: Custom file mapping (optional)
        """
        # Override parent's filemap with CUAD-specific defaults
        self._filemap = {**self.DEFAULT_FILEMAP, **(filemap or {})}
        self._encoding = "utf-8"
        
        # Initialize data_root from base class
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")
        
        # Initialize CUAD-specific database mode
        self._init_cuad_mode()
        
        # Load queries and schemas from database
        self._query_dict = self._load_queries_from_db()
        self._schema_general = self._load_schema_from_db()
        
        # Note: _doc_info is built lazily in get_doc_info()
        self._doc_info = {}

    def _init_cuad_mode(self):
        """Initialize for reading from CUAD SQLite database."""
        db_path = self._path("sqlite_db")
        if not db_path.exists():
            logging.warning(
                f"[{self.__class__.__name__}:_init_cuad_mode] "
                f"SQLite database not found: {db_path}"
            )
            self._db_path = None
            self._doc_ids = []
            self._contracts = {}
            return

        self._db_path = db_path
        
        # Load contract metadata from database
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # Load all contracts
                cursor.execute("""
                    SELECT contract_id, contract_name, contract_type, source, file_path 
                    FROM Contracts 
                    ORDER BY contract_id
                """)
                
                self._contracts = {}
                self._doc_ids = []
                
                for row in cursor.fetchall():
                    contract_id, contract_name, contract_type, source, file_path = row
                    doc_id = str(contract_id)
                    
                    self._contracts[doc_id] = {
                        "contract_id": contract_id,
                        "contract_name": contract_name,
                        "contract_type": contract_type,
                        "source": source,
                        "file_path": file_path
                    }
                    self._doc_ids.append(doc_id)
            
            logging.info(
                f"[{self.__class__.__name__}:_init_cuad_mode] "
                f"Found {len(self._doc_ids)} contracts in CUAD database"
            )
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:_init_cuad_mode] "
                f"Error reading from SQLite database: {e}"
            )
            self._doc_ids = []
            self._contracts = {}

    def _load_queries_from_db(self) -> Dict[str, Any]:
        """Load queries from the Queries table."""
        if self._db_path is None:
            return {}
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT query_id, query, attributes, sql FROM Queries")
                
                query_dict = {}
                for row in cursor.fetchall():
                    query_id, query_text, attributes_str, sql = row
                    
                    # Parse attributes (comma-separated category IDs)
                    attributes_list = []
                    if attributes_str:
                        category_ids = [int(x.strip()) for x in attributes_str.split(",")]
                        # Get category names for these IDs
                        attributes_list = self._get_category_names(cursor, category_ids)
                    
                    query_dict[str(query_id)] = {
                        "query": query_text,
                        "attributes": attributes_list,
                        "sql": sql if sql else ""
                    }
                
                logging.info(
                    f"[{self.__class__.__name__}:_load_queries_from_db] "
                    f"Loaded {len(query_dict)} queries from database"
                )
                return query_dict
                
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:_load_queries_from_db] "
                f"Error loading queries: {e}"
            )
            return {}

    def _get_category_names(self, cursor: sqlite3.Cursor, category_ids: List[int]) -> List[str]:
        """Get category names for a list of category IDs."""
        if not category_ids:
            return []
        
        placeholders = ",".join("?" * len(category_ids))
        cursor.execute(
            f"SELECT category_name FROM Categories WHERE category_id IN ({placeholders}) ORDER BY category_id",
            category_ids
        )
        return [row[0] for row in cursor.fetchall()]

    def _load_schema_from_db(self) -> List[Dict[str, Any]]:
        """Load schema (attributes) from the Categories table."""
        if self._db_path is None:
            return []
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT category_id, category_name, description, answer_format, group_id 
                    FROM Categories 
                    ORDER BY category_id
                """)
                
                # Build a single schema representing all contract attributes
                attributes = []
                for row in cursor.fetchall():
                    category_id, category_name, description, answer_format, group_id = row
                    
                    # Combine description and answer_format
                    full_description = description if description else ""
                    if answer_format:
                        full_description += f" [Expected format: {answer_format}]"
                    
                    attributes.append({
                        "Attribute Name": category_name,
                        "Description": full_description,
                        "category_id": category_id,
                        "answer_format": answer_format,
                        "group_id": group_id
                    })
                
                # Return as a list with a single schema (CUAD has one unified schema)
                schema = [{
                    "Schema Name": "CUAD_Contracts",
                    "Description": "Contract analysis attributes from CUAD dataset",
                    "Attributes": attributes
                }]
                
                logging.info(
                    f"[{self.__class__.__name__}:_load_schema_from_db] "
                    f"Loaded schema with {len(attributes)} attributes"
                )
                return schema
                
        except sqlite3.Error as e:
            logging.error(
                f"[{self.__class__.__name__}:_load_schema_from_db] "
                f"Error loading schema: {e}"
            )
            return []

    # ---------------- document access (override parent) -----------------

    def get_doc(self, doc_id: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Return (doc_text, doc_id, metadata_dict) for the given doc_id.
        
        The doc_text is loaded from the contract file specified in the database.
        
        Args:
            doc_id: Document identifier (contract_id as string)
            
        Returns:
            Tuple of (doc_text, doc_id, metadata_dict)
        """
        if doc_id not in self._contracts:
            logging.warning(
                f"[{self.__class__.__name__}:get_doc] "
                f"Contract {doc_id} not found in database"
            )
            return ("", doc_id, {})
        
        contract_info = self._contracts[doc_id]
        file_path = contract_info["file_path"]
        
        # Read contract text from file
        doc_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                doc_text = f.read()
        except FileNotFoundError:
            logging.error(
                f"[{self.__class__.__name__}:get_doc] "
                f"Contract file not found: {file_path}"
            )
        except Exception as e:
            logging.error(
                f"[{self.__class__.__name__}:get_doc] "
                f"Error reading contract file {file_path}: {e}"
            )
        
        # Build metadata
        metadata = {
            "contract_name": contract_info["contract_name"],
            "contract_type": contract_info["contract_type"],
            "source": contract_info["source"],
            "file_path": file_path,
            "table_name": "CUAD_Contracts"
        }
        
        return (doc_text, doc_id, metadata)

    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Return document info for doc_id, including ground truth data.
        
        The "data" field contains ALL attributes (categories) with their values.
        Attributes without answers are set to empty string "".
        
        Args:
            doc_id: Document identifier (contract_id as string)
            
        Returns:
            Dictionary with document info including:
            - doc: contract text
            - fn: contract name (table name)
            - data: dict of attribute_name -> answer_value (all attributes included)
        """
        # Check cache first
        if doc_id in self._doc_info:
            return self._doc_info[doc_id]
        
        if doc_id not in self._contracts:
            return None
        
        # Get document text and metadata
        doc_text, _, metadata = self.get_doc(doc_id)
        
        # Initialize ground truth data with ALL categories set to empty string
        ground_truth_data = {}
        if self._db_path:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    cursor = conn.cursor()
                    
                    # First, get ALL categories and initialize them to empty string
                    cursor.execute("SELECT category_id, category_name FROM Categories ORDER BY category_id")
                    for row in cursor.fetchall():
                        category_id, category_name = row
                        ground_truth_data[category_name] = ""
                    
                    # Then, get actual answers for this contract and overwrite
                    cursor.execute("""
                        SELECT ca.category_id, ca.answer_value, c.category_name
                        FROM Contract_Answers ca
                        JOIN Categories c ON ca.category_id = c.category_id
                        WHERE ca.contract_id = ?
                    """, (int(doc_id),))
                    
                    for row in cursor.fetchall():
                        category_id, answer_value, category_name = row
                        # Use empty string for NULL/None values
                        ground_truth_data[category_name] = answer_value if answer_value else ""
                    
            except sqlite3.Error as e:
                logging.error(
                    f"[{self.__class__.__name__}:get_doc_info] "
                    f"Error loading ground truth for contract {doc_id}: {e}"
                )
        
        # Build doc_info structure
        doc_info = {
            "doc": doc_text,
            "fn": "CUAD_Contracts",  # Table name
            "doc_id": doc_id,
            "contract_name": metadata["contract_name"],
            "contract_type": metadata["contract_type"],
            "source": metadata["source"],
            "data": ground_truth_data
        }
        
        # Cache it
        self._doc_info[doc_id] = doc_info
        
        return doc_info

    # ---------------- query / schema access (override parent) -------------

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
        """
        Load query-specific schema (subset of attributes for this query).
        
        For CUAD, this filters the general schema to only include attributes
        mentioned in the query.
        """
        query_info = self.get_query_info(qid)
        if not query_info:
            return []
        
        query_attributes = query_info.get("attributes", [])
        if not query_attributes:
            # If no specific attributes, return full schema
            return self._schema_general
        
        # Filter schema to only include requested attributes
        general_schema = self._schema_general[0] if self._schema_general else None
        if not general_schema:
            return []
        
        all_attributes = general_schema.get("Attributes", [])
        filtered_attributes = [
            attr for attr in all_attributes 
            if attr["Attribute Name"] in query_attributes
        ]
        
        return [{
            "Schema Name": general_schema["Schema Name"],
            "Description": general_schema["Description"],
            "Attributes": filtered_attributes
        }]

    # ---------------- hot-reload support -------------------------------

    def refresh(self) -> None:
        """Re-read data from database (hot-reload support)."""
        self._init_cuad_mode()
        self._query_dict = self._load_queries_from_db()
        self._schema_general = self._load_schema_from_db()
        self._doc_info = {}  # Clear cache

