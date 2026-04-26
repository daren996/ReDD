"""
Base class for Chunk Attribute Mapping.

This module defines the abstract interface for mapping partial document chunks
to their relevant schema attributes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..data_loader import DataLoaderBase, create_data_loader


class ChunkAttributeMapperBase(ABC):
    """
    Abstract base class for chunk attribute mapping.
    
    When documents are chunked into smaller segments, partial chunks may only
    contain information relevant to a subset of the original table's attributes.
    This class provides the interface for using LLMs to identify which attributes
    are actually present in each partial chunk.
    
    The results are stored in the SQLite database in a new table:
    - chunk_attr_mapping: Maps (doc_id, table_name) -> list of relevant attributes
    
    Subclasses must implement:
    - _map_chunk_to_attributes(): Use LLM to identify relevant attributes
    """
    
    # Default prompt path (can be overridden in config)
    # Using 1to1 prompt: assumes each chunk belongs to exactly one table
    DEFAULT_PROMPT_PATH = "prompts/chunk_attr_mapping_1to1_1_0.txt"
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        loader: Optional[DataLoaderBase] = None
    ):
        """
        Initialize the chunk attribute mapper.
        
        Args:
            config: Configuration dictionary containing:
                - data_main: Base data directory
                - exp_dataset_task_list: List of data paths to process
                - data_loader_type: Type of data loader
                - prompt_path: Path to the prompt template (optional)
            loader: Optional pre-created DataLoader instance
        """
        self.config = config
        self.loader = loader
        
        # Extract common config values
        self.data_main = config.get("data_main", "dataset/")
        self.loader_type = config.get("data_loader_type")
        
        # Validate loader_type if loader not provided
        if self.loader is None and self.loader_type is None:
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] data_loader_type must be specified in config "
                f"when loader is not provided"
            )
        
        # Load prompt template
        prompt_path = config.get("prompt_path", self.DEFAULT_PROMPT_PATH)
        self.prompt_template = self._load_prompt(prompt_path)
        
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"Initialized with data_main={self.data_main}")
    
    def __call__(self, dataset_task_list: Optional[List[str]] = None) -> None:
        """
        Main entry point for chunk attribute mapping.
        
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
    
    def _process_dataset(self, data_path: str) -> None:
        """
        Process a single dataset to map partial chunks to attributes.
        
        Args:
            data_path: Relative path to the data (e.g., "fixed_size_50k")
        """
        # Build full path (for logging)
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
        
        # Get partial chunks
        partial_chunks = self._get_partial_chunks()
        
        if not partial_chunks:
            logging.info(f"[{self.__class__.__name__}:_process_dataset] "
                        f"No partial chunks found in {full_data_path}")
            return
        
        logging.info(f"[{self.__class__.__name__}:_process_dataset] "
                    f"Found {len(partial_chunks)} partial chunks to process")
        
        # Load schemas
        schema_general = self.loader.load_schema_general()
        table2schema = {s.get("Schema Name", s.get("table_name", "")): s 
                       for s in schema_general}
        
        # Ensure output table exists
        self._create_output_table()
        
        # Load existing mappings to skip already processed chunks
        existing_mappings = self._get_existing_mappings()
        
        # Process each partial chunk and get mappings
        mappings = self._run_mapping(partial_chunks, table2schema, existing_mappings)
        
        # Validate: check if union of chunk attrs equals table attrs
        all_mappings = {**existing_mappings, **mappings}
        self._validate_attr_completeness(all_mappings, table2schema)
    
    def _get_partial_chunks(self) -> List[Dict[str, Any]]:
        """
        Get all partial chunks from the database.
        
        Returns:
            List of dicts with keys: doc_id, table_name, row_id
        """
        partial_chunks = []
        
        try:
            cursor = self.loader._input_conn.cursor()
            
            # Get all partial mappings
            cursor.execute("""
                SELECT DISTINCT m.doc_id, m.table_name, m.row_id
                FROM mapping m
                WHERE m.match_type = 'partial'
                ORDER BY m.doc_id
            """)
            
            for row in cursor.fetchall():
                partial_chunks.append({
                    "doc_id": row[0],
                    "table_name": row[1],
                    "row_id": row[2],
                })
                
        except sqlite3.OperationalError as e:
            logging.warning(f"[{self.__class__.__name__}:_get_partial_chunks] "
                          f"Failed to query partial chunks: {e}")
        
        return partial_chunks
    
    def _get_existing_mappings(self) -> Dict[Tuple[str, str], List[str]]:
        """
        Load existing chunk attribute mappings from the database.
        
        Returns:
            Dict mapping (doc_id, table_name) -> list of mapped attributes
        """
        existing = {}
        
        try:
            cursor = self.loader._input_conn.cursor()
            
            # Check if chunk_attr_mapping table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='chunk_attr_mapping'
            """)
            
            if cursor.fetchone() is None:
                return existing
            
            cursor.execute("""
                SELECT doc_id, table_name, attributes
                FROM chunk_attr_mapping
            """)
            
            for row in cursor.fetchall():
                doc_id, table_name, attrs_json = row
                key = (doc_id, table_name)
                try:
                    attrs = json.loads(attrs_json) if attrs_json else []
                    existing[key] = attrs
                except json.JSONDecodeError:
                    existing[key] = []
                    
        except sqlite3.OperationalError as e:
            logging.warning(f"[{self.__class__.__name__}:_get_existing_mappings] "
                          f"Failed to load existing mappings: {e}")
        
        return existing
    
    def _run_mapping(
        self, 
        partial_chunks: List[Dict[str, Any]], 
        table2schema: Dict[str, Dict[str, Any]],
        existing_mappings: Dict[Tuple[str, str], List[str]]
    ) -> Dict[Tuple[str, str], List[str]]:
        """
        Run the attribute mapping process for all partial chunks.
        
        Args:
            partial_chunks: List of partial chunk info dicts
            table2schema: Mapping from table name to schema dict
            existing_mappings: Already processed mappings to skip
            
        Returns:
            Dict mapping (doc_id, table_name) -> list of relevant attributes
        """
        from tqdm import tqdm
        
        # Result storage
        mappings: Dict[Tuple[str, str], List[str]] = {}
        
        # Group chunks by (doc_id, table_name) to avoid redundant processing
        chunk_groups: Dict[Tuple[str, str], List[str]] = {}
        for chunk in partial_chunks:
            key = (chunk["doc_id"], chunk["table_name"])
            if key not in chunk_groups:
                chunk_groups[key] = []
            chunk_groups[key].append(chunk["row_id"])
        
        # Filter out already processed
        to_process = {k: v for k, v in chunk_groups.items() if k not in existing_mappings}
        
        if not to_process:
            logging.info(f"[{self.__class__.__name__}:_run_mapping] "
                        f"All {len(chunk_groups)} chunk groups already processed")
            return mappings
        
        logging.info(f"[{self.__class__.__name__}:_run_mapping] "
                    f"Processing {len(to_process)} new chunk groups "
                    f"(skipping {len(chunk_groups) - len(to_process)} existing)")
        
        progress_bar = tqdm(total=len(to_process), desc="Mapping chunk attributes")
        
        for (doc_id, table_name), row_ids in to_process.items():
            try:
                # Get document text
                doc_text, _, metadata = self.loader.get_doc(doc_id)
                
                # Get schema for this table
                schema = table2schema.get(table_name)
                if not schema:
                    logging.warning(f"[{self.__class__.__name__}:_run_mapping] "
                                  f"Schema not found for table: {table_name}")
                    progress_bar.update(1)
                    continue
                
                # Map chunk to attributes using LLM
                relevant_attrs = self._map_chunk_to_attributes(doc_text, schema)
                
                # Store result in memory
                mappings[(doc_id, table_name)] = relevant_attrs
                
                # Save to database
                self._save_mapping(doc_id, table_name, relevant_attrs)
                
            except Exception as e:
                logging.error(f"[{self.__class__.__name__}:_run_mapping] "
                             f"Error processing chunk {doc_id}: {e}")
            
            progress_bar.update(1)
        
        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}:_run_mapping] "
                    f"Completed attribute mapping for {len(mappings)} chunks")
        
        return mappings
    
    def _create_output_table(self) -> None:
        """Create the chunk_attr_mapping table if it doesn't exist."""
        cursor = self.loader._input_conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_attr_mapping (
                doc_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                attributes TEXT,
                PRIMARY KEY (doc_id, table_name)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_attr_doc 
            ON chunk_attr_mapping(doc_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_attr_table 
            ON chunk_attr_mapping(table_name)
        """)
        
        self.loader._input_conn.commit()
    
    def _save_mapping(
        self, 
        doc_id: str, 
        table_name: str, 
        attributes: List[str]
    ) -> None:
        """
        Save a chunk attribute mapping to the database.
        
        Args:
            doc_id: Document/chunk ID
            table_name: Table name
            attributes: List of relevant attribute names
        """
        cursor = self.loader._input_conn.cursor()
        
        attrs_json = json.dumps(attributes, ensure_ascii=False)
        
        cursor.execute("""
            INSERT OR REPLACE INTO chunk_attr_mapping 
            (doc_id, table_name, attributes)
            VALUES (?, ?, ?)
        """, (doc_id, table_name, attrs_json))
        
        self.loader._input_conn.commit()
    
    def _validate_attr_completeness(
        self, 
        mappings: Dict[Tuple[str, str], List[str]],
        table2schema: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Validate that the union of chunk attributes equals the table attributes.
        
        For each parent document that was split into chunks, check if the union of
        all chunk attributes matches the original table's attributes.
        
        Args:
            mappings: Dict mapping (doc_id, table_name) -> list of relevant attributes
            table2schema: Mapping from table name to schema dict
        """
        cursor = self.loader._input_conn.cursor()
        
        # Get all parent_doc_ids that have partial chunks
        cursor.execute("""
            SELECT DISTINCT d.parent_doc_id, m.table_name
            FROM documents d
            JOIN mapping m ON d.doc_id = m.doc_id
            WHERE m.match_type = 'partial'
            ORDER BY d.parent_doc_id
        """)
        
        parent_table_pairs = cursor.fetchall()
        
        if not parent_table_pairs:
            logging.info(f"[{self.__class__.__name__}:_validate_attr_completeness] "
                        f"No partial chunks to validate")
            return
        
        logging.info(f"[{self.__class__.__name__}:_validate_attr_completeness] "
                    f"Validating {len(parent_table_pairs)} parent document(s)")
        
        mismatch_count = 0
        
        for parent_doc_id, table_name in parent_table_pairs:
            # Get all chunk doc_ids for this parent
            cursor.execute("""
                SELECT d.doc_id 
                FROM documents d
                WHERE d.parent_doc_id = ?
                ORDER BY d.chunk_index
            """, (parent_doc_id,))
            
            chunk_doc_ids = [row[0] for row in cursor.fetchall()]
            
            if not chunk_doc_ids:
                continue
            
            # Get union of all chunk attributes from mappings
            union_attrs = set()
            for chunk_doc_id in chunk_doc_ids:
                attrs = mappings.get((chunk_doc_id, table_name), [])
                union_attrs.update(attrs)
            
            # Get table schema attributes
            schema = table2schema.get(table_name, {})
            raw_attrs = schema.get("Attributes") or schema.get("columns", [])
            
            table_attrs = set()
            for attr in raw_attrs:
                if isinstance(attr, dict):
                    attr_name = attr.get("Attribute Name") or attr.get("name", "")
                    if attr_name:
                        table_attrs.add(attr_name)
                elif isinstance(attr, str):
                    table_attrs.add(attr)
            
            # Compare
            if union_attrs != table_attrs:
                mismatch_count += 1
                missing = table_attrs - union_attrs
                extra = union_attrs - table_attrs
                
                logging.warning(
                    f"[{self.__class__.__name__}:_validate_attr_completeness] "
                    f"Attribute mismatch for parent_doc_id={parent_doc_id}, table={table_name}:\n"
                    f"  Chunks: {chunk_doc_ids}\n"
                    f"  Union of chunk attrs ({len(union_attrs)}): {sorted(union_attrs)}\n"
                    f"  Table attrs ({len(table_attrs)}): {sorted(table_attrs)}\n"
                    f"  Missing from chunks: {sorted(missing) if missing else 'none'}\n"
                    f"  Extra in chunks: {sorted(extra) if extra else 'none'}"
                )
        
        if mismatch_count == 0:
            logging.info(f"[{self.__class__.__name__}:_validate_attr_completeness] "
                        f"All {len(parent_table_pairs)} parent documents passed validation")
        else:
            logging.warning(f"[{self.__class__.__name__}:_validate_attr_completeness] "
                          f"{mismatch_count}/{len(parent_table_pairs)} parent documents have attribute mismatches")
    
    @abstractmethod
    def _map_chunk_to_attributes(
        self, 
        chunk_text: str, 
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Use LLM to identify which attributes are present in a chunk.
        
        Args:
            chunk_text: The text content of the chunk
            schema: Schema dict with "Schema Name" and "Attributes" keys
            
        Returns:
            List of attribute names that are present/relevant in the chunk
        """
        pass
    
    def _build_prompt_input(
        self, 
        chunk_text: str, 
        schema: Dict[str, Any]
    ) -> str:
        """
        Build the input for the LLM prompt.
        
        Args:
            chunk_text: Document chunk text
            schema: Schema dictionary
            
        Returns:
            JSON-formatted input string
        """
        # Extract schema name and attributes
        schema_name = schema.get("Schema Name") or schema.get("table_name", "")
        raw_attrs = schema.get("Attributes") or schema.get("columns", [])
        
        # Format attributes consistently
        attributes = []
        for attr in raw_attrs:
            if isinstance(attr, dict):
                attr_name = attr.get("Attribute Name") or attr.get("name", "")
                description = attr.get("Description") or attr.get("description", "")
                attributes.append({
                    "Attribute Name": attr_name,
                    "Description": description
                })
            elif isinstance(attr, str):
                attributes.append({
                    "Attribute Name": attr,
                    "Description": ""
                })
        
        input_data = {
            "Document": chunk_text,
            "Schema": {
                "Schema Name": schema_name,
                "Attributes": attributes
            }
        }
        
        return json.dumps(input_data, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _load_prompt(path: str) -> str:
        """Load prompt template from file."""
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return fp.read()
        except FileNotFoundError:
            logging.warning(f"[ChunkAttributeMapperBase:_load_prompt] "
                          f"Prompt file not found: {path}, using default prompt")
            return ChunkAttributeMapperBase._get_default_prompt()
    
    @staticmethod
    def _get_default_prompt() -> str:
        """Return the default prompt template."""
        return """You are a database expert analyzing a document chunk.
Your task is to identify which attributes from a given schema can be found or inferred from the document.

## Instructions
1. Read the document chunk carefully
2. For each attribute in the schema, determine if the document contains information relevant to that attribute
3. Return ONLY the attribute names that are present or can be inferred from the document
4. If no attributes are found, return an empty list

## Output Format
Return a JSON object with a single key "relevant_attributes" containing a list of attribute names.

Example:
```json
{
    "relevant_attributes": ["name", "email", "department"]
}
```

If no attributes are found:
```json
{
    "relevant_attributes": []
}
```

## Input
"""
    
    def __str__(self) -> str:
        """String representation of the mapper."""
        return f"{self.__class__.__name__}(data_main={self.data_main})"
