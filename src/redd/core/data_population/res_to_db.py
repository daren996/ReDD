"""
Result to Database Converter.

This module provides utilities to convert data population extraction results
(res_tabular_data_*.json) into SQLite database format for evaluation.

Output format: result database file (.db) with tables based on schema structure.

Conversion Logic:
1. Load schema from source dataset to get table names and their attributes
2. Group documents by their table assignment (res field)
3. For chunked documents, merge chunks with same parent_doc_id
4. Create output database with tables matching schema structure

TODO:
- [ ] Handle single doc -> multiple rows/tables (1:N mapping)
- [ ] Handle multiple docs -> single row update (N:1 mapping)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.constants import (
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTES_KEY,
    NULL_VALUE,
    RESULT_DATA_KEY,
    RESULT_TABLE_KEY,
    SCHEMA_NAME_KEY,
)

__all__ = ["ResToDBConverter"]


class ResToDBConverter:
    """
    Converter to transform extraction results into SQLite database format.
    
    This class handles:
    - Loading schema from source database to determine table structure
    - Loading extraction results from JSON files
    - Grouping documents by table assignment
    - Merging chunks belonging to the same parent document
    - Creating output database with schema-based table structure
    
    Merge Strategies for chunk attributes:
    - "first_non_null": Use the first non-null value (default)
    - "last_non_null": Use the last non-null value
    - "concat": Concatenate all non-null values with separator
    - "majority": Use the most frequent non-null value
    """
    
    MERGE_STRATEGIES = ["first_non_null", "last_non_null", "concat", "majority"]
    
    def __init__(
        self,
        source_db_path: Union[str, Path],
        result_json_path: Union[str, Path],
        output_db_path: Optional[Union[str, Path]] = None,
        query_id: str = "Q1",
        merge_strategy: str = "first_non_null",
        concat_separator: str = " | ",
    ):
        """
        Initialize the converter.
        
        Args:
            source_db_path: Path to the source dataset (contains schema and document metadata)
            result_json_path: Path to the extraction result JSON file
            output_db_path: Path for the output database. If None, auto-generated
            query_id: Query ID for loading query-specific schema (default: "Q1")
            merge_strategy: Strategy for merging chunk attribute values
            concat_separator: Separator used when merge_strategy is "concat"
        """
        self.source_db_path = Path(source_db_path)
        self.result_json_path = Path(result_json_path)
        self.query_id = query_id
        
        if output_db_path is None:
            output_db_path = self.result_json_path.with_suffix(".db")
        self.output_db_path = Path(output_db_path)
        
        if merge_strategy not in self.MERGE_STRATEGIES:
            logging.warning(f"[{self.__class__.__name__}:__init__] "
                          f"Unknown merge strategy: {merge_strategy}, using 'first_non_null'")
            merge_strategy = "first_non_null"
        self.merge_strategy = merge_strategy
        self.concat_separator = concat_separator
        
        # Validate paths
        if not self.source_db_path.exists():
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Source database not found: {self.source_db_path}")
            raise FileNotFoundError(f"Source database not found: {self.source_db_path}")
        
        if not self.result_json_path.exists():
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Result JSON not found: {self.result_json_path}")
            raise FileNotFoundError(f"Result JSON not found: {self.result_json_path}")
        
        logging.info(f"[{self.__class__.__name__}:__init__] Initialized: "
                    f"source={self.source_db_path.name}, result={self.result_json_path.name}, "
                    f"query_id={query_id}, merge_strategy={merge_strategy}")
    
    def convert(self) -> Path:
        """
        Execute the conversion from result JSON to database.
        
        Returns:
            Path to the created output database
        """
        logging.info(f"[{self.__class__.__name__}:convert] Starting conversion...")
        
        # Step 1: Load schema from source dataset
        schema_dict = self._load_schema()
        if not schema_dict:
            logging.warning(f"[{self.__class__.__name__}:convert] No schema found")
            return self.output_db_path
        logging.info(f"[{self.__class__.__name__}:convert] Loaded schema: {list(schema_dict.keys())}")
        
        # Step 2: Load result JSON
        result_data = self._load_result_json()
        if not result_data:
            logging.warning(f"[{self.__class__.__name__}:convert] No result data")
            return self.output_db_path
        
        # Step 3: Load document metadata for chunk handling
        doc_metadata = self._load_doc_metadata()
        
        # Step 4: Check if chunking is needed
        has_chunking = self._detect_chunking(doc_metadata)
        logging.info(f"[{self.__class__.__name__}:convert] Chunking detected: {has_chunking}")
        
        # Step 5: Group documents by table and merge chunks if needed
        table_data = self._group_by_table(result_data, doc_metadata, has_chunking)
        
        # Step 6: Create output database with schema-based structure
        self._create_output_database(table_data, schema_dict)
        
        logging.info(f"[{self.__class__.__name__}:convert] Conversion complete: {self.output_db_path}")
        return self.output_db_path
    
    def _load_schema(self) -> Dict[str, List[str]]:
        """
        Load schema from the configured ReDD data loader.
        
        Falls back to general schema if no query-specific info is available.
        
        Returns:
            Dict mapping table_name -> list of attribute names
        """
        from ..data_loader import create_data_loader
        
        schema_dict = {}
        
        try:
            loader = create_data_loader(self.source_db_path)
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:_load_schema] "
                         f"Failed to create DataLoader: {e}")
            return schema_dict
        
        try:
            # Try query-specific schema first
            schemas = loader.load_schema_query(self.query_id)
            if not schemas:
                # Fall back to general schema
                schemas = loader.load_schema_general()
            
            for schema in schemas:
                table_name = schema.get(SCHEMA_NAME_KEY, "")
                attributes = schema.get(ATTRIBUTES_KEY, [])
                if table_name and attributes:
                    attr_names = [
                        a.get(ATTRIBUTE_NAME_KEY, "") if isinstance(a, dict) else str(a)
                        for a in attributes
                    ]
                    schema_dict[table_name] = [a for a in attr_names if a]
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:_load_schema] "
                         f"Failed to load schema: {e}")

        return schema_dict
    
    def _load_result_json(self) -> Dict[str, Any]:
        """Load extraction results from JSON file."""
        try:
            with open(self.result_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info(f"[{self.__class__.__name__}:_load_result_json] "
                        f"Loaded {len(data)} document results")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"[{self.__class__.__name__}:_load_result_json] "
                         f"Failed to parse JSON: {e}")
            raise
    
    def _load_doc_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load document metadata from source dataset.
        
        Returns:
            Dict mapping doc_id -> {"parent_doc_id": str, "chunk_index": int}
        """
        from ..data_loader import create_data_loader

        doc_metadata = {}
        try:
            loader = create_data_loader(self.source_db_path)
            for _doc_text, doc_id, metadata in loader.iter_docs():
                doc_metadata[str(doc_id)] = {
                    "parent_doc_id": metadata.get("parent_doc_id"),
                    "chunk_index": metadata.get("chunk_index"),
                }
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}:_load_doc_metadata] "
                           f"Failed to load document metadata: {e}")

        return doc_metadata
    
    def _detect_chunking(self, doc_metadata: Dict[str, Dict[str, Any]]) -> bool:
        """
        Detect if chunking is used (multiple docs share same parent_doc_id).
        """
        if not doc_metadata:
            return False
        
        parent_count: Dict[str, int] = defaultdict(int)
        for meta in doc_metadata.values():
            parent_id = meta.get("parent_doc_id")
            if parent_id is not None:
                parent_count[str(parent_id)] += 1
        
        for count in parent_count.values():
            if count > 1:
                return True
        return False
    
    def _group_by_table(
        self,
        result_data: Dict[str, Any],
        doc_metadata: Dict[str, Dict[str, Any]],
        has_chunking: bool,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Group documents by their table assignment and merge chunks.
        
        Returns:
            Dict mapping table_name -> {row_id: {attr: value, ...}, ...}
        """
        if has_chunking:
            return self._group_with_chunking(result_data, doc_metadata)
        else:
            return self._group_no_chunking(result_data, doc_metadata)
    
    def _group_no_chunking(
        self,
        result_data: Dict[str, Any],
        doc_metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Group documents by table without chunk merging."""
        table_data: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        
        for doc_id, doc_result in result_data.items():
            table_name = doc_result.get(RESULT_TABLE_KEY)
            data = doc_result.get(RESULT_DATA_KEY, {})
            
            if not table_name or table_name == NULL_VALUE:
                continue
            
            # Use parent_doc_id as row_id if available
            meta = doc_metadata.get(doc_id, {})
            row_id = meta.get("parent_doc_id", doc_id)
            if row_id is None:
                row_id = doc_id
            
            table_data[table_name][str(row_id)] = data
        
        logging.info(f"[{self.__class__.__name__}:_group_no_chunking] "
                    f"Grouped {sum(len(v) for v in table_data.values())} rows "
                    f"into {len(table_data)} tables")
        return dict(table_data)
    
    def _group_with_chunking(
        self,
        result_data: Dict[str, Any],
        doc_metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Group documents by table with chunk merging.
        
        Important: Only chunks with the SAME table assignment are merged together.
        If chunks from the same parent have different table assignments, they
        become separate rows in their respective tables.
        """
        # Group chunks by (parent_doc_id, table_name): {(parent, table): [(chunk_idx, data), ...]}
        parent_table_chunks: Dict[tuple, List[tuple]] = defaultdict(list)
        
        for doc_id, doc_result in result_data.items():
            meta = doc_metadata.get(doc_id, {})
            parent_doc_id = meta.get("parent_doc_id", doc_id)
            if parent_doc_id is None:
                parent_doc_id = doc_id
            
            chunk_index = meta.get("chunk_index", 0)
            table_name = doc_result.get(RESULT_TABLE_KEY)
            data = doc_result.get(RESULT_DATA_KEY, {})
            
            # Skip if no table assignment or NULL
            if not table_name or table_name == NULL_VALUE:
                continue
            
            # Group by (parent_doc_id, table_name)
            key = (str(parent_doc_id), table_name)
            parent_table_chunks[key].append((chunk_index, data))
        
        # Merge chunks for each (parent, table) group
        table_data: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        
        for (parent_doc_id, table_name), chunks in parent_table_chunks.items():
            # Sort by chunk_index
            chunks.sort(key=lambda x: x[0] if x[0] is not None else 0)
            
            # Merge attribute values from all chunks with same table
            merged_attrs = self._merge_attributes_simple(chunks)
            table_data[table_name][parent_doc_id] = merged_attrs
        
        logging.info(f"[{self.__class__.__name__}:_group_with_chunking] "
                    f"Grouped {len(parent_table_chunks)} (parent, table) pairs into "
                    f"{sum(len(v) for v in table_data.values())} rows "
                    f"across {len(table_data)} tables")
        return dict(table_data)
    
    def _merge_attributes_simple(self, chunks: List[tuple]) -> Dict[str, Any]:
        """
        Merge attribute values from chunks (already filtered by same table).
        
        Args:
            chunks: List of (chunk_index, data) tuples, sorted by chunk_index
        """
        attr_values: Dict[str, List[Any]] = defaultdict(list)
        
        for _, data in chunks:
            if not isinstance(data, dict):
                continue
            for attr, value in data.items():
                attr_values[attr].append(value)
        
        merged = {}
        for attr, values in attr_values.items():
            merged[attr] = self._merge_values(values)
        return merged
    
    def _merge_attributes(self, chunks: List[tuple]) -> Dict[str, Any]:
        """Merge attribute values from multiple chunks."""
        attr_values: Dict[str, List[Any]] = defaultdict(list)
        
        for _, _, data in chunks:
            if not isinstance(data, dict):
                continue
            for attr, value in data.items():
                attr_values[attr].append(value)
        
        merged = {}
        for attr, values in attr_values.items():
            merged[attr] = self._merge_values(values)
        return merged
    
    def _merge_values(self, values: List[Any]) -> Any:
        """Merge a list of values into a single value based on strategy."""
        non_null_values = [v for v in values if v is not None and v != NULL_VALUE and v != ""]
        
        if not non_null_values:
            return None
        
        if self.merge_strategy == "first_non_null":
            return non_null_values[0]
        elif self.merge_strategy == "last_non_null":
            return non_null_values[-1]
        elif self.merge_strategy == "concat":
            str_values = [str(v) for v in non_null_values]
            seen = set()
            unique_values = [v for v in str_values if not (v in seen or seen.add(v))]
            return self.concat_separator.join(unique_values)
        elif self.merge_strategy == "majority":
            str_values = [str(v) for v in non_null_values]
            return max(set(str_values), key=str_values.count)
        else:
            return non_null_values[0]
    
    def _create_output_database(
        self,
        table_data: Dict[str, Dict[str, Dict[str, Any]]],
        schema_dict: Dict[str, List[str]],
    ) -> None:
        """
        Create the output SQLite database with schema-based table structure.
        
        Args:
            table_data: Dict mapping table_name -> {row_id: {attr: value, ...}, ...}
            schema_dict: Dict mapping table_name -> list of attribute names
        """
        # Remove existing output file
        if self.output_db_path.exists():
            self.output_db_path.unlink()
        
        self.output_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.output_db_path))
        cursor = conn.cursor()
        
        try:
            # Create tables based on schema
            for table_name, attributes in schema_dict.items():
                if not attributes:
                    continue
                
                safe_table = table_name.replace('"', '""')
                
                # Build column definitions from schema
                columns = ["row_id TEXT PRIMARY KEY"]
                for attr in attributes:
                    safe_attr = attr.replace('"', '""')
                    columns.append(f'"{safe_attr}" TEXT')
                
                create_sql = f'CREATE TABLE IF NOT EXISTS "{safe_table}" ({", ".join(columns)})'
                cursor.execute(create_sql)
                
                # Insert rows for this table
                rows = table_data.get(table_name, {})
                if not rows:
                    logging.info(f"[{self.__class__.__name__}:_create_output_database] "
                                f"Table '{table_name}': 0 rows (empty)")
                    continue
                
                # Build insert statement with schema attributes
                placeholders = ", ".join(["?"] * (len(attributes) + 1))
                safe_attrs = [f'"{attr.replace(chr(34), chr(34)+chr(34))}"' for attr in attributes]
                insert_sql = (
                    f'INSERT INTO "{safe_table}" (row_id, {", ".join(safe_attrs)}) '
                    f'VALUES ({placeholders})'
                )
                
                for row_id, row_data in rows.items():
                    # Get values in schema order, None for missing attributes
                    values = [row_id] + [self._serialize_value(row_data.get(attr)) for attr in attributes]
                    cursor.execute(insert_sql, values)
                
                logging.info(f"[{self.__class__.__name__}:_create_output_database] "
                            f"Table '{table_name}': {len(rows)} rows, {len(attributes)} attributes")
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logging.error(f"[{self.__class__.__name__}:_create_output_database] "
                         f"Failed to create database: {e}")
            raise
        finally:
            conn.close()
    
    @staticmethod
    def _serialize_value(value: Any) -> Optional[str]:
        """Serialize a value for storage in SQLite."""
        if value is None or value == NULL_VALUE:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"source={self.source_db_path.name}, "
            f"result={self.result_json_path.name}, "
            f"output={self.output_db_path.name})"
        )


def convert_result_to_db(
    source_db_path: Union[str, Path],
    result_json_path: Union[str, Path],
    output_db_path: Optional[Union[str, Path]] = None,
    query_id: str = "Q1",
    merge_strategy: str = "first_non_null",
    concat_separator: str = " | ",
) -> Path:
    """
    Convenience function to convert extraction result to database.
    
    Args:
        source_db_path: Path to the source dataset
        result_json_path: Path to the extraction result JSON
        output_db_path: Path for output database (auto-generated if None)
        query_id: Query ID for schema loading
        merge_strategy: Strategy for merging chunk values
        concat_separator: Separator for concat strategy
    
    Returns:
        Path to the created output database
    """
    converter = ResToDBConverter(
        source_db_path=source_db_path,
        result_json_path=result_json_path,
        output_db_path=output_db_path,
        query_id=query_id,
        merge_strategy=merge_strategy,
        concat_separator=concat_separator,
    )
    return converter.convert()
