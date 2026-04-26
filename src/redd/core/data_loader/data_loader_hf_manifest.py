"""HuggingFace-style manifest data loader for ReDD datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import yaml

from .data_loader_basic import DataLoaderBase

DEFAULT_EXTRACTION_QUERY_ID = "default"

__all__ = [
    "DEFAULT_EXTRACTION_QUERY_ID",
    "DataLoaderHFManifest",
    "build_default_query_records",
    "normalize_identifier",
]


def normalize_identifier(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or "unknown"


def build_default_query_records(
    schema_contract: Dict[str, Any],
    *,
    dataset_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Build the implicit extraction query for datasets without explicit queries."""
    tables = schema_contract.get("tables") if isinstance(schema_contract, dict) else []
    if not isinstance(tables, list) or not tables:
        return []

    required_tables: List[str] = []
    required_columns: List[str] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("table_id") or table.get("name") or "")
        if table_id:
            required_tables.append(table_id)
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            column_id = str(column.get("column_id") or "")
            if not column_id and table_id:
                column_name = str(column.get("name") or "")
                column_id = f"{table_id}.{column_name}" if column_name else ""
            if column_id:
                required_columns.append(column_id)

    if not required_tables and not required_columns:
        return []

    return [
        {
            "query_id": DEFAULT_EXTRACTION_QUERY_ID,
            "question": "Default extraction: extract every attribute in the query-specific schema.",
            "sql": "",
            "required_tables": required_tables,
            "required_columns": required_columns,
            "output_columns": [],
            "tags": ["default_extraction"],
            "difficulty": None,
            "default_extraction": True,
            "dataset_id": dataset_id,
        }
    ]


class DataLoaderHFManifest(DataLoaderBase):
    """Dataset loader for the ReDD manifest/parquet contract."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        manifest: str | Path = "manifest.yaml",
        filemap: Dict[str, str] | None = None,
        encoding: str = "utf-8",
        **_: Any,
    ) -> None:
        super().__init__(data_root)
        self._encoding = encoding
        self._manifest_path = self._resolve_path(manifest)
        with self._manifest_path.open("r", encoding=encoding) as file:
            self.manifest = yaml.safe_load(file) or {}

        self.dataset_id = str(self.manifest.get("dataset_id") or self.data_root.name)
        paths = self.manifest.get("paths") or {}
        self._documents_path = self._resolve_path(paths.get("documents", "data/documents.parquet"))
        self._ground_truth_path = self._resolve_path(paths.get("ground_truth", "data/ground_truth.parquet"))
        self._filemap = filemap or {}
        self._schema_path = self._resolve_path(
            self._filemap.get("schema_general") or paths.get("schema", "metadata/schema.json")
        )
        self._schema_query_pattern = self._filemap.get("schema_query")
        self._queries_path = self._resolve_path(paths.get("queries", "metadata/queries.json"))

        self._documents_df = pd.read_parquet(self._documents_path)
        self._ground_truth_df = (
            pd.read_parquet(self._ground_truth_path)
            if self._ground_truth_path.exists()
            else pd.DataFrame()
        )
        self._schema_contract = self._read_json(self._schema_path, encoding)
        self._queries_contract = self._read_json(self._queries_path, encoding)

        self._doc_ids = [str(value) for value in self._documents_df["doc_id"].tolist()]
        self._documents_by_id = {
            str(row["doc_id"]): row for _, row in self._documents_df.iterrows()
        }
        self._schema_general = self._schema_to_legacy(self._schema_contract)
        self._query_records = self._normalize_query_records(
            self._queries_contract,
            schema_contract=self._schema_contract,
            dataset_id=self.dataset_id,
        )
        self._query_dict = {
            str(record["query_id"]): self._query_to_legacy(record)
            for record in self._query_records
        }
        self._gt_by_doc = self._build_ground_truth_index()

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
        row = self._documents_by_id.get(str(doc_id))
        if row is None:
            return ("", str(doc_id), {})
        metadata = {
            "source_file": self._clean_scalar(row.get("source_id")),
            "table_name": self._clean_scalar(row.get("source_table")),
            "parent_doc_id": self._clean_scalar(row.get("parent_doc_id")),
            "chunk_index": self._clean_scalar(row.get("chunk_index")),
            "is_chunked": bool(row.get("is_chunked")) if "is_chunked" in row else False,
            "source_row_id": self._clean_scalar(row.get("source_row_id")),
            "split": self._clean_scalar(row.get("split")),
        }
        return (str(row.get("doc_text") or ""), str(doc_id), metadata)

    def get_doc_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        doc_text, resolved_doc_id, metadata = self.get_doc(doc_id)
        if not metadata and not doc_text:
            return None

        records = self._gt_by_doc.get(str(doc_id), [])
        data_records: List[Dict[str, Any]] = []
        for table_name, values in records:
            data_records.append({"table_name": table_name, "data": values})

        info = {
            "doc": doc_text,
            "fn": metadata.get("source_table") or metadata.get("source_file") or resolved_doc_id,
            "doc_id": resolved_doc_id,
            "parent_doc_id": metadata.get("parent_doc_id"),
            "chunk_index": metadata.get("chunk_index"),
            "source_row_id": metadata.get("source_row_id"),
            "data_records": data_records,
        }
        if data_records:
            info["table"] = data_records[0]["table_name"]
            info["data"] = data_records[0]["data"]
        else:
            info["data"] = {}
        return info

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
        if self._schema_query_pattern:
            schema_path = self._resolve_path(self._schema_query_pattern.format(qid=qid))
            if schema_path.exists():
                return self._schema_to_legacy_or_passthrough(self._read_json_any(schema_path, self._encoding))

        query = self._query_dict.get(str(qid))
        if not query:
            return []
        required_tables = set(query.get("tables") or [])
        required_attrs = set(query.get("attributes") or [])
        required_by_table: Dict[str, set[str]] = {}
        for attr in required_attrs:
            if "." not in attr:
                continue
            table, column = attr.split(".", 1)
            required_by_table.setdefault(table, set()).add(column)

        result = []
        for table_schema in self._schema_general:
            table_name = table_schema.get("Schema Name")
            if required_tables and table_name not in required_tables:
                continue
            attrs = table_schema.get("Attributes", [])
            wanted_attrs = required_by_table.get(str(table_name), set())
            if wanted_attrs:
                attrs = [
                    attr for attr in attrs
                    if attr.get("Attribute Name") in wanted_attrs
                ]
            result.append(
                {
                    "Schema Name": table_name,
                    "Description": table_schema.get("Description", ""),
                    "Attributes": attrs,
                }
            )
        return result

    def load_name_map(self, query_id: str | int | None = None, **_: Any) -> Dict[str, Any]:
        del query_id
        table_map = {
            schema["Schema Name"]: schema["Schema Name"]
            for schema in self._schema_general
            if schema.get("Schema Name")
        }
        attr_map: Dict[str, Dict[str, str]] = {}
        for schema in self._schema_general:
            table_name = schema.get("Schema Name")
            if not table_name:
                continue
            attr_map[str(table_name)] = {
                attr["Attribute Name"]: attr["Attribute Name"]
                for attr in schema.get("Attributes", [])
                if attr.get("Attribute Name")
            }
        return {"table": table_map, "attribute": attr_map}

    def refresh(self) -> None:
        reloaded = DataLoaderHFManifest(
            self.data_root,
            manifest=self._manifest_path,
            encoding=self._encoding,
        )
        self.__dict__.update(reloaded.__dict__)

    def _resolve_path(self, value: str | Path | None) -> Path:
        path = Path(value or "")
        if path.is_absolute():
            return path
        return (self.data_root / path).resolve()

    @staticmethod
    def _read_json(path: Path | str, encoding: str = "utf-8") -> Dict[str, Any]:
        resolved_path = Path(path)
        if not resolved_path.exists():
            return {}
        with resolved_path.open("r", encoding=encoding) as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _read_json_any(path: Path, encoding: str) -> Any:
        with path.open("r", encoding=encoding) as file:
            return json.load(file)

    @staticmethod
    def _clean_scalar(value: Any) -> Any:
        if pd.isna(value):
            return None
        return value

    @staticmethod
    def _normalize_query_records(
        raw: Dict[str, Any],
        *,
        schema_contract: Dict[str, Any] | None = None,
        dataset_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        records = raw.get("queries") if isinstance(raw, dict) else []
        if isinstance(records, list):
            normalized = [record for record in records if isinstance(record, dict)]
            if normalized:
                return normalized
            return build_default_query_records(schema_contract or {}, dataset_id=dataset_id)
        if isinstance(records, dict):
            normalized = []
            for query_id, record in records.items():
                if isinstance(record, dict):
                    item = dict(record)
                    item.setdefault("query_id", query_id)
                    normalized.append(item)
            if normalized:
                return normalized
            return build_default_query_records(schema_contract or {}, dataset_id=dataset_id)
        return build_default_query_records(schema_contract or {}, dataset_id=dataset_id)

    def _schema_to_legacy(self, raw_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        tables = raw_schema.get("tables") if isinstance(raw_schema, dict) else []
        result: List[Dict[str, Any]] = []
        if not isinstance(tables, list):
            return result
        for table in tables:
            if not isinstance(table, dict):
                continue
            table_name = str(table.get("name") or table.get("table_id") or "")
            attrs = []
            for column in table.get("columns") or []:
                if not isinstance(column, dict):
                    continue
                attrs.append(
                    {
                        "Attribute Name": str(column.get("name") or column.get("column_id") or ""),
                        "Description": str(column.get("description") or ""),
                        "column_id": str(column.get("column_id") or ""),
                        "type": str(column.get("type") or "string"),
                    }
                )
            result.append(
                {
                    "Schema Name": table_name,
                    "Description": str(table.get("description") or ""),
                    "Attributes": attrs,
                    "table_id": str(table.get("table_id") or normalize_identifier(table_name)),
                }
            )
        return result

    def _schema_to_legacy_or_passthrough(self, raw_schema: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_schema, list):
            return raw_schema
        if isinstance(raw_schema, dict) and raw_schema.get("schema_version") == "redd.schema.v1":
            return self._schema_to_legacy(raw_schema)
        if isinstance(raw_schema, dict) and "tables" in raw_schema:
            tables = raw_schema.get("tables")
            if isinstance(tables, dict):
                result = []
                for table_name, table_info in tables.items():
                    if not isinstance(table_info, dict):
                        continue
                    attrs = []
                    raw_attrs = table_info.get("attributes") or {}
                    if isinstance(raw_attrs, dict):
                        for attr_name, attr_info in raw_attrs.items():
                            description = (
                                attr_info.get("description", "")
                                if isinstance(attr_info, dict)
                                else str(attr_info or "")
                            )
                            attrs.append(
                                {
                                    "Attribute Name": str(attr_name),
                                    "Description": str(description),
                                }
                            )
                    result.append(
                        {
                            "Schema Name": str(table_name),
                            "Description": str(table_info.get("description", "")),
                            "Attributes": attrs,
                        }
                    )
                return result
        return []

    def _query_to_legacy(self, record: Dict[str, Any]) -> Dict[str, Any]:
        column_lookup = self._column_id_to_legacy_name()
        required_columns = [str(value) for value in record.get("required_columns") or []]
        attributes = [
            column_lookup.get(column_id, column_id)
            for column_id in required_columns
        ]
        required_tables = [str(value) for value in record.get("required_tables") or []]
        table_lookup = self._table_id_to_legacy_name()
        tables = [table_lookup.get(table_id, table_id) for table_id in required_tables]
        return {
            **record,
            "query": record.get("question") or record.get("query") or "",
            "sql": record.get("sql") or "",
            "tables": tables,
            "attributes": attributes,
        }

    def _table_id_to_legacy_name(self) -> Dict[str, str]:
        mapping = {}
        for schema in self._schema_general:
            table_id = schema.get("table_id") or normalize_identifier(schema.get("Schema Name"))
            mapping[str(table_id)] = str(schema.get("Schema Name"))
        return mapping

    def _column_id_to_legacy_name(self) -> Dict[str, str]:
        mapping = {}
        for schema in self._schema_general:
            table_name = str(schema.get("Schema Name"))
            for attr in schema.get("Attributes", []):
                column_id = str(attr.get("column_id") or "")
                attr_name = str(attr.get("Attribute Name"))
                if column_id:
                    mapping[column_id] = f"{table_name}.{attr_name}"
        return mapping

    def _build_ground_truth_index(self) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        if self._ground_truth_df.empty:
            return {}
        table_lookup = self._table_id_to_legacy_name()
        column_lookup: Dict[str, str] = {}
        for schema in self._schema_general:
            for attr in schema.get("Attributes", []):
                if attr.get("column_id"):
                    column_lookup[str(attr["column_id"])] = str(attr.get("Attribute Name"))

        grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for _, row in self._ground_truth_df.iterrows():
            doc_id = str(row.get("doc_id"))
            table_id = str(row.get("table_id"))
            record_id = str(row.get("record_id") or row.get("source_row_id") or "0")
            column_id = str(row.get("column_id"))
            table_name = table_lookup.get(table_id, table_id)
            column_name = column_lookup.get(column_id, str(row.get("column_name") or column_id))
            key = (doc_id, table_name, record_id)
            grouped.setdefault(key, {})[column_name] = self._clean_scalar(row.get("value"))

        by_doc: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for (doc_id, table_name, _record_id), values in grouped.items():
            by_doc.setdefault(doc_id, []).append((table_name, values))
        return by_doc
