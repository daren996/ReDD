"""Helpers for loaders backed by already-structured schema/query payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict, List

from .data_loader_hf_manifest import DEFAULT_EXTRACTION_QUERY_ID, normalize_identifier


def coerce_json_payload(value: Any, *, default: Any) -> Any:
    """Return a Python payload from a dict/list/string JSON value."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        return json.loads(text)
    return value


def schema_to_legacy_or_passthrough(raw_schema: Any) -> List[Dict[str, Any]]:
    """Normalize ReDD schema contracts into the legacy runtime schema list."""
    if isinstance(raw_schema, list):
        return [dict(item) for item in raw_schema if isinstance(item, Mapping)]
    if not isinstance(raw_schema, Mapping):
        return []

    tables = raw_schema.get("tables")
    if isinstance(tables, list):
        return [_table_contract_to_legacy(table) for table in tables if isinstance(table, Mapping)]

    if isinstance(tables, Mapping):
        result: List[Dict[str, Any]] = []
        for table_name, table_info in tables.items():
            if not isinstance(table_info, Mapping):
                continue
            attrs = []
            raw_attrs = table_info.get("attributes") or table_info.get("columns") or {}
            if isinstance(raw_attrs, Mapping):
                for attr_name, attr_info in raw_attrs.items():
                    description = (
                        attr_info.get("description", "")
                        if isinstance(attr_info, Mapping)
                        else str(attr_info or "")
                    )
                    attrs.append(
                        {
                            "Attribute Name": str(attr_name),
                            "Description": str(description),
                            "type": (
                                str(attr_info.get("type"))
                                if isinstance(attr_info, Mapping) and attr_info.get("type")
                                else "string"
                            ),
                        }
                    )
            elif isinstance(raw_attrs, list):
                attrs = [
                    _column_contract_to_legacy(column, table_id=normalize_identifier(table_name))
                    for column in raw_attrs
                    if isinstance(column, Mapping)
                ]
            result.append(
                {
                    "Schema Name": str(table_info.get("name") or table_name),
                    "Description": str(table_info.get("description") or ""),
                    "Attributes": attrs,
                    "table_id": str(table_info.get("table_id") or normalize_identifier(table_name)),
                }
            )
        return result

    return []


def normalize_query_records(
    raw_queries: Any,
    *,
    schema_general: List[Dict[str, Any]],
    dataset_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Normalize supported query payload shapes into runtime query records."""
    payload = coerce_json_payload(raw_queries, default={})

    records: list[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("queries"), list):
        records = list(payload.get("queries") or [])
    elif isinstance(payload, Mapping) and isinstance(payload.get("queries"), Mapping):
        records = []
        for query_id, record in payload["queries"].items():
            if isinstance(record, Mapping):
                item = dict(record)
                item.setdefault("query_id", query_id)
                records.append(item)
    elif isinstance(payload, Mapping) and _looks_like_single_query(payload):
        records = [payload]
    elif isinstance(payload, Mapping):
        records = []
        for query_id, record in payload.items():
            if isinstance(record, Mapping):
                item = dict(record)
                item.setdefault("query_id", query_id)
                records.append(item)
    else:
        records = []

    normalized = [
        _normalize_query_record(record)
        for record in records
        if isinstance(record, Mapping)
    ]
    if normalized:
        return normalized
    return build_default_query_records(schema_general, dataset_id=dataset_id)


def build_default_query_records(
    schema_general: List[Dict[str, Any]],
    *,
    dataset_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Build a default query that extracts every table/attribute in the schema."""
    required_tables: List[str] = []
    required_columns: List[str] = []
    for schema in schema_general:
        table_name = str(schema.get("Schema Name") or "")
        table_id = str(schema.get("table_id") or normalize_identifier(table_name))
        if table_id:
            required_tables.append(table_id)
        for attr in schema.get("Attributes") or []:
            if not isinstance(attr, Mapping):
                continue
            attr_name = str(attr.get("Attribute Name") or "")
            column_id = str(attr.get("column_id") or "")
            if not column_id and table_id and attr_name:
                column_id = f"{table_id}.{attr_name}"
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


def query_to_legacy(record: Dict[str, Any], schema_general: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a normalized query record into the current extraction query shape."""
    table_lookup = table_id_to_legacy_name(schema_general)
    column_lookup = column_id_to_legacy_name(schema_general)
    required_tables = [str(value) for value in record.get("required_tables") or []]
    required_columns = [str(value) for value in record.get("required_columns") or []]
    return {
        **record,
        "query": record.get("question") or record.get("query") or "",
        "sql": record.get("sql") or "",
        "tables": [table_lookup.get(table_id, table_id) for table_id in required_tables],
        "attributes": [
            column_lookup.get(column_id, column_id)
            for column_id in required_columns
        ],
    }


def schema_for_query(
    schema_general: List[Dict[str, Any]],
    query: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Filter a general schema by a legacy query record."""
    if not query:
        return []

    required_tables = set(query.get("tables") or [])
    required_attrs = set(query.get("attributes") or [])
    required_by_table: Dict[str, set[str]] = {}
    for attr in required_attrs:
        if "." not in str(attr):
            continue
        table, column = str(attr).split(".", 1)
        required_by_table.setdefault(table, set()).add(column)

    result: List[Dict[str, Any]] = []
    for table_schema in schema_general:
        table_name = table_schema.get("Schema Name")
        if required_tables and table_name not in required_tables:
            continue
        attrs = table_schema.get("Attributes", [])
        wanted_attrs = required_by_table.get(str(table_name), set())
        if wanted_attrs:
            attrs = [
                attr for attr in attrs
                if isinstance(attr, Mapping) and attr.get("Attribute Name") in wanted_attrs
            ]
        result.append(
            {
                "Schema Name": table_name,
                "Description": table_schema.get("Description", ""),
                "Attributes": attrs,
                **({"table_id": table_schema["table_id"]} if table_schema.get("table_id") else {}),
            }
        )
    return result


def identity_name_map(schema_general: List[Dict[str, Any]]) -> Dict[str, Any]:
    table_map = {
        schema["Schema Name"]: schema["Schema Name"]
        for schema in schema_general
        if schema.get("Schema Name")
    }
    attr_map: Dict[str, Dict[str, str]] = {}
    for schema in schema_general:
        table_name = schema.get("Schema Name")
        if not table_name:
            continue
        attr_map[str(table_name)] = {
            attr["Attribute Name"]: attr["Attribute Name"]
            for attr in schema.get("Attributes", [])
            if isinstance(attr, Mapping) and attr.get("Attribute Name")
        }
    return {"table": table_map, "attribute": attr_map}


def table_id_to_legacy_name(schema_general: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for schema in schema_general:
        table_name = str(schema.get("Schema Name") or "")
        table_id = str(schema.get("table_id") or normalize_identifier(table_name))
        if table_id:
            mapping[table_id] = table_name
    return mapping


def column_id_to_legacy_name(schema_general: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for schema in schema_general:
        table_name = str(schema.get("Schema Name") or "")
        table_id = str(schema.get("table_id") or normalize_identifier(table_name))
        for attr in schema.get("Attributes", []):
            if not isinstance(attr, Mapping):
                continue
            attr_name = str(attr.get("Attribute Name") or "")
            column_id = str(attr.get("column_id") or "")
            if column_id:
                mapping[column_id] = f"{table_name}.{attr_name}"
            if table_id and attr_name:
                mapping.setdefault(f"{table_id}.{attr_name}", f"{table_name}.{attr_name}")
    return mapping


def _table_contract_to_legacy(table: Mapping[str, Any]) -> Dict[str, Any]:
    table_id = str(table.get("table_id") or normalize_identifier(table.get("name") or ""))
    table_name = str(table.get("name") or table.get("table_id") or "")
    return {
        "Schema Name": table_name,
        "Description": str(table.get("description") or ""),
        "Attributes": [
            _column_contract_to_legacy(column, table_id=table_id)
            for column in table.get("columns") or []
            if isinstance(column, Mapping)
        ],
        "table_id": table_id or normalize_identifier(table_name),
    }


def _column_contract_to_legacy(column: Mapping[str, Any], *, table_id: str) -> Dict[str, Any]:
    attr_name = str(column.get("name") or column.get("column_id") or "")
    column_id = str(column.get("column_id") or "")
    if not column_id and table_id and attr_name:
        column_id = f"{table_id}.{attr_name}"
    return {
        "Attribute Name": attr_name,
        "Description": str(column.get("description") or ""),
        "column_id": column_id,
        "type": str(column.get("type") or "string"),
    }


def _normalize_query_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("query_id", normalized.get("id"))
    normalized.setdefault("question", normalized.get("natural_language"))
    golden_schema = normalized.get("golden_schema")
    if isinstance(golden_schema, Mapping):
        normalized.setdefault("required_tables", golden_schema.get("tables") or [])
        normalized.setdefault("required_columns", golden_schema.get("columns") or [])
    if not normalized.get("query_id"):
        normalized["query_id"] = DEFAULT_EXTRACTION_QUERY_ID
    return normalized


def _looks_like_single_query(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "query_id",
            "id",
            "question",
            "query",
            "sql",
            "required_tables",
            "required_columns",
            "golden_schema",
        )
    )
