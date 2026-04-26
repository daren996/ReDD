"""Validation helpers for the ReDD dataset contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

DOCUMENT_COLUMNS = {
    "dataset_id",
    "doc_id",
    "doc_text",
    "source_id",
    "source_table",
    "source_row_id",
    "parent_doc_id",
    "chunk_index",
    "is_chunked",
    "split",
}

GROUND_TRUTH_COLUMNS = {
    "dataset_id",
    "doc_id",
    "record_id",
    "table_id",
    "column_id",
    "column_name",
    "value",
    "value_type",
    "source_row_id",
}


def validate_registry(registry_path: str | Path) -> Dict[str, Any]:
    registry_path = Path(registry_path).expanduser().resolve()
    with registry_path.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file) or {}

    errors: List[str] = []
    datasets = registry.get("datasets") or {}
    if not isinstance(datasets, dict):
        errors.append("Root manifest `datasets` must be a mapping.")
        datasets = {}

    checked = []
    for dataset_id, entry in datasets.items():
        if not isinstance(entry, dict):
            errors.append(f"Dataset `{dataset_id}` registry entry must be a mapping.")
            continue
        manifest_path = (registry_path.parent / str(entry.get("path", ""))).resolve()
        result = validate_dataset_manifest(manifest_path)
        checked.append({"dataset_id": dataset_id, **result})
        errors.extend(f"{dataset_id}: {error}" for error in result["errors"])

    return {
        "registry": str(registry_path),
        "datasets_checked": len(checked),
        "valid": not errors,
        "errors": errors,
        "datasets": checked,
    }


def validate_dataset_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    errors: List[str] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file) or {}

    root = manifest_path.parent
    paths = manifest.get("paths") or {}
    resolved = {
        key: (root / str(paths.get(key, ""))).resolve()
        for key in ("documents", "ground_truth", "schema", "queries")
    }
    for key, path in resolved.items():
        if not path.exists():
            errors.append(f"Missing `{key}` resource: {path}")

    documents = _safe_read_parquet(resolved["documents"], errors, "documents")
    ground_truth = _safe_read_parquet(resolved["ground_truth"], errors, "ground_truth")
    schema = _safe_read_json(resolved["schema"], errors, "schema")
    queries = _safe_read_json(resolved["queries"], errors, "queries")

    if documents is not None:
        _check_columns(documents.columns, DOCUMENT_COLUMNS, "documents", errors)
    if ground_truth is not None:
        _check_columns(ground_truth.columns, GROUND_TRUTH_COLUMNS, "ground_truth", errors)

    schema_columns = _schema_column_ids(schema)
    schema_tables = _schema_table_ids(schema)
    doc_ids = set(str(value) for value in documents["doc_id"].tolist()) if documents is not None and "doc_id" in documents else set()

    if queries:
        for query in queries.get("queries") or []:
            query_id = query.get("query_id")
            for table_id in query.get("required_tables") or []:
                if table_id not in schema_tables:
                    errors.append(f"Query `{query_id}` references unknown table `{table_id}`.")
            for column_id in query.get("required_columns") or []:
                if column_id not in schema_columns:
                    errors.append(f"Query `{query_id}` references unknown column `{column_id}`.")

    if ground_truth is not None:
        for column_id in sorted(set(str(value) for value in ground_truth["column_id"].dropna().tolist())):
            if column_id not in schema_columns:
                errors.append(f"Ground truth references unknown column `{column_id}`.")
        for doc_id in sorted(set(str(value) for value in ground_truth["doc_id"].dropna().tolist())):
            if doc_id not in doc_ids:
                errors.append(f"Ground truth references unknown doc_id `{doc_id}`.")

    return {
        "manifest": str(manifest_path),
        "dataset_id": manifest.get("dataset_id"),
        "valid": not errors,
        "errors": errors,
    }


def print_validation_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _safe_read_parquet(path: Path, errors: List[str], label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - message path matters more than type
        errors.append(f"Failed to read `{label}` parquet {path}: {exc}")
        return None


def _safe_read_json(path: Path, errors: List[str], label: str) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        errors.append(f"Failed to read `{label}` json {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"`{label}` json must be an object.")
        return {}
    return data


def _check_columns(columns: Iterable[str], required: set[str], label: str, errors: List[str]) -> None:
    missing = sorted(required.difference(set(columns)))
    if missing:
        errors.append(f"`{label}` is missing columns: {missing}")


def _schema_column_ids(schema: Dict[str, Any]) -> set[str]:
    ids = set()
    for table in schema.get("tables") or []:
        for column in table.get("columns") or []:
            column_id = column.get("column_id")
            if column_id:
                ids.add(str(column_id))
    return ids


def _schema_table_ids(schema: Dict[str, Any]) -> set[str]:
    ids = set()
    for table in schema.get("tables") or []:
        table_id = table.get("table_id")
        if table_id:
            ids.add(str(table_id))
    return ids
