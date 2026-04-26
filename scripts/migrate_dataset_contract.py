from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from redd.dataset_contract import validate_registry

DOCUMENT_COLUMNS = [
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
]

GROUND_TRUTH_COLUMNS = [
    "dataset_id",
    "doc_id",
    "record_id",
    "table_id",
    "column_id",
    "column_name",
    "value",
    "value_type",
    "source_row_id",
]


def normalize_identifier(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--apply", action="store_true", help="Replace the dataset directory after validation.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    tmp_root = dataset_root / ".redd_contract_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    report = migrate_all(dataset_root, tmp_root)
    registry_path = tmp_root / "manifest.yaml"
    validation_report = validate_registry(registry_path)
    report["validation"] = validation_report
    report["valid"] = validation_report["valid"]
    (tmp_root / "migration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not validation_report["valid"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    if args.apply:
        replace_dataset_root(dataset_root, tmp_root)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Migration generated at {tmp_root}. Re-run with --apply to replace dataset/.")
    return 0


def migrate_all(dataset_root: Path, output_root: Path) -> Dict[str, Any]:
    entries = discover_tasks(dataset_root)
    registry: Dict[str, Any] = {
        "schema_version": "redd.registry.v1",
        "collection_id": "redd",
        "format": "hf_parquet_registry",
        "datasets": {},
    }
    report: Dict[str, Any] = {
        "datasets": [],
        "warnings": [],
        "summary": {"canonical": 0, "derived": 0},
    }

    for entry in entries:
        dataset_id, kind = dataset_contract_id(entry)
        out_dir = output_root / kind / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        task_report = migrate_task(entry, dataset_id, kind, out_dir)
        report["datasets"].append(task_report)
        report["summary"][kind] += 1
        rel_manifest = (out_dir / "manifest.yaml").relative_to(output_root).as_posix()
        registry["datasets"][dataset_id] = {
            "kind": kind,
            "path": rel_manifest,
            "parents": entry.get("parents", []),
        }

    (output_root / "manifest.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return report


def discover_tasks(dataset_root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for source_root in sorted(dataset_root.glob("*_sqlite")):
        source = source_root.name.removesuffix("_sqlite")
        direct_task_dirs = [p for p in source_root.iterdir() if p.is_dir() and (p / "documents.db").exists()]
        if direct_task_dirs:
            dataset_name = source
            has_gt_schema = any(p.name == "gt_schema_task" for p in direct_task_dirs)
            for task_dir in sorted(direct_task_dirs):
                kind = "canonical" if task_dir.name == "gt_schema_task" or (not has_gt_schema and task_dir.name == "default_task") else "derived"
                entries.append(
                    {
                        "source": source,
                        "dataset_name": dataset_name,
                        "dataset_root": source_root,
                        "task_name": task_dir.name,
                        "task_dir": task_dir,
                        "documents_db": task_dir / "documents.db",
                        "gt_db": source_root / "gt_data.db",
                        "kind": kind,
                    }
                )
            continue

        for dataset_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
            task_dirs = [p for p in dataset_dir.iterdir() if p.is_dir() and (p / "documents.db").exists()]
            if task_dirs:
                has_gt_schema = any(p.name == "gt_schema_task" for p in task_dirs)
                for task_dir in sorted(task_dirs):
                    kind = "canonical" if task_dir.name == "gt_schema_task" or (not has_gt_schema and task_dir.name == "default_task") else "derived"
                    entries.append(
                        {
                            "source": source,
                            "dataset_name": dataset_dir.name,
                            "dataset_root": dataset_dir,
                            "task_name": task_dir.name,
                            "task_dir": task_dir,
                            "documents_db": task_dir / "documents.db",
                            "gt_db": dataset_dir / "gt_data.db",
                            "kind": kind,
                        }
                    )
            elif (dataset_dir / "default_task.db").exists():
                entries.append(
                    {
                        "source": source,
                        "dataset_name": dataset_dir.name,
                        "dataset_root": dataset_dir,
                        "task_name": "default_task",
                        "task_dir": dataset_dir,
                        "documents_db": dataset_dir / "default_task.db",
                        "gt_db": dataset_dir / "gt_data.db",
                        "kind": "canonical",
                    }
                )
    return entries


def dataset_contract_id(entry: Dict[str, Any]) -> Tuple[str, str]:
    source = normalize_identifier(entry["source"])
    dataset_name = normalize_identifier(entry["dataset_name"])
    base_id = f"{source}.{dataset_name}"
    kind = entry["kind"]
    if kind == "canonical":
        return base_id, kind
    return f"{base_id}.{normalize_identifier(entry['task_name'])}", kind


def migrate_task(entry: Dict[str, Any], dataset_id: str, kind: str, out_dir: Path) -> Dict[str, Any]:
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "metadata" / "query_sets").mkdir(parents=True, exist_ok=True)

    schema, schema_name_index = load_schema(entry["task_dir"], dataset_id)
    if not schema.get("tables") and entry["gt_db"].exists():
        schema = load_schema_from_gt_db(entry["gt_db"], dataset_id)
        schema_name_index = {
            normalize_identifier(table["name"]): table["name"]
            for table in schema.get("tables") or []
        }
    mappings = load_mappings(entry["task_dir"], schema)
    documents = load_documents(entry, dataset_id, schema_name_index)
    gt_rows, gt_report = build_ground_truth(entry, dataset_id, documents, schema, mappings)
    queries, query_report = load_queries(entry["task_dir"], dataset_id, schema)

    documents[DOCUMENT_COLUMNS].to_parquet(out_dir / "data" / "documents.parquet", index=False)
    pd.DataFrame(gt_rows, columns=GROUND_TRUTH_COLUMNS).to_parquet(
        out_dir / "data" / "ground_truth.parquet",
        index=False,
    )
    write_json(out_dir / "metadata" / "schema.json", schema)
    write_json(out_dir / "metadata" / "queries.json", queries)
    migrate_query_sets(entry["dataset_root"], out_dir / "metadata" / "query_sets")

    manifest = {
        "schema_version": "redd.manifest.v1",
        "dataset_id": dataset_id,
        "kind": kind,
        "version": "0.1.0",
        "source": {
            "legacy_dataset_root": entry["dataset_root"].as_posix(),
            "legacy_task": entry["task_name"],
        },
        "paths": {
            "documents": "data/documents.parquet",
            "ground_truth": "data/ground_truth.parquet",
            "schema": "metadata/schema.json",
            "queries": "metadata/queries.json",
        },
    }
    (out_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return {
        "dataset_id": dataset_id,
        "kind": kind,
        "legacy_task": str(entry["task_dir"]),
        "documents": int(len(documents)),
        "ground_truth_rows": int(len(gt_rows)),
        **gt_report,
        **query_report,
    }


def load_documents(entry: Dict[str, Any], dataset_id: str, schema_name_index: Dict[str, str]) -> pd.DataFrame:
    with sqlite3.connect(entry["documents_db"]) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        doc_mapping = load_document_mapping(conn)

    documents = []
    for row in rows:
        raw = dict(row)
        mapping = doc_mapping.get(str(raw.get("doc_id")), {})
        source_id = mapping.get("table_name") or raw.get("source_file")
        inferred_table = infer_table_name(source_id, schema_name_index)
        parent_doc_id = clean(raw.get("parent_doc_id"))
        source_row_id = clean(mapping.get("row_id")) or infer_row_id(parent_doc_id, inferred_table)
        is_chunked = raw.get("is_chunked")
        if is_chunked is None:
            is_chunked = bool(parent_doc_id)
        documents.append(
            {
                "dataset_id": dataset_id,
                "doc_id": str(raw.get("doc_id")),
                "doc_text": str(raw.get("doc_text") or ""),
                "source_id": clean(source_id),
                "source_table": inferred_table,
                "source_row_id": source_row_id,
                "parent_doc_id": parent_doc_id,
                "chunk_index": raw.get("chunk_index"),
                "is_chunked": bool(is_chunked),
                "split": "unspecified",
            }
        )

    df = pd.DataFrame(documents, columns=DOCUMENT_COLUMNS)
    return df


def load_document_mapping(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mapping'"
    ).fetchone()
    if not table_exists:
        return {}
    rows = conn.execute("SELECT doc_id, table_name, row_id FROM mapping").fetchall()
    return {
        str(row["doc_id"]): {
            "table_name": clean(row["table_name"]),
            "row_id": clean(row["row_id"]),
        }
        for row in rows
    }


def build_ground_truth(
    entry: Dict[str, Any],
    dataset_id: str,
    documents: pd.DataFrame,
    schema: Dict[str, Any],
    mappings: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    doc_info_path = entry["task_dir"] / "doc_info.json"
    if doc_info_path.exists():
        rows = build_ground_truth_from_doc_info(doc_info_path, dataset_id, schema)
        return rows, {"unmapped_documents": 0}

    gt_db = entry["gt_db"]
    if not gt_db.exists():
        return [], {"unmapped_documents": int(len(documents))}

    gt_tables = load_gt_tables(gt_db)
    fill_missing_source_rows(documents, gt_tables)
    table_map = mappings.get("table") or {}
    attr_map = mappings.get("attribute") or {}
    schema_tables = schema.get("tables") or []
    gt_rows: List[Dict[str, Any]] = []
    unmapped_documents = 0

    for _, doc in documents.iterrows():
        source_table = clean(doc.get("source_table"))
        source_row_id = clean(doc.get("source_row_id"))
        matched_any = False
        for table in schema_tables:
            task_table = str(table["name"])
            gt_table = table_map.get(task_table, task_table)
            single_table_schema = len(schema_tables) == 1
            if (
                source_table
                and normalize_identifier(source_table) != normalize_identifier(gt_table)
                and not single_table_schema
            ):
                continue
            gt_df = find_gt_table(gt_tables, gt_table)
            if gt_df is None or source_row_id is None:
                continue
            row = find_gt_row(gt_df, source_row_id)
            if row is None and single_table_schema:
                row = find_gt_row(gt_df, str(doc["doc_id"]).split("-", 1)[0])
            if row is None:
                continue
            matched_any = True
            table_id = table["table_id"]
            task_to_gt_attrs = attr_map.get(gt_table, {})
            for column in table.get("columns") or []:
                task_attr = column["name"]
                gt_attr = task_to_gt_attrs.get(task_attr, task_attr)
                if isinstance(gt_attr, list):
                    value = " ".join(str(row.get(attr)) for attr in gt_attr if attr in row and row.get(attr) is not None)
                else:
                    value = row.get(gt_attr)
                gt_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "doc_id": str(doc["doc_id"]),
                        "record_id": str(source_row_id),
                        "table_id": table_id,
                        "column_id": column["column_id"],
                        "column_name": task_attr,
                        "value": clean(value),
                        "value_type": "string",
                        "source_row_id": str(source_row_id),
                    }
                )
        if not matched_any:
            unmapped_documents += 1
    return gt_rows, {"unmapped_documents": unmapped_documents}


def build_ground_truth_from_doc_info(doc_info_path: Path, dataset_id: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = json.loads(doc_info_path.read_text(encoding="utf-8"))
    column_lookup = {
        (table["name"], column["name"]): (table["table_id"], column["column_id"])
        for table in schema.get("tables") or []
        for column in table.get("columns") or []
    }
    rows = []
    for doc_id, info in data.items():
        for record_index, record in enumerate(info.get("data_records") or []):
            table_name = record.get("table_name")
            values = record.get("data") or {}
            for column_name, value in values.items():
                ids = column_lookup.get((table_name, column_name))
                if not ids:
                    continue
                table_id, column_id = ids
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "doc_id": str(doc_id),
                        "record_id": str(record_index),
                        "table_id": table_id,
                        "column_id": column_id,
                        "column_name": str(column_name),
                        "value": clean(value),
                        "value_type": "string",
                        "source_row_id": str(record_index),
                    }
                )
    return rows


def load_schema(task_dir: Path, dataset_id: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    schema_json = task_dir / "schema.json"
    if schema_json.exists():
        raw = json.loads(schema_json.read_text(encoding="utf-8"))
        tables = schema_tables_from_json(raw)
    else:
        tables = schema_tables_from_db(task_dir / "schema.db")

    schema = {
        "schema_version": "redd.schema.v1",
        "dataset_id": dataset_id,
        "tables": tables,
        "relationships": [],
    }
    name_index = {normalize_identifier(table["name"]): table["name"] for table in tables}
    return schema, name_index


def schema_tables_from_json(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        result = []
        for table in raw:
            table_name = str(table.get("Schema Name") or table.get("name") or table.get("table_id"))
            attrs = []
            for attr in table.get("Attributes") or table.get("columns") or []:
                attr_name = str(attr.get("Attribute Name") or attr.get("name") or attr.get("column_id"))
                attrs.append(make_column(table_name, attr_name, attr.get("Description") or attr.get("description")))
            result.append(make_table(table_name, table.get("Description") or table.get("description"), attrs))
        return result

    if isinstance(raw, dict) and isinstance(raw.get("tables"), dict):
        result = []
        for table_name, table_info in raw["tables"].items():
            attrs = []
            raw_attrs = table_info.get("attributes") or {}
            if isinstance(raw_attrs, dict):
                for attr_name, attr_info in raw_attrs.items():
                    description = attr_info.get("description") if isinstance(attr_info, dict) else attr_info
                    attrs.append(make_column(table_name, attr_name, description))
            result.append(make_table(table_name, table_info.get("description"), attrs))
        return result

    if isinstance(raw, dict) and isinstance(raw.get("tables"), list):
        result = []
        for table in raw["tables"]:
            table_name = str(table.get("name") or table.get("table_id"))
            attrs = [
                make_column(table_name, column.get("name") or column.get("column_id"), column.get("description"))
                for column in table.get("columns") or []
            ]
            result.append(make_table(table_name, table.get("description"), attrs))
        return result
    return []


def schema_tables_from_db(schema_db: Path) -> List[Dict[str, Any]]:
    if not schema_db.exists():
        return []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    descriptions: Dict[str, str] = {}
    with sqlite3.connect(schema_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT table_name, attribute_name, description FROM schema_description ORDER BY table_name, attribute_name"
        ).fetchall()
    for row in rows:
        table_name = str(row["table_name"])
        attr_name = str(row["attribute_name"])
        description = clean(row["description"])
        if normalize_identifier(attr_name) in {"table", "description"}:
            descriptions[table_name] = description or ""
            continue
        grouped[table_name].append(make_column(table_name, attr_name, description))
    return [make_table(table_name, descriptions.get(table_name), attrs) for table_name, attrs in grouped.items()]


def load_schema_from_gt_db(gt_db: Path, dataset_id: str) -> Dict[str, Any]:
    tables = []
    with sqlite3.connect(gt_db) as conn:
        table_names = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        for table_name in table_names:
            columns = []
            for column in conn.execute(f'PRAGMA table_info("{table_name}")'):
                column_name = column[1]
                if column_name == "row_id":
                    continue
                columns.append(make_column(table_name, column_name, ""))
            tables.append(make_table(table_name, "", columns))
    return {
        "schema_version": "redd.schema.v1",
        "dataset_id": dataset_id,
        "tables": tables,
        "relationships": [],
    }


def make_table(table_name: Any, description: Any, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    table_name = str(table_name)
    return {
        "table_id": normalize_identifier(table_name),
        "name": table_name,
        "description": str(description or ""),
        "primary_key": ["row_id"],
        "columns": columns,
    }


def make_column(table_name: Any, attr_name: Any, description: Any) -> Dict[str, Any]:
    attr_name = str(attr_name)
    return {
        "column_id": f"{normalize_identifier(table_name)}.{normalize_identifier(attr_name)}",
        "name": attr_name,
        "type": "string",
        "description": str(description or ""),
        "nullable": True,
        "examples": [],
    }


def load_mappings(task_dir: Path, schema: Dict[str, Any]) -> Dict[str, Any]:
    schema_db = task_dir / "schema.db"
    table_map = {table["name"]: table["name"] for table in schema.get("tables") or []}
    attr_map = {
        table["name"]: {column["name"]: column["name"] for column in table.get("columns") or []}
        for table in schema.get("tables") or []
    }
    if schema_db.exists():
        with sqlite3.connect(schema_db) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts_table_name, ts_attribute_name, gt_table_name, gt_attribute_name FROM schema_mapping"
            ).fetchall()
        for row in rows:
            task_table = clean(row["ts_table_name"])
            gt_table = clean(row["gt_table_name"])
            task_attr = clean(row["ts_attribute_name"])
            gt_attr = clean(row["gt_attribute_name"])
            if task_table and gt_table:
                table_map[task_table] = gt_table
            if gt_table and task_attr and gt_attr:
                attr_map.setdefault(gt_table, {})[task_attr] = gt_attr
    return {"table": table_map, "attribute": attr_map}


def load_queries(task_dir: Path, dataset_id: str, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    query_path = task_dir / "queries.json"
    if not query_path.exists():
        return {"schema_version": "redd.queries.v1", "dataset_id": dataset_id, "queries": []}, {"dropped_query_columns": 0}
    raw = json.loads(query_path.read_text(encoding="utf-8"))
    raw_queries = raw.get("queries") if isinstance(raw, dict) and "queries" in raw else raw
    if not isinstance(raw_queries, dict):
        raw_queries = {}

    schema_columns = {
        column["column_id"]: (table["table_id"], table["name"], column["name"])
        for table in schema.get("tables") or []
        for column in table.get("columns") or []
    }
    schema_tables = {table["table_id"]: table["name"] for table in schema.get("tables") or []}
    dropped = 0
    queries = []
    for query_id, query in raw_queries.items():
        tables = []
        for table in query.get("tables") or []:
            table_id = normalize_identifier(table)
            if table_id in schema_tables:
                tables.append(table_id)
        columns = []
        for attr in query.get("attributes") or []:
            column_id = attribute_to_column_id(attr)
            if column_id in schema_columns:
                columns.append(column_id)
                table_id = schema_columns[column_id][0]
                if table_id not in tables:
                    tables.append(table_id)
            else:
                dropped += 1
        queries.append(
            {
                "query_id": str(query_id),
                "question": query.get("query") or query.get("question") or "",
                "sql": query.get("sql") or "",
                "required_tables": tables,
                "required_columns": columns,
                "output_columns": columns[:1],
                "tags": query.get("tags") or [],
                "difficulty": query.get("difficulty"),
            }
        )
    return {
        "schema_version": "redd.queries.v1",
        "dataset_id": dataset_id,
        "queries": queries,
    }, {"dropped_query_columns": dropped}


def migrate_query_sets(dataset_root: Path, query_sets_dir: Path) -> None:
    for path in sorted(dataset_root.glob("*.json")):
        if path.name in {"generated_queries.json"} or path.name.startswith("queries_proposed"):
            shutil.copy2(path, query_sets_dir / path.name)


def load_gt_tables(gt_db: Path) -> Dict[str, pd.DataFrame]:
    tables = {}
    with sqlite3.connect(gt_db) as conn:
        table_names = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        for table_name in table_names:
            tables[table_name] = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    return tables


def fill_missing_source_rows(documents: pd.DataFrame, gt_tables: Dict[str, pd.DataFrame]) -> None:
    if len(gt_tables) == 1:
        table_name, gt_df = next(iter(gt_tables.items()))
        missing_indexes = [
            idx for idx in documents.index
            if clean(documents.at[idx, "source_row_id"]) is None
        ]
        if missing_indexes and "row_id" in gt_df.columns and len(missing_indexes) <= len(gt_df):
            gt_row_ids = [str(value) for value in gt_df["row_id"].tolist()]
            for idx, row_id in zip(missing_indexes, gt_row_ids):
                documents.at[idx, "source_table"] = table_name
                documents.at[idx, "source_row_id"] = row_id

    for table_name, group in documents.groupby("source_table", dropna=False):
        gt_df = find_gt_table(gt_tables, table_name)
        if gt_df is None or "row_id" not in gt_df.columns:
            continue
        missing_indexes = [idx for idx in group.index if clean(documents.at[idx, "source_row_id"]) is None]
        if not missing_indexes:
            continue
        gt_row_ids = [str(value) for value in gt_df["row_id"].tolist()]
        if len(missing_indexes) > len(gt_row_ids):
            continue
        for idx, row_id in zip(missing_indexes, gt_row_ids):
            documents.at[idx, "source_row_id"] = row_id


def find_gt_table(gt_tables: Dict[str, pd.DataFrame], table_name: Any) -> pd.DataFrame | None:
    wanted = normalize_identifier(table_name)
    for name, df in gt_tables.items():
        if normalize_identifier(name) == wanted:
            return df
    return None


def find_gt_row(gt_df: pd.DataFrame, source_row_id: Any) -> Dict[str, Any] | None:
    wanted = str(source_row_id)
    if "row_id" in gt_df.columns:
        matches = gt_df[gt_df["row_id"].astype(str) == wanted]
        if not matches.empty:
            return matches.iloc[0].to_dict()

    prefixed = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)-(.+)$", wanted)
    if prefixed:
        key, value = prefixed.groups()
        for column in gt_df.columns:
            if normalize_identifier(column) == normalize_identifier(key):
                matches = gt_df[gt_df[column].astype(str) == value]
                if not matches.empty:
                    return matches.iloc[0].to_dict()

    for column in gt_df.columns:
        matches = gt_df[gt_df[column].astype(str) == wanted]
        if not matches.empty:
            return matches.iloc[0].to_dict()
    return None


def infer_table_name(source_id: Any, schema_name_index: Dict[str, str]) -> str | None:
    if source_id is None:
        return None
    stem = Path(str(source_id)).stem
    return schema_name_index.get(normalize_identifier(stem), stem)


def infer_row_id(parent_doc_id: Any, source_table: Any) -> str | None:
    parent = clean(parent_doc_id)
    if parent is None:
        return None
    table_norm = normalize_identifier(source_table)
    match = re.match(r"^(.+)-([0-9]+)$", str(parent))
    if match and normalize_identifier(match.group(1)) == table_norm:
        return match.group(2)
    return str(parent)


def attribute_to_column_id(attr: Any) -> str:
    text = str(attr)
    if "." in text:
        table, column = text.split(".", 1)
        return f"{normalize_identifier(table)}.{normalize_identifier(column)}"
    return normalize_identifier(text)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def replace_dataset_root(dataset_root: Path, tmp_root: Path) -> None:
    replacement_root = dataset_root / ".redd_contract_ready"
    if replacement_root.exists():
        shutil.rmtree(replacement_root)
    shutil.move(str(tmp_root), str(replacement_root))

    for child in list(dataset_root.iterdir()):
        if child.name == replacement_root.name:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for child in list(replacement_root.iterdir()):
        shutil.move(str(child), str(dataset_root / child.name))
    shutil.rmtree(replacement_root)


if __name__ == "__main__":
    raise SystemExit(main())
