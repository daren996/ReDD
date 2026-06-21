from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value or "").lower())


def _read_mapping(documents_db: Path) -> dict[str, dict[str, str]]:
    with sqlite3.connect(documents_db) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mapping'"
        ).fetchone()
        if not exists:
            return {}
        rows = conn.execute("SELECT doc_id, table_name, row_id FROM mapping").fetchall()
    return {
        str(row["doc_id"]): {
            "table_name": str(row["table_name"]),
            "row_id": str(row["row_id"]),
        }
        for row in rows
    }


def _read_gt_tables(gt_db: Path) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(gt_db) as conn:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return {name: pd.read_sql_query(f'SELECT * FROM "{name}"', conn) for name in names}


def _legacy_dataset_root(current_source: str | None, legacy_dataset_root: Path) -> Path | None:
    if not current_source:
        return None
    marker = "/dataset/"
    text = str(current_source)
    if marker in text:
        suffix = text.split(marker, 1)[1]
        return legacy_dataset_root / suffix
    return None


def _schema_indexes(schema: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    table_by_compact: dict[str, dict[str, Any]] = {}
    column_by_table: dict[str, dict[str, str]] = {}
    for table in schema.get("tables") or []:
        table_id = str(table.get("table_id") or table.get("name") or "")
        table_key = _compact(table_id) or _compact(table.get("name"))
        if not table_key:
            continue
        table_by_compact[table_key] = table
        column_index: dict[str, str] = {}
        for column in table.get("columns") or []:
            name = str(column.get("name") or "")
            column_id = str(column.get("column_id") or "")
            if name:
                column_index[_compact(name)] = name
            if "." in column_id:
                column_index[_compact(column_id.split(".", 1)[1])] = name
        column_by_table[table_key] = column_index
    return table_by_compact, column_by_table


def _legacy_column_index(gt_df: pd.DataFrame) -> dict[str, str]:
    return {_compact(column): str(column) for column in gt_df.columns if column != "row_id"}


def _row_by_id(gt_df: pd.DataFrame, row_id: str) -> dict[str, Any] | None:
    if "row_id" not in gt_df.columns:
        return None
    matches = gt_df[gt_df["row_id"].astype(str) == str(row_id)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def repair_dataset(
    dataset_dir: Path,
    *,
    legacy_dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    dataset_id = str(manifest.get("dataset_id") or dataset_dir.name)
    source_root = _legacy_dataset_root(
        ((manifest.get("source") or {}).get("legacy_dataset_root")),
        legacy_dataset_root,
    )
    task_name = str((manifest.get("source") or {}).get("legacy_task") or "gt_schema_task")
    if source_root is None:
        raise ValueError(f"{dataset_id}: manifest has no usable legacy_dataset_root")

    documents_db = source_root / task_name / "documents.db"
    gt_db = source_root / "gt_data.db"
    if not documents_db.exists() or not gt_db.exists():
        raise FileNotFoundError(
            f"{dataset_id}: missing legacy inputs documents_db={documents_db} gt_db={gt_db}"
        )

    mapping = _read_mapping(documents_db)
    gt_tables = _read_gt_tables(gt_db)
    gt_tables_by_key = {_compact(name): (name, df) for name, df in gt_tables.items()}

    paths = manifest.get("paths") or {}
    docs_path = dataset_dir / paths.get("documents", "data/documents.parquet")
    schema_path = dataset_dir / paths.get("schema", "metadata/schema.json")
    docs = pd.read_parquet(docs_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    table_by_key, _column_by_table = _schema_indexes(schema)

    fixed_docs = docs.copy()
    rows: list[dict[str, Any]] = []
    missing_mapping = missing_table = missing_row = missing_column = 0
    repaired_docs = 0

    for idx, doc in fixed_docs.iterrows():
        doc_id = str(doc["doc_id"])
        mapped = mapping.get(doc_id)
        if not mapped:
            missing_mapping += 1
            continue
        legacy_table = mapped["table_name"]
        row_id = mapped["row_id"]
        table_key = _compact(legacy_table)
        schema_table = table_by_key.get(table_key)
        gt_table = gt_tables_by_key.get(table_key)
        fixed_docs.at[idx, "source_table"] = str(schema_table.get("table_id") if schema_table else legacy_table)
        fixed_docs.at[idx, "source_row_id"] = row_id
        if schema_table is None or gt_table is None:
            missing_table += 1
            continue
        _gt_name, gt_df = gt_table
        gt_row = _row_by_id(gt_df, row_id)
        if gt_row is None:
            missing_row += 1
            continue
        legacy_cols = _legacy_column_index(gt_df)
        repaired_docs += 1
        table_id = str(schema_table.get("table_id") or schema_table.get("name"))
        for column in schema_table.get("columns") or []:
            column_name = str(column.get("name") or "")
            column_id = str(column.get("column_id") or f"{table_id}.{column_name}")
            legacy_col = legacy_cols.get(_compact(column_name))
            value = gt_row.get(legacy_col) if legacy_col else None
            if legacy_col is None:
                missing_column += 1
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "doc_id": doc_id,
                    "record_id": row_id,
                    "table_id": table_id,
                    "column_id": column_id,
                    "column_name": column_name,
                    "value": _clean(value),
                    "value_type": "string",
                    "source_row_id": row_id,
                }
            )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(dataset_dir, output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    fixed_docs.to_parquet(data_dir / "documents.parquet", index=False)
    pd.DataFrame(rows).to_parquet(data_dir / "ground_truth.parquet", index=False)
    return {
        "dataset_id": dataset_id,
        "documents": int(len(fixed_docs)),
        "ground_truth_rows": int(len(rows)),
        "repaired_docs": int(repaired_docs),
        "missing_mapping": int(missing_mapping),
        "missing_table": int(missing_table),
        "missing_row": int(missing_row),
        "missing_column": int(missing_column),
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/canonical")
    parser.add_argument("--legacy-dataset-root", default="/Users/drchao/CursorSpace/ReDD/ReDD_Dev/dataset")
    parser.add_argument("--output-root", default="dataset/repaired_canonical")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="Replace dataset directories in place after writing repaired copies.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    legacy_root = Path(args.legacy_dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    dataset_names = args.dataset or [path.name for path in sorted(dataset_root.iterdir()) if path.is_dir()]

    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset_name in dataset_names:
        src = dataset_root / dataset_name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = output_root / dataset_name
        report = repair_dataset(src, legacy_dataset_root=legacy_root, output_dir=dst)
        reports.append(report)
        if args.apply:
            backup = src.with_name(src.name + ".pre_gt_alignment_repair")
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(src, backup)
            shutil.rmtree(src)
            shutil.copytree(dst, src)

    report_payload = {"datasets": reports, "applied": bool(args.apply)}
    (output_root / "repair_report.json").write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
