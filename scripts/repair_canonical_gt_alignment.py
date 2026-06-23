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


TABLE_ALIASES: dict[str, dict[str, str]] = {
    "spider.college_2.course_teaches_instructor": {
        "course": "course_information",
        "instructor": "instructors",
        "teaches": "course_instructor_assignments",
    },
    "spider.flight_4.routes_airports_airlines": {
        "airlines": "airlines",
        "airports": "airports",
        "routes": "flight_routes",
    },
    "spider.wine_1.wine_appellations": {
        "appellations": "appellation",
        "wine": "wines",
    },
}


COLUMN_ALIASES: dict[str, dict[tuple[str, str], str]] = {
    "spider.college_2.course_teaches_instructor": {
        ("course_information", "course_title"): "title",
        ("course_information", "department"): "dept_name",
        ("course_instructor_assignments", "offering_year"): "year",
        ("instructors", "annual_salary"): "salary",
        ("instructors", "department"): "dept_name",
        ("instructors", "instructor_name"): "name",
    },
    "spider.flight_4.routes_airports_airlines": {
        ("airlines", "airline_name"): "name",
        ("airlines", "iata_code"): "iata",
        ("airlines", "icao_code"): "icao",
        ("airports", "airport_name"): "name",
        ("airports", "iata_code"): "iata",
        ("airports", "icao_code"): "icao",
        ("airports", "latitude"): "y",
        ("airports", "longitude"): "x",
        ("flight_routes", "arrival_airport_code"): "dst_ap",
        ("flight_routes", "arrival_airport_name"): "dst_ap_name",
        ("flight_routes", "arrival_country"): "dst_ap_country",
        ("flight_routes", "departure_airport_code"): "src_ap",
        ("flight_routes", "departure_airport_name"): "src_ap_name",
        ("flight_routes", "departure_country"): "src_ap_country",
    },
    "spider.wine_1.wine_appellations": {
        ("appellation", "appellation"): "Appelation",
        ("appellation", "ava_status"): "isAVA",
        ("appellation", "main_location"): "State",
        ("appellation", "region_name"): "Area",
        ("wines", "appellation"): "Appelation",
        ("wines", "drinking_window"): "Drink",
        ("wines", "grape_variety"): "Grape",
        ("wines", "price"): "Price",
        ("wines", "production_volume"): "Cases",
        ("wines", "region"): "State",
        ("wines", "score"): "Score",
        ("wines", "vintage_year"): "Year",
        ("wines", "wine_name"): "Name",
        ("wines", "winery_name"): "Winery",
    },
    "galois.fortune.default_task": {
        ("fortune500_companies", "best_companies_to_work_for"): "Best_Companies_to_Work_For (boolean)",
        ("fortune500_companies", "founder_is_ceo"): "Founder_is_CEO (boolean)",
        ("fortune500_companies", "is_female_ceo"): "Is_FemaleCEO (boolean)",
        ("fortune500_companies", "is_profitable"): "Is_Profitable (boolean)",
        ("fortune500_companies", "number_of_employees"): "Number_of_employees (integer)",
    },
}


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


def _resolve_schema_table(
    dataset_id: str,
    legacy_table: str,
    table_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    table_key = _compact(legacy_table)
    alias = TABLE_ALIASES.get(dataset_id, {}).get(table_key)
    if alias:
        return table_by_key.get(_compact(alias))
    return table_by_key.get(table_key)


def _resolve_legacy_column(
    dataset_id: str,
    table_id: str,
    column_name: str,
    legacy_cols: dict[str, str],
) -> str | None:
    alias = COLUMN_ALIASES.get(dataset_id, {}).get((table_id, column_name))
    if alias:
        resolved = legacy_cols.get(_compact(alias))
        if resolved:
            return resolved
    return legacy_cols.get(_compact(column_name))


def _original_ground_truth_lookup(
    dataset_dir: Path,
    manifest: dict[str, Any],
) -> dict[tuple[str, str, str], Any]:
    paths = manifest.get("paths") or {}
    gt_path = dataset_dir / paths.get("ground_truth", "data/ground_truth.parquet")
    if not gt_path.exists():
        return {}
    original = pd.read_parquet(gt_path)
    lookup: dict[tuple[str, str, str], Any] = {}
    for _, row in original.iterrows():
        doc_id = str(row.get("doc_id"))
        table_id = str(row.get("table_id"))
        column_name = str(row.get("column_name") or "")
        if not column_name:
            column_id = str(row.get("column_id") or "")
            column_name = column_id.split(".", 1)[1] if "." in column_id else column_id
        lookup[(doc_id, table_id, column_name)] = row.get("value")
    return lookup


def _original_ground_truth_scope(
    dataset_dir: Path,
    manifest: dict[str, Any],
) -> dict[tuple[str, str], set[str]]:
    paths = manifest.get("paths") or {}
    gt_path = dataset_dir / paths.get("ground_truth", "data/ground_truth.parquet")
    if not gt_path.exists():
        return {}
    original = pd.read_parquet(gt_path)
    scope: dict[tuple[str, str], set[str]] = {}
    for _, row in original.iterrows():
        doc_id = str(row.get("doc_id"))
        table_id = str(row.get("table_id"))
        column_name = str(row.get("column_name") or "")
        if not column_name:
            column_id = str(row.get("column_id") or "")
            column_name = column_id.split(".", 1)[1] if "." in column_id else column_id
        if column_name:
            scope.setdefault((doc_id, table_id), set()).add(column_name)
    return scope


def repair_dataset(
    dataset_dir: Path,
    *,
    legacy_dataset_root: Path,
    output_dir: Path,
    preserve_ground_truth_scope: bool = False,
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
    original_lookup = _original_ground_truth_lookup(dataset_dir, manifest)
    original_scope = (
        _original_ground_truth_scope(dataset_dir, manifest)
        if preserve_ground_truth_scope
        else {}
    )

    fixed_docs = docs.copy()
    rows: list[dict[str, Any]] = []
    missing_mapping = missing_table = missing_row = missing_column = 0
    fallback_column = 0
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
        schema_table = _resolve_schema_table(dataset_id, legacy_table, table_by_key)
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
        columns = list(schema_table.get("columns") or [])
        if preserve_ground_truth_scope:
            scoped_column_names = original_scope.get((doc_id, table_id), set())
            if not scoped_column_names:
                continue
            columns = [
                column
                for column in columns
                if str(column.get("name") or "") in scoped_column_names
            ]
        for column in columns:
            column_name = str(column.get("name") or "")
            column_id = str(column.get("column_id") or f"{table_id}.{column_name}")
            legacy_col = _resolve_legacy_column(dataset_id, table_id, column_name, legacy_cols)
            if legacy_col:
                value = gt_row.get(legacy_col)
            else:
                fallback_key = (doc_id, table_id, column_name)
                value = original_lookup.get(fallback_key)
                if fallback_key in original_lookup:
                    fallback_column += 1
                else:
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
        "fallback_column": int(fallback_column),
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/canonical")
    parser.add_argument("--legacy-dataset-root", default="/Users/drchao/CursorSpace/ReDD/ReDD_Dev/dataset")
    parser.add_argument("--output-root", default="dataset/repaired_canonical")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="Replace dataset directories in place after writing repaired copies.")
    parser.add_argument(
        "--preserve-ground-truth-scope",
        action="store_true",
        help="Only rewrite ground-truth cells that were already present in the source dataset.",
    )
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
        report = repair_dataset(
            src,
            legacy_dataset_root=legacy_root,
            output_dir=dst,
            preserve_ground_truth_scope=args.preserve_ground_truth_scope,
        )
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
