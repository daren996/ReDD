"""Profile and propose ReDD query sets by document coverage.

This script is intentionally read-only with respect to datasets. It writes
reports under ``outputs/query_curation`` so query manifests can be curated after
inspection instead of mutating source manifests during exploration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from redd.core.data_loader import create_data_loader
from redd.core.utils.data_split import split_doc_ids
from redd.core.utils.utils import is_null
from redd.exp.evaluation import EvalDataExtraction

DEFAULT_BUCKETS = (0.03, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    root: Path
    manifest: Path


@dataclass(frozen=True)
class JoinPair:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    shared_values: int
    left_values: int
    right_values: int
    score: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover_datasets(dataset_root: Path) -> list[DatasetRef]:
    refs: list[DatasetRef] = []
    for manifest in sorted(dataset_root.glob("**/manifest.yaml")):
        try:
            payload = _read_yaml(manifest)
        except Exception:
            continue
        dataset_id = str(payload.get("dataset_id") or manifest.parent.name)
        refs.append(DatasetRef(dataset_id=dataset_id, root=manifest.parent, manifest=manifest))
    return refs


def _loader(ref: DatasetRef):
    return create_data_loader(
        ref.root,
        loader_type="hf_manifest",
        loader_config={"manifest": ref.manifest.name},
    )


def _required_attrs_by_table(
    evaluator: EvalDataExtraction,
    loader: Any,
    query_id: str,
    query_info: dict[str, Any],
) -> dict[str, list[str]]:
    return evaluator._required_attrs_by_table(loader, query_id, query_info)


def _count_required_cells(
    evaluator: EvalDataExtraction,
    loader: Any,
    doc_ids: Iterable[str],
    required_by_table: dict[str, list[str]],
    answer_doc_ids_by_table: dict[str, set[str]] | None,
) -> tuple[int, int, dict[str, int]]:
    relevant_docs: set[str] = set()
    required_cells = 0
    docs_by_table: dict[str, set[str]] = {table: set() for table in required_by_table}
    for doc_id in doc_ids:
        records = evaluator._query_required_gt_records(
            loader,
            str(doc_id),
            required_by_table,
            answer_doc_ids_by_table=answer_doc_ids_by_table,
        )
        if records:
            relevant_docs.add(str(doc_id))
        for record in records:
            table = str(record["table"])
            docs_by_table.setdefault(table, set()).add(str(doc_id))
            data = record.get("data") if isinstance(record.get("data"), dict) else {}
            for attr in required_by_table.get(table, []):
                if not is_null(data.get(attr)):
                    required_cells += 1
    return len(relevant_docs), required_cells, {table: len(ids) for table, ids in docs_by_table.items()}


def profile_query(
    *,
    dataset: DatasetRef,
    loader: Any,
    evaluator: EvalDataExtraction,
    query_id: str,
    query_info: dict[str, Any],
    train_count: int,
    split_strategy: str,
    split_seed: int,
    min_docs: int,
    min_cells: int,
) -> dict[str, Any]:
    all_doc_ids = [str(doc_id) for doc_id in loader.doc_ids]
    _, eval_doc_ids = split_doc_ids(
        all_doc_ids,
        train_count,
        strategy=split_strategy,
        seed=split_seed,
    )
    required_by_table = _required_attrs_by_table(evaluator, loader, query_id, query_info)
    sql = str(query_info.get("sql") or "").strip()
    answer_all = evaluator._answer_doc_ids_by_table(
        loader=loader,
        query_info=query_info,
        eval_doc_ids=all_doc_ids,
        required_by_table=required_by_table,
    )
    answer_eval = evaluator._answer_doc_ids_by_table(
        loader=loader,
        query_info=query_info,
        eval_doc_ids=eval_doc_ids,
        required_by_table=required_by_table,
    )
    provenance_status = "not_sql" if not sql else ("ok" if answer_eval is not None else "fallback_all_required_rows")

    all_docs, all_cells, all_by_table = _count_required_cells(
        evaluator,
        loader,
        all_doc_ids,
        required_by_table,
        answer_all,
    )
    eval_docs, eval_cells, eval_by_table = _count_required_cells(
        evaluator,
        loader,
        eval_doc_ids,
        required_by_table,
        answer_eval,
    )
    eval_total = len(eval_doc_ids)
    all_total = len(all_doc_ids)
    keep = bool(eval_docs >= min_docs and eval_cells >= min_cells)
    if not required_by_table:
        status = "remove_no_required_attrs"
    elif not keep:
        status = "remove_too_few_docs_or_cells"
    elif provenance_status == "fallback_all_required_rows":
        status = "review_sql_provenance"
    else:
        status = "keep"
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_root": str(dataset.root),
        "query_id": query_id,
        "question": query_info.get("question") or query_info.get("query") or "",
        "sql": sql,
        "is_sql": bool(sql),
        "required_tables": sorted(required_by_table),
        "required_columns": [
            f"{table}.{attr}"
            for table, attrs in sorted(required_by_table.items())
            for attr in attrs
        ],
        "train_count": train_count,
        "split_strategy": split_strategy,
        "split_seed": split_seed,
        "all_docs": all_total,
        "eval_docs": eval_total,
        "all_relevant_docs": all_docs,
        "eval_relevant_docs": eval_docs,
        "all_required_cells": all_cells,
        "eval_required_cells": eval_cells,
        "all_relevant_doc_rate": all_docs / all_total if all_total else None,
        "eval_relevant_doc_rate": eval_docs / eval_total if eval_total else None,
        "all_docs_by_table": all_by_table,
        "eval_docs_by_table": eval_by_table,
        "provenance_status": provenance_status,
        "curation_status": status,
        "remove_reason": "" if status == "keep" else status.replace("remove_", ""),
    }


def profile_existing_queries(
    datasets: list[DatasetRef],
    *,
    train_count: int,
    split_strategy: str,
    split_seed: int,
    min_docs: int,
    min_cells: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        try:
            loader = _loader(dataset)
            query_dict = loader.load_query_dict()
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "dataset_root": str(dataset.root),
                    "curation_status": "load_error",
                    "error": str(exc),
                }
            )
            continue
        evaluator = EvalDataExtraction(
            {
                "training_data_count": train_count,
                "training_data_split": split_strategy,
                "training_data_split_seed": split_seed,
            },
            data_loader=loader,
        )
        for query_id, query_info in query_dict.items():
            try:
                rows.append(
                    profile_query(
                        dataset=dataset,
                        loader=loader,
                        evaluator=evaluator,
                        query_id=str(query_id),
                        query_info=dict(query_info),
                        train_count=train_count,
                        split_strategy=split_strategy,
                        split_seed=split_seed,
                        min_docs=min_docs,
                        min_cells=min_cells,
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "dataset_root": str(dataset.root),
                        "query_id": str(query_id),
                        "curation_status": "profile_error",
                        "error": str(exc),
                    }
                )
    return rows


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    return value


def _decimal(value: Any) -> Decimal | None:
    if is_null(value):
        return None
    text = str(value).strip().replace(",", "")
    if text.startswith("$"):
        text = text[1:]
    if text.endswith("%"):
        text = text[:-1]
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _quote_literal(value: Any) -> str:
    if _decimal(value) is not None:
        return str(value).strip().replace(",", "")
    return "'" + str(value).replace("'", "''") + "'"


def _humanize_identifier(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ")).strip()


def _records_by_table(ref: DatasetRef) -> dict[str, pd.DataFrame]:
    manifest = _read_yaml(ref.manifest)
    gt_path = ref.root / manifest.get("paths", {}).get("ground_truth", "data/ground_truth.parquet")
    gt = pd.read_parquet(gt_path)
    if gt.empty:
        return {}
    gt = gt.copy()
    gt["value"] = gt["value"].map(_clean_scalar)
    index_cols = ["table_id", "doc_id", "record_id"]
    if "source_row_id" in gt.columns:
        index_cols.append("source_row_id")
    rows: list[dict[str, Any]] = []
    for keys, group in gt.groupby(index_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: dict[str, Any] = dict(zip(index_cols, keys))
        for _, item in group.iterrows():
            column = str(item.get("column_name") or "").strip()
            if column:
                record[column] = _clean_scalar(item.get("value"))
        rows.append(record)
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    result: dict[str, pd.DataFrame] = {}
    for table, table_df in frame.groupby("table_id", dropna=False):
        if is_null(table):
            continue
        result[str(table)] = table_df.reset_index(drop=True)
    return result


def _usable_columns(table_df: pd.DataFrame) -> list[str]:
    excluded = {"table_id", "doc_id", "record_id", "source_row_id", "row_id"}
    columns = []
    for column in table_df.columns:
        if column in excluded:
            continue
        non_null = table_df[column].map(lambda value: not is_null(value)).sum()
        if non_null >= 2:
            columns.append(str(column))
    return columns


def _choose_output_column(table_df: pd.DataFrame, predicate_column: str | None) -> str | None:
    candidates = _usable_columns(table_df)
    if predicate_column:
        non_pred = [column for column in candidates if column != predicate_column]
        if non_pred:
            candidates = non_pred
    if not candidates:
        return predicate_column

    preferred_name_tokens = (
        "name",
        "title",
        "team",
        "club",
        "school",
        "district",
        "county",
        "city",
        "state",
        "region",
        "course",
        "department",
        "instructor",
        "player",
        "wine",
        "company",
    )
    discouraged_name_tokens = (
        "id",
        "code",
        "status",
        "date",
        "year",
        "count",
        "num",
        "number",
        "percent",
        "ratio",
        "score",
        "avg",
        "latitude",
        "longitude",
        "lat",
        "long",
        "zip",
        "phone",
        "email",
    )

    def score(column: str) -> tuple[float, int, str]:
        series = table_df[column]
        non_null = series[series.map(lambda value: not is_null(value))]
        non_null_count = int(len(non_null))
        values = non_null.astype(str).str.strip()
        if values.empty:
            text_ratio = 0.0
            avg_len = 0.0
        else:
            text_ratio = float(values.map(lambda text: _decimal(text) is None).mean())
            avg_len = float(values.str.len().clip(upper=80).mean())
        name = column.lower()
        token_bonus = sum(1 for token in preferred_name_tokens if token in name)
        token_penalty = sum(1 for token in discouraged_name_tokens if token in name)
        uniqueness = float(values.nunique()) / max(non_null_count, 1)
        semantic_score = (
            token_bonus * 4.0
            + text_ratio * 3.0
            + min(avg_len / 20.0, 2.0)
            + min(uniqueness, 1.0)
            - token_penalty * 3.0
        )
        return (semantic_score, non_null_count, column)

    return max(candidates, key=score)


def _row_doc_count(table_df: pd.DataFrame, mask: pd.Series) -> int:
    if "doc_id" not in table_df.columns:
        return int(mask.sum())
    return int(table_df.loc[mask, "doc_id"].astype(str).nunique())


def _nearest_bucket(actual_coverage: float, buckets: tuple[float, ...]) -> float:
    if not buckets:
        return actual_coverage
    return min(buckets, key=lambda bucket: (abs(float(bucket) - actual_coverage), float(bucket)))


def _qualified_ref(table: str, column: str) -> str:
    return f"{table}.{column}"


def _is_join_candidate(candidate: dict[str, Any]) -> bool:
    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    if meta.get("is_join"):
        return True
    sql = str(candidate.get("sql") or "")
    return bool(re.search(r"\bJOIN\b", sql, flags=re.IGNORECASE))


def _candidate_predicate_key(candidate: dict[str, Any]) -> str:
    meta = candidate.get("meta") if isinstance(candidate.get("meta"), dict) else {}
    predicate = meta.get("predicate_column")
    if predicate:
        return str(predicate)
    tables = candidate.get("required_tables") or []
    return "ALL:" + "|".join(str(table) for table in tables)


def _candidate_payload(
    *,
    ref: DatasetRef,
    table: str,
    predicate_column: str | None,
    output_column: str,
    predicate_sql: str,
    predicate_text: str,
    bucket: float,
    dataset_total_docs: int,
    table_total_docs: int,
    result_docs: int,
    operator: str,
    value: Any,
) -> dict[str, Any]:
    select_col = _quote_ident(output_column)
    table_sql = _quote_ident(table)
    where = f" WHERE {predicate_sql}" if predicate_sql else ""
    sql = f"SELECT {select_col} FROM {table_sql}{where};"
    if predicate_text:
        question = f"List {_humanize_identifier(output_column)} from {table} where {predicate_text}."
    else:
        question = f"List {_humanize_identifier(output_column)} from all {table} records."
    required_columns = [f"{table}.{output_column}"]
    if predicate_column and predicate_column != output_column:
        required_columns.insert(0, f"{table}.{predicate_column}")
    return {
        "query_id": (
            f"Q_gen_{_slug(table)}_{_slug(predicate_column or 'all')}_"
            f"{_slug(operator)}_{int(round(bucket * 100)):03d}_{_slug(str(value)[:24])}"
        ),
        "dataset_id": ref.dataset_id,
        "question": question,
        "sql": sql,
        "required_tables": [table],
        "required_columns": required_columns,
        "output_columns": [f"{table}.{output_column}"],
        "meta": {
            "generated_by": "scripts/query_curation.py",
            "coverage_bucket": bucket,
            "target_doc_coverage": bucket,
            "actual_doc_coverage": result_docs / dataset_total_docs if dataset_total_docs else None,
            "actual_table_doc_coverage": result_docs / table_total_docs if table_total_docs else None,
            "result_docs": result_docs,
            "total_docs": dataset_total_docs,
            "table_total_docs": table_total_docs,
            "predicate_table": table if predicate_column else None,
            "predicate_column": _qualified_ref(table, predicate_column) if predicate_column else None,
            "operator": operator,
            "value": value,
            "is_join": False,
        },
    }


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return text[:48] or "value"


def _numeric_candidates(
    *,
    ref: DatasetRef,
    table: str,
    table_df: pd.DataFrame,
    column: str,
    buckets: tuple[float, ...],
    dataset_total_docs: int,
) -> list[dict[str, Any]]:
    values = table_df[column].map(_decimal)
    numeric_mask = values.map(lambda value: value is not None)
    if numeric_mask.sum() < 3 or numeric_mask.sum() / max(len(table_df), 1) < 0.6:
        return []
    numeric_values = sorted(set(value for value in values[numeric_mask] if value is not None))
    if len(numeric_values) < 3:
        return []
    if "doc_id" in table_df.columns:
        numeric_doc_frame = table_df.loc[numeric_mask, ["doc_id"]].copy()
        numeric_doc_frame["_numeric_value"] = values[numeric_mask]
        value_doc_sets = {
            value: set(group["doc_id"].astype(str).tolist())
            for value, group in numeric_doc_frame.groupby("_numeric_value", dropna=False)
        }
    else:
        numeric_doc_frame = pd.DataFrame({"_numeric_value": values[numeric_mask]})
        value_doc_sets = {
            value: set(str(index) for index in group.index.tolist())
            for value, group in numeric_doc_frame.groupby("_numeric_value", dropna=False)
        }
    table_total_docs = int(table_df["doc_id"].astype(str).nunique()) if "doc_id" in table_df.columns else len(table_df)
    output_column = _choose_output_column(table_df, column)
    if not output_column:
        return []
    candidates: list[dict[str, Any]] = []
    for bucket in buckets:
        target = max(1, round(dataset_total_docs * bucket))
        for op in (">=", "<="):
            best: tuple[int, Decimal, pd.Series] | None = None
            for threshold in numeric_values:
                mask = values.map(lambda value: value is not None and (value >= threshold if op == ">=" else value <= threshold))
                count = _row_doc_count(table_df, mask)
                if count <= 0:
                    continue
                distance = abs(count - target)
                if best is None or distance < best[0]:
                    best = (distance, threshold, mask)
            if best is None:
                continue
            _, threshold, mask = best
            result_docs = _row_doc_count(table_df, mask)
            predicate_sql = f"{_quote_ident(column)} {op} {_quote_literal(threshold)}"
            text_op = "is at least" if op == ">=" else "is at most"
            candidates.append(
                _candidate_payload(
                    ref=ref,
                    table=table,
                    predicate_column=column,
                    output_column=output_column,
                    predicate_sql=predicate_sql,
                    predicate_text=f"{_humanize_identifier(column)} {text_op} {threshold}",
                    bucket=bucket,
                    dataset_total_docs=dataset_total_docs,
                    table_total_docs=table_total_docs,
                    result_docs=result_docs,
                    operator=op,
                    value=str(threshold),
                )
            )
        if len(numeric_values) >= 4:
            start_fraction = max((1.0 - bucket) / 2.0, 0.0)
            end_fraction = min(1.0 - start_fraction, 1.0)
            lower_index = min(int(round(start_fraction * (len(numeric_values) - 1))), len(numeric_values) - 1)
            upper_index = min(int(round(end_fraction * (len(numeric_values) - 1))), len(numeric_values) - 1)
            lower = numeric_values[lower_index]
            upper = numeric_values[upper_index]
            if lower <= upper:
                mask = values.map(lambda value: value is not None and lower <= value <= upper)
                result_docs = _row_doc_count(table_df, mask)
                if result_docs > 0:
                    predicate_sql = f"{_quote_ident(column)} BETWEEN {_quote_literal(lower)} AND {_quote_literal(upper)}"
                    candidates.append(
                        _candidate_payload(
                            ref=ref,
                            table=table,
                            predicate_column=column,
                            output_column=output_column,
                            predicate_sql=predicate_sql,
                            predicate_text=f"{_humanize_identifier(column)} is between {lower} and {upper}",
                            bucket=bucket,
                            dataset_total_docs=dataset_total_docs,
                            table_total_docs=table_total_docs,
                            result_docs=result_docs,
                            operator="BETWEEN",
                            value=[str(lower), str(upper)],
                        )
                    )
        if len(numeric_values) >= 5:
            start_step = max(1, len(numeric_values) // 10)
            start_positions = sorted(
                {
                    0,
                    len(numeric_values) // 8,
                    len(numeric_values) // 4,
                    len(numeric_values) // 2,
                    (len(numeric_values) * 3) // 4,
                    (len(numeric_values) * 7) // 8,
                    len(numeric_values) - 2,
                    *range(0, len(numeric_values), start_step),
                }
            )
            window_options: list[tuple[int, Decimal, Decimal, int]] = []
            seen_windows: set[tuple[Decimal, Decimal]] = set()
            for start in start_positions:
                if start >= len(numeric_values):
                    continue
                docs: set[str] = set()
                for end in range(start, len(numeric_values)):
                    docs.update(value_doc_sets.get(numeric_values[end], set()))
                    count = len(docs)
                    if count <= 0:
                        continue
                    if count >= target or end == len(numeric_values) - 1:
                        lower = numeric_values[start]
                        upper = numeric_values[end]
                        key = (lower, upper)
                        if key not in seen_windows:
                            seen_windows.add(key)
                            window_options.append((abs(count - target), lower, upper, count))
                        break
            for _, lower, upper, result_docs in sorted(window_options, key=lambda item: (item[0], item[1], item[2]))[:3]:
                if lower > upper:
                    continue
                predicate_sql = f"{_quote_ident(column)} BETWEEN {_quote_literal(lower)} AND {_quote_literal(upper)}"
                candidates.append(
                    _candidate_payload(
                        ref=ref,
                        table=table,
                        predicate_column=column,
                        output_column=output_column,
                        predicate_sql=predicate_sql,
                        predicate_text=f"{_humanize_identifier(column)} is between {lower} and {upper}",
                        bucket=bucket,
                        dataset_total_docs=dataset_total_docs,
                        table_total_docs=table_total_docs,
                        result_docs=result_docs,
                        operator="BETWEEN",
                        value=[str(lower), str(upper)],
                    )
                )
    return candidates


def _nullness_candidates(
    *,
    ref: DatasetRef,
    table: str,
    table_df: pd.DataFrame,
    column: str,
    buckets: tuple[float, ...],
    dataset_total_docs: int,
) -> list[dict[str, Any]]:
    output_column = _choose_output_column(table_df, column)
    if not output_column:
        return []
    table_total_docs = int(table_df["doc_id"].astype(str).nunique()) if "doc_id" in table_df.columns else len(table_df)
    candidates: list[dict[str, Any]] = []
    for operator, expected_null, text in (
        ("IS NOT NULL", False, "is not null"),
        ("IS NULL", True, "is null"),
    ):
        mask = table_df[column].map(lambda value: is_null(value) if expected_null else not is_null(value))
        result_docs = _row_doc_count(table_df, mask)
        if result_docs <= 0:
            continue
        bucket = _nearest_bucket(result_docs / dataset_total_docs if dataset_total_docs else 0.0, buckets)
        candidates.append(
            _candidate_payload(
                ref=ref,
                table=table,
                predicate_column=column,
                output_column=output_column,
                predicate_sql=f"{_quote_ident(column)} {operator}",
                predicate_text=f"{_humanize_identifier(column)} {text}",
                bucket=bucket,
                dataset_total_docs=dataset_total_docs,
                table_total_docs=table_total_docs,
                result_docs=result_docs,
                operator=operator,
                value=text,
            )
        )
    return candidates


def _categorical_candidates(
    *,
    ref: DatasetRef,
    table: str,
    table_df: pd.DataFrame,
    column: str,
    buckets: tuple[float, ...],
    max_in_values: int,
    dataset_total_docs: int,
) -> list[dict[str, Any]]:
    non_null = table_df[table_df[column].map(lambda value: not is_null(value))]
    if non_null.empty:
        return []
    table_total_docs = int(table_df["doc_id"].astype(str).nunique()) if "doc_id" in table_df.columns else len(table_df)
    output_column = _choose_output_column(table_df, column)
    if not output_column:
        return []
    value_counts = (
        non_null.groupby(column)["doc_id"].nunique()
        if "doc_id" in non_null.columns
        else non_null.groupby(column).size()
    )
    value_counts = value_counts.sort_values(ascending=False)
    if value_counts.empty or len(value_counts) > 500:
        return []
    candidates: list[dict[str, Any]] = []
    if "doc_id" in non_null.columns:
        value_doc_sets = {
            value: set(group["doc_id"].astype(str).tolist())
            for value, group in non_null.groupby(column, dropna=False)
        }
        non_null_docs = set().union(*value_doc_sets.values()) if value_doc_sets else set()
    else:
        value_doc_sets = {
            value: set(str(index) for index in group.index.tolist())
            for value, group in non_null.groupby(column, dropna=False)
        }
        non_null_docs = set().union(*value_doc_sets.values()) if value_doc_sets else set()

    def selected_doc_count(values: list[Any], *, include: bool) -> int:
        selected_docs: set[str] = set()
        for value in values:
            selected_docs.update(value_doc_sets.get(value, set()))
        if include:
            return len(selected_docs)
        return len(non_null_docs - selected_docs)

    def prefix_best(
        values: list[Any],
        *,
        target: int,
        include: bool,
    ) -> tuple[int, list[Any], int] | None:
        chosen: list[Any] = []
        best: tuple[int, list[Any], int] | None = None
        for value in values[:max_in_values]:
            chosen.append(value)
            count = selected_doc_count(chosen, include=include)
            if count <= 0:
                continue
            distance = abs(count - target)
            if best is None or distance < best[0]:
                best = (distance, list(chosen), count)
        return best

    def greedy_best(
        values: list[Any],
        *,
        target: int,
        include: bool,
    ) -> tuple[int, list[Any], int] | None:
        chosen: list[Any] = []
        remaining = list(values[: max(max_in_values * 2, max_in_values)])
        best: tuple[int, list[Any], int] | None = None
        while remaining and len(chosen) < max_in_values:
            step_best: tuple[int, Any, int] | None = None
            for value in remaining:
                trial = chosen + [value]
                count = selected_doc_count(trial, include=include)
                if count <= 0:
                    continue
                distance = abs(count - target)
                if step_best is None or distance < step_best[0]:
                    step_best = (distance, value, count)
            if step_best is None:
                break
            distance, value, count = step_best
            chosen.append(value)
            remaining.remove(value)
            if best is None or distance < best[0]:
                best = (distance, list(chosen), count)
            if count == target:
                break
        return best

    def append_multi_value_candidate(
        *,
        bucket: float,
        values: list[Any],
        result_docs: int,
        include: bool,
    ) -> None:
        if len(values) <= 1 or result_docs <= 0:
            return
        literal_list = ", ".join(_quote_literal(value) for value in values)
        value_text = ", ".join(str(value) for value in values[:4])
        if len(values) > 4:
            value_text += f", and {len(values) - 4} more"
        operator = "IN" if include else "NOT IN"
        predicate_sql = (
            f"{_quote_ident(column)} IN ({literal_list})"
            if include
            else f"{_quote_ident(column)} NOT IN ({literal_list})"
        )
        predicate_text = (
            f"{_humanize_identifier(column)} is one of {value_text}"
            if include
            else f"{_humanize_identifier(column)} is not one of {value_text}"
        )
        candidates.append(
            _candidate_payload(
                ref=ref,
                table=table,
                predicate_column=column,
                output_column=output_column,
                predicate_sql=predicate_sql,
                predicate_text=predicate_text,
                bucket=bucket,
                dataset_total_docs=dataset_total_docs,
                table_total_docs=table_total_docs,
                result_docs=result_docs,
                operator=operator,
                value=[str(value) for value in values],
            )
        )

    for bucket in buckets:
        target = max(1, round(dataset_total_docs * bucket))
        best_value: tuple[int, Any, int] | None = None
        for value, count in value_counts.items():
            count = int(count)
            distance = abs(count - target)
            if best_value is None or distance < best_value[0]:
                best_value = (distance, value, count)
        if best_value is not None:
            _, value, result_docs = best_value
            predicate_sql = f"{_quote_ident(column)} = {_quote_literal(value)}"
            candidates.append(
                _candidate_payload(
                    ref=ref,
                    table=table,
                    predicate_column=column,
                    output_column=output_column,
                    predicate_sql=predicate_sql,
                    predicate_text=f"{_humanize_identifier(column)} equals {value}",
                    bucket=bucket,
                    dataset_total_docs=dataset_total_docs,
                    table_total_docs=table_total_docs,
                    result_docs=result_docs,
                    operator="=",
                    value=str(value),
                )
            )

        non_null_total = int(non_null["doc_id"].astype(str).nunique()) if "doc_id" in non_null.columns else int(len(non_null))
        best_not_value: tuple[int, Any, int] | None = None
        for value, value_count in value_counts.items():
            count = non_null_total - int(value_count)
            if count <= 0:
                continue
            distance = abs(count - target)
            if best_not_value is None or distance < best_not_value[0]:
                best_not_value = (distance, value, count)
        if best_not_value is not None:
            _, value, result_docs = best_not_value
            predicate_sql = f"{_quote_ident(column)} <> {_quote_literal(value)}"
            candidates.append(
                _candidate_payload(
                    ref=ref,
                    table=table,
                    predicate_column=column,
                    output_column=output_column,
                    predicate_sql=predicate_sql,
                    predicate_text=f"{_humanize_identifier(column)} does not equal {value}",
                    bucket=bucket,
                    dataset_total_docs=dataset_total_docs,
                    table_total_docs=table_total_docs,
                    result_docs=result_docs,
                    operator="!=",
                    value=str(value),
                )
            )

        chosen_values = []
        covered_docs: set[str] = set()
        best_in: tuple[int, list[Any], int] | None = None
        for value in value_counts.index.tolist()[:max_in_values]:
            chosen_values.append(value)
            mask = table_df[column].isin(chosen_values)
            if "doc_id" in table_df.columns:
                covered_docs = set(table_df.loc[mask, "doc_id"].astype(str).tolist())
                count = len(covered_docs)
            else:
                count = int(mask.sum())
            distance = abs(count - target)
            if best_in is None or distance < best_in[0]:
                best_in = (distance, list(chosen_values), count)
        if best_in is not None and len(best_in[1]) > 1:
            _, values, result_docs = best_in
            append_multi_value_candidate(bucket=bucket, values=values, result_docs=result_docs, include=True)

        best_not_in: tuple[int, list[Any], int] | None = None
        excluded_values = []
        for value in value_counts.index.tolist()[:max_in_values]:
            excluded_values.append(value)
            mask = ~table_df[column].isin(excluded_values) & table_df[column].map(lambda raw: not is_null(raw))
            if "doc_id" in table_df.columns:
                count = int(table_df.loc[mask, "doc_id"].astype(str).nunique())
            else:
                count = int(mask.sum())
            if count <= 0:
                continue
            distance = abs(count - target)
            if best_not_in is None or distance < best_not_in[0]:
                best_not_in = (distance, list(excluded_values), count)
        if best_not_in is not None and len(best_not_in[1]) > 1:
            _, values, result_docs = best_not_in
            append_multi_value_candidate(bucket=bucket, values=values, result_docs=result_docs, include=False)

        common_values = value_counts.index.tolist()
        rare_values = value_counts.sort_values(ascending=True).index.tolist()
        extra_sets: list[tuple[bool, tuple[int, list[Any], int] | None]] = [
            (True, prefix_best(rare_values, target=target, include=True)),
            (True, greedy_best(common_values, target=target, include=True)),
            (True, greedy_best(rare_values, target=target, include=True)),
            (False, prefix_best(rare_values, target=target, include=False)),
            (False, greedy_best(common_values, target=target, include=False)),
            (False, greedy_best(rare_values, target=target, include=False)),
        ]
        for include, best in extra_sets:
            if best is None:
                continue
            _, values, result_docs = best
            append_multi_value_candidate(bucket=bucket, values=values, result_docs=result_docs, include=include)
    return candidates


def _all_rows_candidate(
    ref: DatasetRef,
    table: str,
    table_df: pd.DataFrame,
    bucket: float,
    dataset_total_docs: int,
) -> dict[str, Any] | None:
    output_column = _choose_output_column(table_df, None)
    if not output_column:
        return None
    table_total_docs = int(table_df["doc_id"].astype(str).nunique()) if "doc_id" in table_df.columns else len(table_df)
    return _candidate_payload(
        ref=ref,
        table=table,
        predicate_column=None,
        output_column=output_column,
        predicate_sql="",
        predicate_text="",
        bucket=bucket,
        dataset_total_docs=dataset_total_docs,
        table_total_docs=table_total_docs,
        result_docs=table_total_docs,
        operator="ALL",
        value="all",
    )


def _normalize_join_value(value: Any) -> str | None:
    if is_null(value):
        return None
    decimal = _decimal(value)
    if decimal is not None:
        return str(decimal.normalize())
    text = re.sub(r"\s+", " ", str(value).strip()).strip(" \t\r\n\"'")
    text = re.sub(r"[\s.。:;；，,]+$", "", text).casefold()
    if not text or len(text) > 120:
        return None
    return text


def _column_value_profile(table_df: pd.DataFrame, column: str) -> set[str]:
    values: set[str] = set()
    for value in table_df[column].tolist():
        normalized = _normalize_join_value(value)
        if normalized:
            values.add(normalized)
    return values


def _column_name_similarity(left: str, right: str) -> float:
    left_norm = _slug(left)
    right_norm = _slug(right)
    if left_norm == right_norm:
        return 4.0
    left_tokens = {token for token in left_norm.split("_") if token}
    right_tokens = {token for token in right_norm.split("_") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    suffix_bonus = 0.0
    if left_norm.endswith(right_norm) or right_norm.endswith(left_norm):
        suffix_bonus = 1.0
    if left_norm.endswith("_id") and right_norm.endswith("_id"):
        suffix_bonus = max(suffix_bonus, 1.5)
    if ("name" in left_tokens and "name" in right_tokens) or ("id" in left_tokens and "id" in right_tokens):
        suffix_bonus = max(suffix_bonus, 1.0)
    return overlap + suffix_bonus


def _name_tokens(value: str) -> set[str]:
    return {token for token in _slug(value).split("_") if token}


def _base_name_tokens(value: str) -> set[str]:
    generic = {
        "id",
        "name",
        "title",
        "code",
        "number",
        "num",
        "date",
        "time",
    }
    return {token for token in _name_tokens(value) if token not in generic}


def _column_join_name_compatible(left_table: str, left_column: str, right_table: str, right_column: str) -> bool:
    left_norm = _slug(left_column)
    right_norm = _slug(right_column)
    if left_norm == right_norm:
        return True

    left_base = _base_name_tokens(left_column)
    right_base = _base_name_tokens(right_column)
    if left_base and right_base and left_base == right_base:
        return True
    if left_base and right_base and (left_base <= right_base or right_base <= left_base):
        return True

    left_tokens = _name_tokens(left_column)
    right_tokens = _name_tokens(right_column)
    left_table_tokens = _base_name_tokens(left_table)
    right_table_tokens = _base_name_tokens(right_table)
    if left_tokens <= {"name", "title"} and left_table_tokens & right_tokens:
        return True
    if right_tokens <= {"name", "title"} and right_table_tokens & left_tokens:
        return True
    if {"name", "title"} & left_tokens and {"name", "title"} & right_tokens:
        if left_table_tokens & right_tokens or right_table_tokens & left_tokens:
            return True

    left_is_id = left_norm.endswith("_id") or left_norm == "id"
    right_is_id = right_norm.endswith("_id") or right_norm == "id"
    if left_is_id or right_is_id:
        return False

    meaningful = {
        "airport",
        "appellation",
        "apt",
        "building",
        "country",
        "course",
        "customer",
        "department",
        "dept",
        "event",
        "gas",
        "guest",
        "instructor",
        "major",
        "member",
        "player",
        "product",
        "school",
        "station",
        "team",
        "wine",
    }
    return bool((left_base & right_base) & meaningful)


def infer_join_pairs(
    tables: dict[str, pd.DataFrame],
    *,
    max_pairs: int = 32,
) -> list[JoinPair]:
    column_profiles: list[tuple[str, str, set[str]]] = []
    for table, table_df in tables.items():
        if table_df.empty:
            continue
        for column in _usable_columns(table_df):
            values = _column_value_profile(table_df, column)
            if 2 <= len(values) <= 5000:
                column_profiles.append((table, column, values))

    pairs: list[JoinPair] = []
    seen: set[tuple[str, str, str, str]] = set()
    for idx, (left_table, left_column, left_values) in enumerate(column_profiles):
        for right_table, right_column, right_values in column_profiles[idx + 1 :]:
            if left_table == right_table:
                continue
            shared = left_values & right_values
            if not shared:
                continue
            shared_count = len(shared)
            min_ratio = shared_count / max(min(len(left_values), len(right_values)), 1)
            max_ratio = shared_count / max(max(len(left_values), len(right_values)), 1)
            if not _column_join_name_compatible(left_table, left_column, right_table, right_column):
                continue
            name_score = _column_name_similarity(left_column, right_column)
            if name_score < 1.0 and shared_count < 5 and min_ratio < 0.12:
                continue
            score = name_score * 10.0 + shared_count + min_ratio * 20.0 + max_ratio * 5.0
            key = tuple(sorted([(left_table, left_column), (right_table, right_column)]))
            flat_key = (key[0][0], key[0][1], key[1][0], key[1][1])
            if flat_key in seen:
                continue
            seen.add(flat_key)
            pairs.append(
                JoinPair(
                    left_table=left_table,
                    left_column=left_column,
                    right_table=right_table,
                    right_column=right_column,
                    shared_values=shared_count,
                    left_values=len(left_values),
                    right_values=len(right_values),
                    score=score,
                )
            )
    pairs.sort(key=lambda pair: (-pair.score, pair.left_table, pair.left_column, pair.right_table, pair.right_column))
    return pairs[:max_pairs]


def _join_working_frame(
    tables: dict[str, pd.DataFrame],
    pair: JoinPair,
    *,
    preserved_table: str | None = None,
) -> pd.DataFrame | None:
    left = tables.get(pair.left_table)
    right = tables.get(pair.right_table)
    if left is None or right is None or left.empty or right.empty:
        return None
    if pair.left_column not in left.columns or pair.right_column not in right.columns:
        return None

    left_keys = left[pair.left_column].map(_normalize_join_value)
    right_keys = right[pair.right_column].map(_normalize_join_value)
    left_counts = left_keys[left_keys.map(lambda value: value is not None)].value_counts()
    right_counts = right_keys[right_keys.map(lambda value: value is not None)].value_counts()
    if preserved_table == pair.left_table:
        estimated_rows = sum(max(1, int(right_counts.get(value, 0))) if value is not None else 1 for value in left_keys)
        join_how = "left"
    elif preserved_table == pair.right_table:
        estimated_rows = sum(max(1, int(left_counts.get(value, 0))) if value is not None else 1 for value in right_keys)
        join_how = "right"
    else:
        estimated_rows = sum(
            int(left_counts.get(value, 0)) * int(right_counts.get(value, 0))
            for value in set(left_counts.index) & set(right_counts.index)
        )
        join_how = "inner"
    if estimated_rows <= 0 or estimated_rows > 50000:
        return None

    left_cols = ["doc_id", pair.left_column, *_usable_columns(left)]
    right_cols = ["doc_id", pair.right_column, *_usable_columns(right)]
    left_cols = list(dict.fromkeys(column for column in left_cols if column in left.columns))
    right_cols = list(dict.fromkeys(column for column in right_cols if column in right.columns))
    left_renamed = left[left_cols].rename(
        columns={column: _qualified_ref(pair.left_table, column) for column in left_cols}
    )
    right_renamed = right[right_cols].rename(
        columns={column: _qualified_ref(pair.right_table, column) for column in right_cols}
    )
    merged = left_renamed.merge(
        right_renamed,
        left_on=_qualified_ref(pair.left_table, pair.left_column),
        right_on=_qualified_ref(pair.right_table, pair.right_column),
        how=join_how,
    )
    if merged.empty:
        return None
    return merged


def _join_doc_count(merged: pd.DataFrame, mask: pd.Series, left_table: str, right_table: str) -> int:
    doc_ids: set[str] = set()
    for table in (left_table, right_table):
        column = _qualified_ref(table, "doc_id")
        if column in merged.columns:
            doc_ids.update(
                str(value)
                for value in merged.loc[mask, column].tolist()
                if not is_null(value)
            )
    return len(doc_ids)


def _join_candidate_payload(
    *,
    ref: DatasetRef,
    pair: JoinPair,
    output_table: str,
    output_column: str,
    predicate_table: str | None,
    predicate_column: str | None,
    predicate_sql: str,
    predicate_text: str,
    bucket: float,
    dataset_total_docs: int,
    result_docs: int,
    operator: str,
    value: Any,
    join_type: str = "INNER",
) -> dict[str, Any]:
    select_col = f"{_quote_ident(output_table)}.{_quote_ident(output_column)}"
    left_ref = f"{_quote_ident(pair.left_table)}.{_quote_ident(pair.left_column)}"
    right_ref = f"{_quote_ident(pair.right_table)}.{_quote_ident(pair.right_column)}"
    if join_type == "LEFT" and output_table == pair.right_table:
        join_sql = (
            f"FROM {_quote_ident(pair.right_table)} "
            f"LEFT JOIN {_quote_ident(pair.left_table)} ON {right_ref} = {left_ref}"
        )
        other_table = pair.left_table
    elif join_type == "LEFT":
        join_sql = (
            f"FROM {_quote_ident(pair.left_table)} "
            f"LEFT JOIN {_quote_ident(pair.right_table)} ON {left_ref} = {right_ref}"
        )
        other_table = pair.right_table
    else:
        join_sql = (
            f"FROM {_quote_ident(pair.left_table)} "
            f"JOIN {_quote_ident(pair.right_table)} ON {left_ref} = {right_ref}"
        )
        other_table = pair.right_table if output_table == pair.left_table else pair.left_table
    where = f" WHERE {predicate_sql}" if predicate_sql else ""
    sql = f"SELECT {select_col} {join_sql}{where};"
    if predicate_text and predicate_table:
        link_text = "optionally linked to" if join_type == "LEFT" else "linked to"
        question = (
            f"List {_humanize_identifier(output_column)} from {output_table} records {link_text} "
            f"{other_table} where {predicate_table} {_humanize_identifier(predicate_column or '')} {predicate_text}."
        )
    else:
        link_text = "optionally linked" if join_type == "LEFT" else "matching"
        question = (
            f"List {_humanize_identifier(output_column)} from {output_table} records that have {link_text} "
            f"{other_table} records on {_humanize_identifier(pair.left_column)}."
        )

    required_columns = [
        _qualified_ref(pair.left_table, pair.left_column),
        _qualified_ref(pair.right_table, pair.right_column),
        _qualified_ref(output_table, output_column),
    ]
    if predicate_table and predicate_column:
        required_columns.append(_qualified_ref(predicate_table, predicate_column))
    required_columns = list(dict.fromkeys(required_columns))
    return {
        "query_id": (
            f"Q_{'left_join' if join_type == 'LEFT' else 'join'}_"
            f"{_slug(pair.left_table)}_{_slug(pair.right_table)}_"
            f"{_slug(output_table)}_"
            f"{_slug(predicate_table or 'all')}_{_slug(predicate_column or 'all')}_"
            f"{int(round(bucket * 100)):03d}_{_slug(str(value)[:24])}"
        ),
        "dataset_id": ref.dataset_id,
        "question": question,
        "sql": sql,
        "required_tables": [pair.left_table, pair.right_table],
        "required_columns": required_columns,
        "output_columns": [_qualified_ref(output_table, output_column)],
        "meta": {
            "generated_by": "scripts/query_curation.py",
            "coverage_bucket": bucket,
            "target_doc_coverage": bucket,
            "actual_doc_coverage": result_docs / dataset_total_docs if dataset_total_docs else None,
            "result_docs": result_docs,
            "total_docs": dataset_total_docs,
            "predicate_table": predicate_table,
            "predicate_column": _qualified_ref(predicate_table, predicate_column) if predicate_table and predicate_column else None,
            "operator": operator,
            "value": value,
            "is_join": True,
            "join_type": join_type,
            "join_tables": [pair.left_table, pair.right_table],
            "join_columns": [
                _qualified_ref(pair.left_table, pair.left_column),
                _qualified_ref(pair.right_table, pair.right_column),
            ],
            "join_shared_values": pair.shared_values,
            "join_score": pair.score,
        },
    }


def _predicate_columns_for_join(
    tables: dict[str, pd.DataFrame],
    table: str,
    excluded: set[str],
    *,
    max_columns: int,
) -> list[str]:
    table_df = tables.get(table)
    if table_df is None:
        return []
    columns = [column for column in _usable_columns(table_df) if column not in excluded]

    def score(column: str) -> tuple[float, str]:
        non_null = table_df[column][table_df[column].map(lambda value: not is_null(value))]
        distinct = int(non_null.astype(str).str.strip().nunique()) if not non_null.empty else 0
        numeric_ratio = float(non_null.map(lambda value: _decimal(value) is not None).mean()) if not non_null.empty else 0.0
        name = column.lower()
        semantic_bonus = sum(
            1.0
            for token in (
                "name",
                "type",
                "status",
                "city",
                "state",
                "country",
                "department",
                "major",
                "event",
                "team",
                "score",
                "amount",
                "date",
                "year",
            )
            if token in name
        )
        distinct_score = min(distinct / 20.0, 2.0)
        return (semantic_bonus + distinct_score + numeric_ratio, column)

    columns.sort(key=score, reverse=True)
    return columns[:max_columns]


def _join_predicate_candidates(
    *,
    ref: DatasetRef,
    pair: JoinPair,
    tables: dict[str, pd.DataFrame],
    buckets: tuple[float, ...],
    max_in_values: int,
    dataset_total_docs: int,
    max_predicate_columns: int,
) -> list[dict[str, Any]]:
    merged = _join_working_frame(tables, pair)
    if merged is None or merged.empty:
        return []

    candidates: list[dict[str, Any]] = []
    table_pairs = [
        (pair.left_table, pair.left_column, pair.right_table),
        (pair.right_table, pair.right_column, pair.left_table),
    ]
    for output_table, join_column, predicate_table in table_pairs:
        output_df = tables[output_table]
        output_column = _choose_output_column(output_df, join_column)
        if not output_column:
            continue
        output_ref = _qualified_ref(output_table, output_column)
        if output_ref not in merged.columns:
            continue
        all_mask = pd.Series([True] * len(merged), index=merged.index)
        result_docs = _join_doc_count(merged, all_mask, pair.left_table, pair.right_table)
        if result_docs > 0:
            bucket = _nearest_bucket(result_docs / dataset_total_docs if dataset_total_docs else 0.0, buckets)
            candidates.append(
                _join_candidate_payload(
                    ref=ref,
                    pair=pair,
                    output_table=output_table,
                    output_column=output_column,
                    predicate_table=None,
                    predicate_column=None,
                    predicate_sql="",
                    predicate_text="",
                    bucket=bucket,
                    dataset_total_docs=dataset_total_docs,
                    result_docs=result_docs,
                    operator="JOIN_ALL",
                    value="all",
                )
            )

        excluded = {join_column}
        if predicate_table == pair.left_table:
            excluded.add(pair.left_column)
        if predicate_table == pair.right_table:
            excluded.add(pair.right_column)
        predicate_columns = _predicate_columns_for_join(
            tables,
            predicate_table,
            excluded,
            max_columns=max_predicate_columns,
        )
        for predicate_column in predicate_columns:
            predicate_ref = _qualified_ref(predicate_table, predicate_column)
            if predicate_ref not in merged.columns:
                continue
            values = merged[predicate_ref]
            decimals = values.map(_decimal)
            numeric_mask = decimals.map(lambda value: value is not None)
            if numeric_mask.sum() >= 3 and numeric_mask.sum() / max(len(values), 1) >= 0.6:
                numeric_values = sorted(set(value for value in decimals[numeric_mask] if value is not None))
                for bucket in buckets:
                    target = max(1, round(dataset_total_docs * bucket))
                    for op in (">=", "<="):
                        best: tuple[int, Decimal, int] | None = None
                        for threshold in numeric_values:
                            mask = decimals.map(
                                lambda value: value is not None and (value >= threshold if op == ">=" else value <= threshold)
                            )
                            count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                            if count <= 0:
                                continue
                            distance = abs(count - target)
                            if best is None or distance < best[0]:
                                best = (distance, threshold, count)
                        if best is None:
                            continue
                        _, threshold, result_docs = best
                        predicate_sql = f"{_quote_ident(predicate_table)}.{_quote_ident(predicate_column)} {op} {_quote_literal(threshold)}"
                        text_op = "is at least" if op == ">=" else "is at most"
                        candidates.append(
                            _join_candidate_payload(
                                ref=ref,
                                pair=pair,
                                output_table=output_table,
                                output_column=output_column,
                                predicate_table=predicate_table,
                                predicate_column=predicate_column,
                                predicate_sql=predicate_sql,
                                predicate_text=f"{text_op} {threshold}",
                                bucket=bucket,
                                dataset_total_docs=dataset_total_docs,
                                result_docs=result_docs,
                                operator=op,
                                value=str(threshold),
                            )
                        )
                continue

            non_null = merged[merged[predicate_ref].map(lambda value: not is_null(value))]
            if non_null.empty:
                continue
            value_counts = non_null.groupby(predicate_ref).size().sort_values(ascending=False)
            if value_counts.empty or len(value_counts) > 500:
                continue
            for bucket in buckets:
                target = max(1, round(dataset_total_docs * bucket))
                best_value: tuple[int, Any, int] | None = None
                for value in value_counts.index.tolist():
                    mask = merged[predicate_ref] == value
                    count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                    if count <= 0:
                        continue
                    distance = abs(count - target)
                    if best_value is None or distance < best_value[0]:
                        best_value = (distance, value, count)
                if best_value is not None:
                    _, value, result_docs = best_value
                    predicate_sql = f"{_quote_ident(predicate_table)}.{_quote_ident(predicate_column)} = {_quote_literal(value)}"
                    candidates.append(
                        _join_candidate_payload(
                            ref=ref,
                            pair=pair,
                            output_table=output_table,
                            output_column=output_column,
                            predicate_table=predicate_table,
                            predicate_column=predicate_column,
                            predicate_sql=predicate_sql,
                            predicate_text=f"equals {value}",
                            bucket=bucket,
                            dataset_total_docs=dataset_total_docs,
                            result_docs=result_docs,
                            operator="=",
                            value=str(value),
                        )
                    )

                chosen_values = []
                best_in: tuple[int, list[Any], int] | None = None
                for value in value_counts.index.tolist()[:max_in_values]:
                    chosen_values.append(value)
                    mask = merged[predicate_ref].isin(chosen_values)
                    count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                    if count <= 0:
                        continue
                    distance = abs(count - target)
                    if best_in is None or distance < best_in[0]:
                        best_in = (distance, list(chosen_values), count)
                if best_in is not None and len(best_in[1]) > 1:
                    _, values_in, result_docs = best_in
                    literal_list = ", ".join(_quote_literal(value) for value in values_in)
                    value_text = ", ".join(str(value) for value in values_in[:4])
                    if len(values_in) > 4:
                        value_text += f", and {len(values_in) - 4} more"
                    predicate_sql = f"{_quote_ident(predicate_table)}.{_quote_ident(predicate_column)} IN ({literal_list})"
                    candidates.append(
                        _join_candidate_payload(
                            ref=ref,
                            pair=pair,
                            output_table=output_table,
                            output_column=output_column,
                            predicate_table=predicate_table,
                            predicate_column=predicate_column,
                            predicate_sql=predicate_sql,
                            predicate_text=f"is one of {value_text}",
                            bucket=bucket,
                            dataset_total_docs=dataset_total_docs,
                            result_docs=result_docs,
                            operator="IN",
                            value=[str(value) for value in values_in],
                        )
                    )

                excluded_values = []
                best_not_in: tuple[int, list[Any], int] | None = None
                for value in value_counts.index.tolist()[:max_in_values]:
                    excluded_values.append(value)
                    mask = (
                        ~merged[predicate_ref].isin(excluded_values)
                        & merged[predicate_ref].map(lambda raw: not is_null(raw))
                    )
                    count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                    if count <= 0:
                        continue
                    distance = abs(count - target)
                    if best_not_in is None or distance < best_not_in[0]:
                        best_not_in = (distance, list(excluded_values), count)
                if best_not_in is not None and len(best_not_in[1]) > 1:
                    _, values_not_in, result_docs = best_not_in
                    literal_list = ", ".join(_quote_literal(value) for value in values_not_in)
                    value_text = ", ".join(str(value) for value in values_not_in[:4])
                    if len(values_not_in) > 4:
                        value_text += f", and {len(values_not_in) - 4} more"
                    predicate_sql = f"{_quote_ident(predicate_table)}.{_quote_ident(predicate_column)} NOT IN ({literal_list})"
                    candidates.append(
                        _join_candidate_payload(
                            ref=ref,
                            pair=pair,
                            output_table=output_table,
                            output_column=output_column,
                            predicate_table=predicate_table,
                            predicate_column=predicate_column,
                            predicate_sql=predicate_sql,
                            predicate_text=f"is not one of {value_text}",
                            bucket=bucket,
                            dataset_total_docs=dataset_total_docs,
                            result_docs=result_docs,
                            operator="NOT IN",
                            value=[str(value) for value in values_not_in],
                        )
                    )
    return candidates


def _left_join_predicate_candidates(
    *,
    ref: DatasetRef,
    pair: JoinPair,
    tables: dict[str, pd.DataFrame],
    buckets: tuple[float, ...],
    max_in_values: int,
    dataset_total_docs: int,
    max_predicate_columns: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    table_pairs = [
        (pair.left_table, pair.left_column),
        (pair.right_table, pair.right_column),
    ]
    for output_table, join_column in table_pairs:
        merged = _join_working_frame(tables, pair, preserved_table=output_table)
        if merged is None or merged.empty:
            continue
        output_df = tables[output_table]
        output_column = _choose_output_column(output_df, join_column)
        if not output_column:
            continue
        output_ref = _qualified_ref(output_table, output_column)
        if output_ref not in merged.columns:
            continue

        all_mask = pd.Series([True] * len(merged), index=merged.index)
        result_docs = _join_doc_count(merged, all_mask, pair.left_table, pair.right_table)
        if result_docs > 0:
            bucket = _nearest_bucket(result_docs / dataset_total_docs if dataset_total_docs else 0.0, buckets)
            candidates.append(
                _join_candidate_payload(
                    ref=ref,
                    pair=pair,
                    output_table=output_table,
                    output_column=output_column,
                    predicate_table=None,
                    predicate_column=None,
                    predicate_sql="",
                    predicate_text="",
                    bucket=bucket,
                    dataset_total_docs=dataset_total_docs,
                    result_docs=result_docs,
                    operator="LEFT_JOIN_ALL",
                    value="all",
                    join_type="LEFT",
                )
            )

        predicate_columns = _predicate_columns_for_join(
            tables,
            output_table,
            {join_column},
            max_columns=max_predicate_columns,
        )
        for predicate_column in predicate_columns:
            predicate_ref = _qualified_ref(output_table, predicate_column)
            if predicate_ref not in merged.columns:
                continue

            null_mask = merged[predicate_ref].map(lambda value: not is_null(value))
            null_docs = _join_doc_count(merged, null_mask, pair.left_table, pair.right_table)
            if null_docs > 0:
                bucket = _nearest_bucket(null_docs / dataset_total_docs if dataset_total_docs else 0.0, buckets)
                candidates.append(
                    _join_candidate_payload(
                        ref=ref,
                        pair=pair,
                        output_table=output_table,
                        output_column=output_column,
                        predicate_table=output_table,
                        predicate_column=predicate_column,
                        predicate_sql=f"{_quote_ident(output_table)}.{_quote_ident(predicate_column)} IS NOT NULL",
                        predicate_text="is not null",
                        bucket=bucket,
                        dataset_total_docs=dataset_total_docs,
                        result_docs=null_docs,
                        operator="IS NOT NULL",
                        value="is not null",
                        join_type="LEFT",
                    )
                )

            # LEFT JOIN candidates primarily fill high-coverage join gaps that
            # inner joins cannot reach. Nullness predicates provide cheap
            # attribute diversity without repeatedly scanning large join frames.
            continue

            values = merged[predicate_ref]
            decimals = values.map(_decimal)
            numeric_mask = decimals.map(lambda value: value is not None)
            if numeric_mask.sum() >= 3 and numeric_mask.sum() / max(len(values), 1) >= 0.6:
                numeric_values = sorted(set(value for value in decimals[numeric_mask] if value is not None))
                for bucket in buckets:
                    target = max(1, round(dataset_total_docs * bucket))
                    for op in (">=", "<="):
                        best: tuple[int, Decimal, int] | None = None
                        for threshold in numeric_values:
                            mask = decimals.map(
                                lambda value: value is not None and (value >= threshold if op == ">=" else value <= threshold)
                            )
                            count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                            if count <= 0:
                                continue
                            distance = abs(count - target)
                            if best is None or distance < best[0]:
                                best = (distance, threshold, count)
                        if best is None:
                            continue
                        _, threshold, result_docs = best
                        text_op = "is at least" if op == ">=" else "is at most"
                        candidates.append(
                            _join_candidate_payload(
                                ref=ref,
                                pair=pair,
                                output_table=output_table,
                                output_column=output_column,
                                predicate_table=output_table,
                                predicate_column=predicate_column,
                                predicate_sql=f"{_quote_ident(output_table)}.{_quote_ident(predicate_column)} {op} {_quote_literal(threshold)}",
                                predicate_text=f"{text_op} {threshold}",
                                bucket=bucket,
                                dataset_total_docs=dataset_total_docs,
                                result_docs=result_docs,
                                operator=op,
                                value=str(threshold),
                                join_type="LEFT",
                            )
                        )
                continue

            non_null = merged[merged[predicate_ref].map(lambda value: not is_null(value))]
            if non_null.empty:
                continue
            value_counts = non_null.groupby(predicate_ref).size().sort_values(ascending=False)
            if value_counts.empty or len(value_counts) > 500:
                continue
            for bucket in buckets:
                target = max(1, round(dataset_total_docs * bucket))
                best_value: tuple[int, Any, int] | None = None
                for value in value_counts.index.tolist():
                    mask = merged[predicate_ref] == value
                    count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                    if count <= 0:
                        continue
                    distance = abs(count - target)
                    if best_value is None or distance < best_value[0]:
                        best_value = (distance, value, count)
                if best_value is not None:
                    _, value, result_docs = best_value
                    candidates.append(
                        _join_candidate_payload(
                            ref=ref,
                            pair=pair,
                            output_table=output_table,
                            output_column=output_column,
                            predicate_table=output_table,
                            predicate_column=predicate_column,
                            predicate_sql=f"{_quote_ident(output_table)}.{_quote_ident(predicate_column)} = {_quote_literal(value)}",
                            predicate_text=f"equals {value}",
                            bucket=bucket,
                            dataset_total_docs=dataset_total_docs,
                            result_docs=result_docs,
                            operator="=",
                            value=str(value),
                            join_type="LEFT",
                        )
                    )

                chosen_values = []
                best_in: tuple[int, list[Any], int] | None = None
                for value in value_counts.index.tolist()[:max_in_values]:
                    chosen_values.append(value)
                    mask = merged[predicate_ref].isin(chosen_values)
                    count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                    if count <= 0:
                        continue
                    distance = abs(count - target)
                    if best_in is None or distance < best_in[0]:
                        best_in = (distance, list(chosen_values), count)
                if best_in is not None and len(best_in[1]) > 1:
                    _, values_in, result_docs = best_in
                    literal_list = ", ".join(_quote_literal(value) for value in values_in)
                    value_text = ", ".join(str(value) for value in values_in[:4])
                    if len(values_in) > 4:
                        value_text += f", and {len(values_in) - 4} more"
                    candidates.append(
                        _join_candidate_payload(
                            ref=ref,
                            pair=pair,
                            output_table=output_table,
                            output_column=output_column,
                            predicate_table=output_table,
                            predicate_column=predicate_column,
                            predicate_sql=f"{_quote_ident(output_table)}.{_quote_ident(predicate_column)} IN ({literal_list})",
                            predicate_text=f"is one of {value_text}",
                            bucket=bucket,
                            dataset_total_docs=dataset_total_docs,
                            result_docs=result_docs,
                            operator="IN",
                            value=[str(value) for value in values_in],
                            join_type="LEFT",
                        )
                    )

                excluded_values = []
                best_not_in: tuple[int, list[Any], int] | None = None
                for value in value_counts.index.tolist()[:max_in_values]:
                    excluded_values.append(value)
                    mask = (
                        ~merged[predicate_ref].isin(excluded_values)
                        & merged[predicate_ref].map(lambda raw: not is_null(raw))
                    )
                    count = _join_doc_count(merged, mask, pair.left_table, pair.right_table)
                    if count <= 0:
                        continue
                    distance = abs(count - target)
                    if best_not_in is None or distance < best_not_in[0]:
                        best_not_in = (distance, list(excluded_values), count)
                if best_not_in is not None and len(best_not_in[1]) > 1:
                    _, values_not_in, result_docs = best_not_in
                    literal_list = ", ".join(_quote_literal(value) for value in values_not_in)
                    value_text = ", ".join(str(value) for value in values_not_in[:4])
                    if len(values_not_in) > 4:
                        value_text += f", and {len(values_not_in) - 4} more"
                    candidates.append(
                        _join_candidate_payload(
                            ref=ref,
                            pair=pair,
                            output_table=output_table,
                            output_column=output_column,
                            predicate_table=output_table,
                            predicate_column=predicate_column,
                            predicate_sql=f"{_quote_ident(output_table)}.{_quote_ident(predicate_column)} NOT IN ({literal_list})",
                            predicate_text=f"is not one of {value_text}",
                            bucket=bucket,
                            dataset_total_docs=dataset_total_docs,
                            result_docs=result_docs,
                            operator="NOT IN",
                            value=[str(value) for value in values_not_in],
                            join_type="LEFT",
                        )
                    )
    return candidates


def _select_diverse_candidates(
    candidates: list[dict[str, Any]],
    *,
    per_bucket: int,
    min_join_per_bucket: int,
) -> list[dict[str, Any]]:
    def base_key(item: dict[str, Any]) -> tuple[float, int, str]:
        return (
            float(item.get("selection_score", 999.0)),
            -int(item.get("profile", {}).get("all_required_cells", 0) or 0),
            str(item.get("sql")),
        )

    eligible = sorted(
        [
            item
            for item in candidates
            if item.get("profile", {}).get("curation_status") == "keep"
            and item.get("profile", {}).get("provenance_status") != "fallback_all_required_rows"
        ],
        key=base_key,
    )
    selected: list[dict[str, Any]] = []
    seen_sql: set[str] = set()
    used_predicates: set[str] = set()
    used_table_sets: set[tuple[str, ...]] = set()

    def add(item: dict[str, Any]) -> bool:
        sql = str(item.get("sql") or "")
        if not sql or sql in seen_sql or len(selected) >= per_bucket:
            return False
        selected.append(item)
        seen_sql.add(sql)
        used_predicates.add(_candidate_predicate_key(item))
        used_table_sets.add(tuple(sorted(str(table) for table in item.get("required_tables") or [])))
        return True

    join_candidates = [item for item in eligible if _is_join_candidate(item)]
    while sum(1 for item in selected if _is_join_candidate(item)) < min_join_per_bucket:
        remaining = [item for item in join_candidates if str(item.get("sql") or "") not in seen_sql]
        if not remaining:
            break
        remaining.sort(
            key=lambda item: (
                _candidate_predicate_key(item) in used_predicates,
                tuple(sorted(str(table) for table in item.get("required_tables") or [])) in used_table_sets,
                *base_key(item),
            )
        )
        add(remaining[0])

    while len(selected) < per_bucket:
        remaining = [item for item in eligible if str(item.get("sql") or "") not in seen_sql]
        if not remaining:
            break
        remaining.sort(
            key=lambda item: (
                _candidate_predicate_key(item) in used_predicates,
                tuple(sorted(str(table) for table in item.get("required_tables") or [])) in used_table_sets,
                not _is_join_candidate(item),
                *base_key(item),
            )
        )
        if not add(remaining[0]):
            break
    return selected


def _prefilter_raw_candidates(
    raw: list[dict[str, Any]],
    *,
    buckets: tuple[float, ...],
    max_per_bucket: int,
) -> list[dict[str, Any]]:
    if max_per_bucket <= 0:
        return raw
    selected: list[dict[str, Any]] = []
    for bucket in buckets:
        bucket_candidates = [
            item
            for item in raw
            if abs(float(item.get("meta", {}).get("coverage_bucket", -1)) - bucket) < 1e-12
        ]
        if len(bucket_candidates) <= max_per_bucket:
            selected.extend(bucket_candidates)
            continue

        def score(item: dict[str, Any]) -> tuple[float, bool, str]:
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            actual = meta.get("actual_doc_coverage")
            target = meta.get("target_doc_coverage")
            try:
                distance = abs(float(actual) - float(target))
            except (TypeError, ValueError):
                distance = 999.0
            return (distance, not _is_join_candidate(item), str(item.get("sql")))

        ordered = sorted(bucket_candidates, key=score)
        used_predicates: set[str] = set()
        used_sql: set[str] = set()
        bucket_selected: list[dict[str, Any]] = []
        for item in ordered:
            predicate = _candidate_predicate_key(item)
            sql = str(item.get("sql") or "")
            if predicate in used_predicates or sql in used_sql:
                continue
            bucket_selected.append(item)
            used_predicates.add(predicate)
            used_sql.add(sql)
            if len(bucket_selected) >= max_per_bucket:
                break
        if len(bucket_selected) < max_per_bucket:
            for item in ordered:
                sql = str(item.get("sql") or "")
                if sql in used_sql:
                    continue
                bucket_selected.append(item)
                used_sql.add(sql)
                if len(bucket_selected) >= max_per_bucket:
                    break
        selected.extend(bucket_selected)
    return selected


def generate_candidates(
    datasets: list[DatasetRef],
    *,
    buckets: tuple[float, ...],
    per_bucket: int,
    min_join_per_bucket: int,
    max_in_values: int,
    include_joins: bool,
    join_only: bool,
    left_joins_only: bool,
    max_join_pairs: int,
    max_join_predicate_columns: int,
    max_raw_candidates_per_bucket: int,
    min_docs: int,
    min_cells: int,
    train_count: int,
    split_strategy: str,
    split_seed: int,
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for ref in datasets:
        print(f"[generate] {ref.dataset_id}", flush=True)
        try:
            tables = _records_by_table(ref)
            loader = _loader(ref)
        except Exception as exc:
            generated.append({"dataset_id": ref.dataset_id, "dataset_root": str(ref.root), "error": str(exc)})
            continue
        evaluator = EvalDataExtraction(
            {
                "training_data_count": train_count,
                "training_data_split": split_strategy,
                "training_data_split_seed": split_seed,
            },
            data_loader=loader,
        )
        dataset_total_docs = len(loader.doc_ids)
        raw: list[dict[str, Any]] = []
        if not join_only:
            for table, table_df in tables.items():
                if table_df.empty or "doc_id" not in table_df.columns:
                    continue
                table_total_docs = int(table_df["doc_id"].astype(str).nunique())
                all_rows_bucket = _nearest_bucket(table_total_docs / dataset_total_docs if dataset_total_docs else 0.0, buckets)
                candidate = _all_rows_candidate(ref, table, table_df, all_rows_bucket, dataset_total_docs)
                if candidate:
                    raw.append(candidate)
                for column in _usable_columns(table_df):
                    raw.extend(
                        _nullness_candidates(
                            ref=ref,
                            table=table,
                            table_df=table_df,
                            column=column,
                            buckets=buckets,
                            dataset_total_docs=dataset_total_docs,
                        )
                    )
                    raw.extend(
                        _numeric_candidates(
                            ref=ref,
                            table=table,
                            table_df=table_df,
                            column=column,
                            buckets=buckets,
                            dataset_total_docs=dataset_total_docs,
                        )
                    )
                    raw.extend(
                        _categorical_candidates(
                            ref=ref,
                            table=table,
                            table_df=table_df,
                            column=column,
                            buckets=buckets,
                            max_in_values=max_in_values,
                            dataset_total_docs=dataset_total_docs,
                        )
                    )
        if include_joins:
            for pair in infer_join_pairs(tables, max_pairs=max_join_pairs):
                if not left_joins_only:
                    raw.extend(
                        _join_predicate_candidates(
                            ref=ref,
                            pair=pair,
                            tables=tables,
                            buckets=buckets,
                            max_in_values=max_in_values,
                            dataset_total_docs=dataset_total_docs,
                            max_predicate_columns=max_join_predicate_columns,
                        )
                    )
                raw.extend(
                    _left_join_predicate_candidates(
                        ref=ref,
                        pair=pair,
                        tables=tables,
                        buckets=buckets,
                        max_in_values=max_in_values,
                        dataset_total_docs=dataset_total_docs,
                        max_predicate_columns=max_join_predicate_columns,
                    )
                )
        raw = _prefilter_raw_candidates(
            raw,
            buckets=buckets,
            max_per_bucket=max_raw_candidates_per_bucket,
        )
        print(f"[generate] {ref.dataset_id}: raw_after_prefilter={len(raw)}", flush=True)

        seen_sql: set[str] = set()
        validated: list[dict[str, Any]] = []
        for candidate in raw:
            sql = str(candidate.get("sql") or "")
            if not sql or sql in seen_sql:
                continue
            seen_sql.add(sql)
            try:
                profile = profile_query(
                    dataset=ref,
                    loader=loader,
                    evaluator=evaluator,
                    query_id=str(candidate["query_id"]),
                    query_info={
                        "query_id": candidate["query_id"],
                        "question": candidate["question"],
                        "sql": candidate["sql"],
                        "required_tables": candidate["required_tables"],
                        "required_columns": candidate["required_columns"],
                        "output_columns": candidate["output_columns"],
                        "tables": candidate["required_tables"],
                        "attributes": candidate["required_columns"],
                    },
                    train_count=train_count,
                    split_strategy=split_strategy,
                    split_seed=split_seed,
                    min_docs=min_docs,
                    min_cells=min_cells,
                )
            except Exception as exc:
                candidate["validation_error"] = str(exc)
                continue
            candidate["profile"] = profile
            actual = profile.get("all_relevant_doc_rate")
            target = candidate.get("meta", {}).get("target_doc_coverage")
            candidate["selection_score"] = abs(float(actual or 0.0) - float(target or 0.0))
            validated.append(candidate)
        print(f"[generate] {ref.dataset_id}: validated={len(validated)}", flush=True)

        for bucket in buckets:
            bucket_candidates = [
                item
                for item in validated
                if abs(float(item.get("meta", {}).get("coverage_bucket", -1)) - bucket) < 1e-12
                and item.get("profile", {}).get("all_relevant_docs", 0) > 0
            ]
            generated.extend(
                _select_diverse_candidates(
                    bucket_candidates,
                    per_bucket=per_bucket,
                    min_join_per_bucket=min_join_per_bucket,
                )
            )
    return generated


def _flatten_profile_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("all_docs_by_table", "eval_docs_by_table"):
            if isinstance(item.get(key), dict):
                item[key] = json.dumps(item[key], ensure_ascii=False, sort_keys=True)
        flattened.append(item)
    return pd.DataFrame(flattened)


def _flatten_candidate_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "dataset_id": row.get("dataset_id"),
            "query_id": row.get("query_id"),
            "question": row.get("question"),
            "sql": row.get("sql"),
            "required_columns": json.dumps(row.get("required_columns", []), ensure_ascii=False),
            "output_columns": json.dumps(row.get("output_columns", []), ensure_ascii=False),
        }
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
        for key in (
            "coverage_bucket",
            "target_doc_coverage",
            "actual_doc_coverage",
            "actual_table_doc_coverage",
            "result_docs",
            "total_docs",
            "table_total_docs",
            "predicate_table",
            "predicate_column",
            "operator",
            "is_join",
            "join_shared_values",
            "join_score",
        ):
            item[f"meta_{key}"] = meta.get(key)
        item["meta_value"] = json.dumps(meta.get("value"), ensure_ascii=False)
        item["meta_join_tables"] = json.dumps(meta.get("join_tables", []), ensure_ascii=False)
        item["meta_join_columns"] = json.dumps(meta.get("join_columns", []), ensure_ascii=False)
        for key in (
            "all_relevant_docs",
            "all_required_cells",
            "all_relevant_doc_rate",
            "eval_relevant_docs",
            "eval_required_cells",
            "eval_relevant_doc_rate",
            "curation_status",
            "provenance_status",
        ):
            item[f"profile_{key}"] = profile.get(key)
        flattened.append(item)
    return pd.DataFrame(flattened)


def _write_summary(path: Path, profile_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    total_queries = sum(1 for row in profile_rows if row.get("query_id"))
    keep = sum(1 for row in profile_rows if row.get("curation_status") == "keep")
    remove = sum(1 for row in profile_rows if str(row.get("curation_status", "")).startswith("remove"))
    generated = sum(1 for row in candidates if row.get("query_id"))
    by_status: dict[str, int] = {}
    for row in profile_rows:
        status = str(row.get("curation_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    lines = [
        "# Query Curation Summary",
        "",
        f"- Existing profiled queries: {total_queries}",
        f"- Existing queries marked keep: {keep}",
        f"- Existing queries marked remove/review: {remove}",
        f"- Generated candidate queries: {generated}",
        "",
        "## Existing Query Status",
        "",
    ]
    for status, count in sorted(by_status.items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Generated Candidates By Dataset", ""])
    by_dataset: dict[str, int] = {}
    for row in candidates:
        dataset_id = str(row.get("dataset_id") or "unknown")
        by_dataset[dataset_id] = by_dataset.get(dataset_id, 0) + 1
    for dataset_id, count in sorted(by_dataset.items()):
        lines.append(f"- `{dataset_id}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/derived")
    parser.add_argument("--output-dir", default="outputs/query_curation")
    parser.add_argument("--train-count", type=int, default=25)
    parser.add_argument("--split-strategy", default="prefix")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-docs", type=int, default=3)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--coverage-buckets", default=",".join(str(value) for value in DEFAULT_BUCKETS))
    parser.add_argument("--candidates-per-bucket", type=int, default=6)
    parser.add_argument("--min-join-per-bucket", type=int, default=1)
    parser.add_argument("--max-in-values", type=int, default=8)
    parser.add_argument("--max-join-pairs", type=int, default=32)
    parser.add_argument("--max-join-predicate-columns", type=int, default=8)
    parser.add_argument("--max-raw-candidates-per-bucket", type=int, default=180)
    parser.add_argument("--no-joins", action="store_true")
    parser.add_argument("--join-only", action="store_true", help="Generate only join candidates.")
    parser.add_argument("--left-joins-only", action="store_true", help="When generating joins, skip inner-join candidates.")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--dataset-id-contains", action="append", default=[])
    parser.add_argument("--exclude-dataset-id-contains", action="append", default=[])
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Skip generated DeepSeek smoke/aligned/repair/ablation datasets.",
    )
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = _repo_root()
    dataset_root = (repo / args.dataset_root).resolve()
    output_dir = (repo / args.output_dir).resolve()
    buckets = tuple(float(value.strip()) for value in str(args.coverage_buckets).split(",") if value.strip())
    datasets = _discover_datasets(dataset_root)
    if args.dataset_id_contains:
        needles = [str(value) for value in args.dataset_id_contains]
        datasets = [dataset for dataset in datasets if any(needle in dataset.dataset_id for needle in needles)]
    if args.exclude_dataset_id_contains:
        needles = [str(value) for value in args.exclude_dataset_id_contains]
        datasets = [dataset for dataset in datasets if not any(needle in dataset.dataset_id for needle in needles)]
    if args.canonical_only:
        datasets = [
            dataset
            for dataset in datasets
            if not dataset.dataset_id.startswith("deepseek_")
            and "/deepseek_" not in dataset.root.as_posix()
        ]
    if args.max_datasets > 0:
        datasets = datasets[: args.max_datasets]

    profile_rows = profile_existing_queries(
        datasets,
        train_count=args.train_count,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        min_docs=args.min_docs,
        min_cells=args.min_cells,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "query_profile.json", profile_rows)
    _flatten_profile_rows(profile_rows).to_csv(output_dir / "query_profile.csv", index=False)

    candidates: list[dict[str, Any]] = []
    if not args.profile_only:
        candidates = generate_candidates(
            datasets,
            buckets=buckets,
            per_bucket=args.candidates_per_bucket,
            min_join_per_bucket=args.min_join_per_bucket,
            max_in_values=args.max_in_values,
            include_joins=not args.no_joins,
            join_only=args.join_only,
            left_joins_only=args.left_joins_only,
            max_join_pairs=args.max_join_pairs,
            max_join_predicate_columns=args.max_join_predicate_columns,
            max_raw_candidates_per_bucket=args.max_raw_candidates_per_bucket,
            min_docs=args.min_docs,
            min_cells=args.min_cells,
            train_count=args.train_count,
            split_strategy=args.split_strategy,
            split_seed=args.split_seed,
        )
        _write_json(output_dir / "generated_query_candidates.json", candidates)
        _flatten_candidate_rows(candidates).to_csv(output_dir / "generated_query_candidates.csv", index=False)
    _write_summary(output_dir / "curation_summary.md", profile_rows, candidates)

    print(f"datasets={len(datasets)} profile_rows={len(profile_rows)} candidates={len(candidates)}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
