"""Apply generated query-curation candidates and report bucket diversity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from query_curation import (
    DEFAULT_BUCKETS,
    _discover_datasets,
    _read_json,
    _records_by_table,
    _repo_root,
    _write_json,
    infer_join_pairs,
)


def _bucket_label(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "No bucket"


def _bucket_value(query: dict[str, Any]) -> float | None:
    meta = query.get("meta") if isinstance(query.get("meta"), dict) else {}
    value = meta.get("coverage_bucket")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_join(query: dict[str, Any]) -> bool:
    meta = query.get("meta") if isinstance(query.get("meta"), dict) else {}
    if meta.get("is_join"):
        return True
    return bool(re.search(r"\bJOIN\b", str(query.get("sql") or ""), flags=re.IGNORECASE))


def _predicate(query: dict[str, Any]) -> str:
    meta = query.get("meta") if isinstance(query.get("meta"), dict) else {}
    predicate = meta.get("predicate_column")
    if predicate:
        return str(predicate)
    return "ALL"


def _operator(query: dict[str, Any]) -> str:
    meta = query.get("meta") if isinstance(query.get("meta"), dict) else {}
    return str(meta.get("operator") or "")


def _payload_queries(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return [dict(item) for item in payload["queries"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        result = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("query_id", str(key))
            result.append(item)
        return result
    return []


def _canonical_refs(dataset_root: Path, canonical_only: bool):
    refs = _discover_datasets(dataset_root)
    if not canonical_only:
        return refs
    return [
        ref
        for ref in refs
        if not ref.dataset_id.startswith("deepseek_")
        and "/deepseek_" not in ref.root.as_posix()
    ]


def _query_record(candidate: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    meta = dict(candidate.get("meta") or {})
    profile = dict(candidate.get("profile") or {})
    root_text = str(profile.get("dataset_root") or dataset_root)
    query_id = str(candidate["query_id"])
    tags = ["query_curation", "generated_sql_first", "bucket_candidate"]
    if _is_join(candidate):
        tags.append("join_query")
    return {
        "query_id": query_id,
        "question": candidate.get("question") or candidate.get("query") or "",
        "sql": candidate.get("sql") or "",
        "required_tables": [str(value) for value in candidate.get("required_tables") or []],
        "required_columns": [str(value) for value in candidate.get("required_columns") or []],
        "output_columns": [str(value) for value in candidate.get("output_columns") or []],
        "tags": tags,
        "difficulty": None,
        "meta": {
            **meta,
            "dataset_id": candidate.get("dataset_id"),
            "source_dataset_id": candidate.get("dataset_id"),
            "source_dataset_root": root_text,
            "source_query_id": query_id,
            "curation_status": profile.get("curation_status"),
            "profile_all_relevant_docs": profile.get("all_relevant_docs"),
            "profile_all_required_cells": profile.get("all_required_cells"),
            "profile_all_relevant_doc_rate": profile.get("all_relevant_doc_rate"),
            "profile_eval_relevant_docs": profile.get("eval_relevant_docs"),
            "profile_eval_required_cells": profile.get("eval_required_cells"),
            "profile_eval_relevant_doc_rate": profile.get("eval_relevant_doc_rate"),
            "profile_provenance_status": profile.get("provenance_status"),
            "applied_to_source": True,
        },
    }


def apply_candidates(candidates_path: Path, dataset_root: Path, *, canonical_only: bool) -> list[dict[str, Any]]:
    candidates = _read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError(f"Candidate file must contain a list: {candidates_path}")

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        profile = candidate.get("profile") if isinstance(candidate.get("profile"), dict) else {}
        if profile.get("curation_status") != "keep":
            continue
        dataset_id = str(candidate.get("dataset_id") or "")
        if dataset_id:
            by_dataset.setdefault(dataset_id, []).append(candidate)

    applied: list[dict[str, Any]] = []
    for ref in _canonical_refs(dataset_root, canonical_only):
        selected = by_dataset.get(ref.dataset_id, [])
        records: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}
        for candidate in selected:
            record = _query_record(candidate, ref.root)
            base_id = record["query_id"]
            count = seen_ids.get(base_id, 0)
            seen_ids[base_id] = count + 1
            if count:
                record["query_id"] = f"{base_id}_{count + 1}"
                record["meta"]["source_query_id"] = base_id
            records.append(record)

        payload = {
            "schema_version": "redd.queries.v1",
            "dataset_id": ref.dataset_id,
            "queries": records,
        }
        _write_json(ref.root / "metadata" / "queries.json", payload)
        applied.append({"dataset_id": ref.dataset_id, "dataset_root": str(ref.root), "queries": len(records)})
    return applied


def _join_capability(ref) -> dict[str, Any]:
    try:
        tables = _records_by_table(ref)
        pairs = infer_join_pairs(tables, max_pairs=1000)
    except Exception as exc:
        return {
            "Dataset": ref.dataset_id,
            "Tables": None,
            "Inferred join pairs": 0,
            "Join capable": False,
            "Top join pairs": "",
            "Error": str(exc),
        }
    top_pairs = [
        f"{pair.left_table}.{pair.left_column}={pair.right_table}.{pair.right_column}({pair.shared_values})"
        for pair in pairs[:8]
    ]
    return {
        "Dataset": ref.dataset_id,
        "Tables": len(tables),
        "Inferred join pairs": len(pairs),
        "Join capable": bool(pairs),
        "Top join pairs": "; ".join(top_pairs),
        "Error": "",
    }


def write_reports(
    dataset_root: Path,
    output_dir: Path,
    *,
    canonical_only: bool,
    buckets: tuple[float, ...],
    target_per_bucket: int,
    min_join_per_bucket: int,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = _canonical_refs(dataset_root, canonical_only)
    bucket_labels = [_bucket_label(bucket) for bucket in buckets]

    capability_rows = [_join_capability(ref) for ref in refs]
    capability = pd.DataFrame(capability_rows)
    join_capable = {
        str(row["Dataset"]): bool(row["Join capable"])
        for row in capability_rows
    }

    count_rows: list[dict[str, Any]] = []
    join_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    for ref in refs:
        queries_path = ref.root / "metadata" / "queries.json"
        queries = _payload_queries(queries_path) if queries_path.exists() else []
        total_by_bucket = {label: 0 for label in bucket_labels}
        join_by_bucket = {label: 0 for label in bucket_labels}
        no_bucket = 0
        for query in queries:
            value = _bucket_value(query)
            label = _bucket_label(value) if value is not None else "No bucket"
            if label not in total_by_bucket:
                no_bucket += 1
                continue
            total_by_bucket[label] += 1
            if _is_join(query):
                join_by_bucket[label] += 1
            usage_rows.append(
                {
                    "Dataset": ref.dataset_id,
                    "Bucket": label,
                    "Query": query.get("query_id"),
                    "Join": _is_join(query),
                    "Predicate": _predicate(query),
                    "Operator": _operator(query),
                    "Tables": ", ".join(str(value) for value in query.get("required_tables") or []),
                }
            )

        count_rows.append({"Dataset": ref.dataset_id, "Total": len(queries), **total_by_bucket, "No bucket": no_bucket})
        join_rows.append({"Dataset": ref.dataset_id, "Join queries": sum(join_by_bucket.values()), **join_by_bucket})

        present_buckets = []
        all_predicates = set()
        for label in bucket_labels:
            bucket_queries = [
                query
                for query in queries
                if _bucket_value(query) is not None and _bucket_label(_bucket_value(query)) == label
            ]
            predicates = sorted({_predicate(query) for query in bucket_queries})
            tables = sorted(
                {
                    str(table)
                    for query in bucket_queries
                    for table in (query.get("required_tables") or [])
                }
            )
            operators = sorted({_operator(query) for query in bucket_queries if _operator(query)})
            all_predicates.update(predicates)
            if bucket_queries:
                present_buckets.append(label)
            diversity_rows.append(
                {
                    "Dataset": ref.dataset_id,
                    "Bucket": label,
                    "Queries": len(bucket_queries),
                    "Join queries": sum(1 for query in bucket_queries if _is_join(query)),
                    "Unique predicates": len(predicates),
                    "Predicates": "; ".join(predicates[:24]),
                    "Unique tables": len(tables),
                    "Tables": "; ".join(tables),
                    "Operators": "; ".join(operators),
                }
            )
            join_needed = min_join_per_bucket if join_capable.get(ref.dataset_id, False) else 0
            total_gap = max(0, target_per_bucket - len(bucket_queries))
            join_gap = max(0, join_needed - sum(1 for query in bucket_queries if _is_join(query)))
            if total_gap == 0 and join_gap == 0:
                status = "ok"
            elif not join_capable.get(ref.dataset_id, False) and join_gap == 0:
                status = "needs_more_queries"
            else:
                status = "needs_more_queries_or_joins"
            audit_rows.append(
                {
                    "Dataset": ref.dataset_id,
                    "Bucket": label,
                    "Queries": len(bucket_queries),
                    "Target queries": target_per_bucket,
                    "Query gap": total_gap,
                    "Join queries": sum(1 for query in bucket_queries if _is_join(query)),
                    "Target join queries": join_needed,
                    "Join gap": join_gap,
                    "Join capable": join_capable.get(ref.dataset_id, False),
                    "Unique predicates": len(predicates),
                    "Status": status,
                }
            )
        dataset_rows.append(
            {
                "Dataset": ref.dataset_id,
                "Queries": len(queries),
                "Buckets": ", ".join(present_buckets),
                "Join queries": sum(join_by_bucket.values()),
                "Unique predicates": len(all_predicates),
                "Join capable": join_capable.get(ref.dataset_id, False),
                "Status": "ok" if all(row["Status"] == "ok" for row in audit_rows if row["Dataset"] == ref.dataset_id) else "has_gaps",
            }
        )

    outputs = {
        "query_counts_by_bucket": pd.DataFrame(count_rows),
        "join_counts_by_bucket": pd.DataFrame(join_rows),
        "predicate_diversity_by_bucket": pd.DataFrame(diversity_rows),
        "bucket_coverage_audit": pd.DataFrame(audit_rows),
        "predicate_usage": pd.DataFrame(usage_rows),
        "dataset_join_capability": capability,
        "current_dataset_table": pd.DataFrame(dataset_rows),
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    _write_markdown_summary(output_dir / "summary.md", outputs)
    return outputs


def _write_markdown_summary(path: Path, outputs: dict[str, pd.DataFrame]) -> None:
    dataset_table = outputs["current_dataset_table"]
    audit = outputs["bucket_coverage_audit"]
    ok_buckets = int((audit["Status"] == "ok").sum()) if not audit.empty else 0
    total_buckets = int(len(audit))
    lines = [
        "# Query Curation Dataset Summary",
        "",
        f"- Datasets: {len(dataset_table)}",
        f"- Dataset/bucket cells passing query + join targets: {ok_buckets}/{total_buckets}",
        "",
        "## Dataset Overview",
        "",
        _df_to_markdown(dataset_table) if not dataset_table.empty else "_No rows._",
        "",
        "## Remaining Gaps",
        "",
    ]
    gaps = audit[audit["Status"] != "ok"] if not audit.empty else audit
    if gaps.empty:
        lines.append("_No gaps._")
    else:
        lines.append(
            _df_to_markdown(gaps[
                [
                    "Dataset",
                    "Bucket",
                    "Queries",
                    "Query gap",
                    "Join queries",
                    "Join gap",
                    "Join capable",
                    "Unique predicates",
                ]
            ])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _df_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[column]) for column in df.columns])
    widths = [
        max(len(column), *(len(row[idx]) for row in rows)) if rows else len(column)
        for idx, column in enumerate(columns)
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [
        fmt(columns),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/derived")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--output-dir", default="outputs/query_curation_applied_expanded")
    parser.add_argument("--coverage-buckets", default=",".join(str(value) for value in DEFAULT_BUCKETS))
    parser.add_argument("--target-per-bucket", type=int, default=5)
    parser.add_argument("--min-join-per-bucket", type=int, default=1)
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = _repo_root()
    dataset_root = (repo / args.dataset_root).resolve()
    output_dir = (repo / args.output_dir).resolve()
    buckets = tuple(float(value.strip()) for value in str(args.coverage_buckets).split(",") if value.strip())
    applied: list[dict[str, Any]] = []
    if args.apply:
        if not args.candidates:
            raise ValueError("--apply requires --candidates")
        applied = apply_candidates((repo / args.candidates).resolve(), dataset_root, canonical_only=args.canonical_only)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(applied).to_csv(output_dir / "applied_datasets.csv", index=False)
    outputs = write_reports(
        dataset_root,
        output_dir,
        canonical_only=args.canonical_only,
        buckets=buckets,
        target_per_bucket=args.target_per_bucket,
        min_join_per_bucket=args.min_join_per_bucket,
    )
    print(f"applied={len(applied)}")
    print(f"datasets={len(outputs['current_dataset_table'])}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
