"""Dataset text/ground-truth consistency diagnostics."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from redd.core.utils.sql_filter_parser import SQLFilterParser
from redd.proxy.predicate_proxy.heuristic_proxy import (
    _explicit_attribute_numbers,
    _has_explicit_attribute_parser,
    _nearby_numbers,
    _predicate_value,
    _satisfies,
)


def audit_dataset_consistency(
    dataset_id: str,
    root: str | Path,
    *,
    query_ids: Sequence[str] | None = None,
    queries_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report explicit document text that disagrees with GT predicate outcomes."""
    import pandas as pd

    data_root = Path(root)
    documents = pd.read_parquet(data_root / "data" / "documents.parquet")
    ground_truth = pd.read_parquet(data_root / "data" / "ground_truth.parquet")
    queries_file = data_root / queries_path if queries_path else data_root / "metadata" / "queries.json"
    queries_payload = json.loads(queries_file.read_text(encoding="utf-8"))
    queries = (
        queries_payload.get("queries", [])
        if isinstance(queries_payload, dict)
        else queries_payload
    )
    allowed_query_ids = {str(query_id) for query_id in query_ids or [] if str(query_id)}

    doc_text_by_id = {
        str(row.get("doc_id")): str(row.get("doc_text") or "")
        for row in documents.to_dict("records")
    }
    doc_metadata_by_id = {
        str(row.get("doc_id")): {
            "source_table": row.get("source_table"),
            "source_row_id": row.get("source_row_id"),
        }
        for row in documents.to_dict("records")
    }
    gt_values = _gt_values_by_doc_attribute(ground_truth)
    parser = SQLFilterParser(strip_table_aliases=True)

    conflicts: list[dict[str, Any]] = []
    checked_doc_predicates = 0
    skipped_without_explicit_parser = 0
    skipped_without_text_evidence = 0

    for query in queries:
        query_id = str(query.get("query_id") or query.get("id") or "")
        if allowed_query_ids and query_id not in allowed_query_ids:
            continue
        sql = str(query.get("sql") or query.get("query") or "")
        for predicate in parser.parse(sql):
            attribute = str(predicate.attribute or "").lower()
            expected = _predicate_value(predicate.value)
            is_numeric_predicate = isinstance(expected, (int, float))
            is_string_predicate = _is_supported_string_predicate(predicate.operator, expected)
            if not is_numeric_predicate and not is_string_predicate:
                skipped_without_explicit_parser += 1
                continue

            for (doc_id, column_name), raw_gt_values in gt_values.items():
                if column_name != attribute:
                    continue
                text = doc_text_by_id.get(doc_id, "")
                if is_numeric_predicate:
                    text_candidates = _explicit_attribute_numbers(text, attribute)
                    if not text_candidates and not _has_explicit_attribute_parser(attribute):
                        text_candidates = _nearby_numbers(text, attribute)
                    if not text_candidates:
                        skipped_without_text_evidence += 1
                        continue

                    gt_candidates = []
                    for value in raw_gt_values:
                        candidate = _predicate_value(value)
                        if isinstance(candidate, (int, float)):
                            gt_candidates.append(candidate)
                    if not gt_candidates:
                        continue

                    text_passes = any(
                        _satisfies(candidate, predicate.operator, expected)
                        for candidate in text_candidates
                    )
                else:
                    text_candidates = _string_text_candidates(
                        text,
                        predicate.operator,
                        expected,
                    )
                    gt_candidates = [
                        candidate
                        for candidate in (_predicate_value(value) for value in raw_gt_values)
                        if isinstance(candidate, str) and candidate.strip()
                    ]
                    if not gt_candidates:
                        continue

                    text_passes = _string_text_satisfies(
                        text,
                        predicate.operator,
                        expected,
                    )

                checked_doc_predicates += 1
                gt_passes = any(
                    _predicate_satisfies(candidate, predicate.operator, expected)
                    for candidate in gt_candidates
                )
                if text_passes == gt_passes:
                    continue

                doc_metadata = doc_metadata_by_id.get(doc_id) or {}
                conflicts.append(
                    {
                        "dataset": dataset_id,
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "source_table": doc_metadata.get("source_table"),
                        "source_row_id": doc_metadata.get("source_row_id"),
                        "attribute": attribute,
                        "operator": predicate.operator,
                        "expected": expected,
                        "conflict_type": (
                            "text_pass_gt_fail" if text_passes else "gt_pass_text_fail"
                        ),
                        "text_values": text_candidates,
                        "ground_truth_values": gt_candidates,
                        "text_excerpt": _text_excerpt(text, text_candidates),
                    }
                )

    by_type: dict[str, int] = {}
    for conflict in conflicts:
        kind = str(conflict["conflict_type"])
        by_type[kind] = by_type.get(kind, 0) + 1
    return {
        "dataset": dataset_id,
        "checked_doc_predicates": checked_doc_predicates,
        "skipped_without_explicit_parser": skipped_without_explicit_parser,
        "skipped_without_text_evidence": skipped_without_text_evidence,
        "conflict_count": len(conflicts),
        "conflicts_by_type": by_type,
        "conflicts": sorted(
            conflicts,
            key=lambda item: (
                item["conflict_type"],
                item["dataset"],
                item["query_id"],
                item["doc_id"],
                item["attribute"],
            ),
        ),
    }


def build_dataset_consistency_audit(
    datasets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a diagnostic-only consistency audit for configured datasets."""
    dataset_reports = [
        audit_dataset_consistency(
            dataset_id,
            dataset_config["root"],
            query_ids=dataset_config.get("query_ids"),
            queries_path=_dataset_queries_path(dataset_config),
        )
        for dataset_id, dataset_config in datasets.items()
    ]
    conflicts = [
        conflict
        for dataset in dataset_reports
        for conflict in dataset["conflicts"]
    ]
    by_type: dict[str, int] = {}
    for conflict in conflicts:
        kind = str(conflict["conflict_type"])
        by_type[kind] = by_type.get(kind, 0) + 1

    return {
        "generated_at_unix": time.time(),
        "purpose": (
            "Report explicit text evidence that disagrees with ground-truth predicate "
            "outcomes. This is diagnostic only and is not used for optimizer ranking."
        ),
        "total_conflicts": len(conflicts),
        "conflicts_by_type": by_type,
        "datasets": dataset_reports,
    }


def write_dataset_consistency_audit(output_dir: str | Path, audit: Mapping[str, Any]) -> None:
    """Write JSON, Markdown, and flat conflict exports for consistency audit reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_consistency_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    (root / "dataset_consistency_audit.md").write_text(
        _dataset_consistency_audit_markdown(audit),
        encoding="utf-8",
    )
    conflicts = _flat_audit_conflicts(audit)
    (root / "dataset_consistency_audit_conflicts.jsonl").write_text(
        "".join(json.dumps(conflict, sort_keys=True) + "\n" for conflict in conflicts),
        encoding="utf-8",
    )
    with (root / "dataset_consistency_audit_conflicts.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fieldnames = [
            "conflict_ref",
            "dataset",
            "query_id",
            "doc_id",
            "source_table",
            "source_row_id",
            "attribute",
            "operator",
            "expected",
            "conflict_type",
            "text_values",
            "ground_truth_values",
            "text_excerpt",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for conflict in conflicts:
            row = {key: conflict.get(key, "") for key in fieldnames}
            row["text_values"] = json.dumps(conflict.get("text_values") or [])
            row["ground_truth_values"] = json.dumps(
                conflict.get("ground_truth_values") or []
            )
            writer.writerow(row)


def _flat_audit_conflicts(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for dataset in audit.get("datasets") or []:
        if not isinstance(dataset, Mapping):
            continue
        for conflict in dataset.get("conflicts") or []:
            if not isinstance(conflict, Mapping):
                continue
            row = dict(conflict)
            row["conflict_ref"] = "::".join(
                str(part or "")
                for part in (
                    row.get("dataset"),
                    row.get("query_id"),
                    row.get("doc_id"),
                    row.get("attribute"),
                )
            )
            conflicts.append(row)
    return sorted(
        conflicts,
        key=lambda item: (
            str(item.get("dataset") or ""),
            str(item.get("query_id") or ""),
            str(item.get("doc_id") or ""),
            str(item.get("attribute") or ""),
        ),
    )


def _gt_values_by_doc_attribute(ground_truth: Any) -> dict[tuple[str, str], list[Any]]:
    values: dict[tuple[str, str], list[Any]] = {}
    for row in ground_truth.to_dict("records"):
        doc_id = str(row.get("doc_id"))
        column_name = str(row.get("column_name") or "").lower()
        if not doc_id or not column_name:
            continue
        values.setdefault((doc_id, column_name), []).append(row.get("value"))
    return values


def _dataset_queries_path(dataset_config: Mapping[str, Any]) -> str | Path | None:
    loader_options = dataset_config.get("loader_options")
    if not isinstance(loader_options, Mapping):
        return None
    filemap = loader_options.get("filemap")
    if not isinstance(filemap, Mapping):
        return None
    queries = filemap.get("queries")
    return queries if isinstance(queries, (str, Path)) and str(queries) else None


def _is_supported_string_predicate(operator: str, expected: Any) -> bool:
    if not isinstance(expected, str) or not expected.strip():
        return False
    return str(operator or "").strip().upper() in {"=", "LIKE"}


def _predicate_satisfies(value: Any, operator: str, expected: Any) -> bool:
    if isinstance(value, str) and isinstance(expected, str):
        op = str(operator or "").strip().upper()
        if op == "LIKE":
            return _string_contains_like(value, expected)
    return _satisfies(value, operator, expected)


def _string_text_candidates(text: str, operator: str, expected: str) -> list[str]:
    needle = _string_needle(operator, expected)
    return [expected] if needle and needle in _norm_for_contains(text) else []


def _string_text_satisfies(text: str, operator: str, expected: str) -> bool:
    normalized = _string_needle(operator, expected)
    return bool(normalized and normalized in _norm_for_contains(text))


def _string_contains_like(value: Any, expected: Any) -> bool:
    needle = _string_needle("LIKE", str(expected or ""))
    return bool(needle and needle in _norm_for_contains(value))


def _string_needle(operator: str, expected: str) -> str:
    needle = str(expected or "")
    if str(operator or "").strip().upper() == "LIKE":
        needle = needle.replace("%", " ").replace("_", " ")
    return _norm_for_contains(needle)


def _norm_for_contains(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _text_excerpt(text: str, values: list[Any], max_len: int = 220) -> str:
    source = " ".join(str(text or "").split())
    if not source:
        return ""
    for value in values:
        if isinstance(value, (int, float)):
            needle = str(int(value)) if float(value).is_integer() else str(value)
            index = source.find(needle)
        else:
            needle = str(value or "")
            index = source.casefold().find(needle.casefold())
        if index >= 0:
            start = max(index - max_len // 3, 0)
            return source[start : start + max_len]
    return source[:max_len]


def _dataset_consistency_audit_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Dataset Consistency Audit",
        "",
        "Diagnostic only: these conflicts are not used for optimizer ranking.",
        "",
        f"- Total conflicts: `{audit['total_conflicts']}`",
        f"- Conflicts by type: `{json.dumps(audit['conflicts_by_type'], sort_keys=True)}`",
        "",
    ]
    for dataset in audit["datasets"]:
        lines.extend(
            [
                f"## {dataset['dataset']}",
                f"- Checked doc-predicates: `{dataset['checked_doc_predicates']}`",
                f"- Conflicts: `{dataset['conflict_count']}`",
                f"- By type: `{json.dumps(dataset['conflicts_by_type'], sort_keys=True)}`",
                "",
            ]
        )
        for conflict in dataset["conflicts"][:20]:
            lines.extend(
                [
                    (
                        f"- `{conflict['query_id']}` `{conflict['doc_id']}` "
                        f"`{conflict['attribute']} {conflict['operator']} "
                        f"{conflict['expected']}`: `{conflict['conflict_type']}`"
                    ),
                    (
                        f"  - text={conflict['text_values']} "
                        f"gt={conflict['ground_truth_values']}"
                    ),
                    f"  - excerpt: {conflict['text_excerpt']}",
                ]
            )
        if len(dataset["conflicts"]) > 20:
            lines.append(f"- ... {len(dataset['conflicts']) - 20} more conflicts")
        lines.append("")
    return "\n".join(lines)
