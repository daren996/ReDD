#!/usr/bin/env python3
"""Summarize current canonical datasets as paper-like Table 1 evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _count_parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None
    return int(len(pd.read_parquet(path)))


def summarize_datasets(dataset_root: Path) -> dict[str, Any]:
    rows = []
    family_counts: Counter[str] = Counter()
    total_queries = 0
    total_docs = 0
    for manifest_path in sorted(dataset_root.glob("*/manifest.yaml")):
        dataset_dir = manifest_path.parent
        dataset_id = dataset_dir.name
        family = dataset_id.split(".")[0]
        queries = _read_json(dataset_dir / "metadata" / "queries.json", {})
        query_count = len(queries) if isinstance(queries, (dict, list)) else 0
        doc_count = _count_parquet_rows(dataset_dir / "data" / "documents.parquet")
        family_counts[family] += query_count
        total_queries += query_count
        if doc_count is not None:
            total_docs += doc_count
        rows.append(
            {
                "dataset_id": dataset_id,
                "family": family,
                "query_count": query_count,
                "document_count": doc_count,
                "manifest": str(manifest_path),
            }
        )
    return {
        "dataset_root": str(dataset_root),
        "dataset_count": len(rows),
        "total_queries": total_queries,
        "total_documents": total_docs,
        "query_counts_by_family": dict(sorted(family_counts.items())),
        "rows": rows,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ReDD Dataset Setup Summary",
        "",
        f"Dataset root: `{summary['dataset_root']}`",
        f"Datasets: {summary['dataset_count']}",
        f"Queries: {summary['total_queries']}",
        f"Documents: {summary['total_documents']}",
        "",
        "## Query Counts By Family",
        "",
    ]
    for family, count in summary["query_counts_by_family"].items():
        lines.append(f"- {family}: {count}")
    lines.extend(
        [
            "",
            "## Datasets",
            "",
            "| Dataset | Queries | Documents |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(f"| {row['dataset_id']} | {row['query_count']} | {row['document_count']} |")
    lines.append("")
    path.write_text("\n".join(lines))


def _analogous_result(summary: dict[str, Any]) -> dict[str, Any]:
    ok = summary.get("dataset_count", 0) > 0 and summary.get("total_queries", 0) > 0
    return {
        "experiment_id": "table1_dataset_setup",
        "status": "analogous_supported" if ok else "missing",
        "observed": (
            f"Analogous current dataset setup: datasets={summary.get('dataset_count')}, "
            f"queries={summary.get('total_queries')}, documents={summary.get('total_documents')}, "
            f"query_counts_by_family={summary.get('query_counts_by_family')}; "
            f"summary={summary.get('summary_path')}"
        ),
        "comparison": (
            "Paper-like Table 1 evidence for the current canonical dataset setup; "
            "not exact paper Table 1 reproduction."
        ),
        "next_action": "Use the exact paper split to reproduce Table 1 verbatim.",
    }


def _merge_analogous_result(path: Path, result: dict[str, Any]) -> None:
    payload = _read_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    existing = payload.get("results", [])
    if not isinstance(existing, list):
        existing = []
    experiment_id = result["experiment_id"]
    merged = [item for item in existing if not (isinstance(item, dict) and item.get("experiment_id") == experiment_id)]
    merged.append(result)
    payload.update(
        {
            "evidence_mode": "analogous",
            "notes": [
                "This file contains paper-like analogous LLM experiment evidence.",
                "It does not claim exact reproduction of the original paper tables.",
            ],
            "results": merged,
        }
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-root", default="outputs/deepseek_analogous_single_doc")
    parser.add_argument("--paper-output-root", default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    run_root = Path(args.run_root)
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = summarize_datasets(dataset_root)
    summary_path = reports / "redd_paper_dataset_setup_summary.json"
    summary_md_path = reports / "redd_paper_dataset_setup_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary_md_path, summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_md_path}")

    if args.paper_output_root:
        paper_reports = Path(args.paper_output_root) / "reports"
        paper_reports.mkdir(parents=True, exist_ok=True)
        path = paper_reports / "redd_paper_analogous_results.json"
        result = _analogous_result({**summary, "summary_path": str(summary_path)})
        _merge_analogous_result(path, result)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
