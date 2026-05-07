#!/usr/bin/env python3
"""Summarize a paper-like analogous LLM extraction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _eval_files(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("**/data_extraction/**/eval_*.json"))


def _ratio_add(acc: dict[str, int], metric: dict[str, Any], prefix: str) -> None:
    acc[f"{prefix}_covered"] += int(metric.get("covered", 0) or 0)
    acc[f"{prefix}_total"] += int(metric.get("total", 0) or 0)


def summarize_run(run_root: Path) -> dict[str, Any]:
    files = _eval_files(run_root)
    totals = {
        "table_covered": 0,
        "table_total": 0,
        "cell_covered": 0,
        "cell_total": 0,
        "answer_covered": 0,
        "answer_total": 0,
        "queries": 0,
        "can_answer": 0,
    }
    rows = []
    for path in files:
        data = _read_json(path)
        qa = data.get("query_aware", {}) if isinstance(data, dict) else {}
        if not qa:
            continue
        totals["queries"] += 1
        summary = qa.get("summary", {})
        if summary.get("can_answer_query"):
            totals["can_answer"] += 1
        table = qa.get("table_assignment", {})
        cell = qa.get("cell_recall", {})
        answer = qa.get("answer_recall", {})
        _ratio_add(totals, table, "table")
        _ratio_add(totals, cell, "cell")
        _ratio_add(totals, answer, "answer")
        rows.append(
            {
                "path": str(path),
                "query_id": qa.get("query_id"),
                "table_recall": table.get("recall"),
                "cell_recall": cell.get("recall"),
                "answer_recall": answer.get("recall"),
                "answer_precision": answer.get("precision"),
            }
        )
    metrics = {
        "table_recall": totals["table_covered"] / totals["table_total"] if totals["table_total"] else None,
        "cell_recall": totals["cell_covered"] / totals["cell_total"] if totals["cell_total"] else None,
        "answer_recall": totals["answer_covered"] / totals["answer_total"] if totals["answer_total"] else None,
        "queries": totals["queries"],
        "can_answer": totals["can_answer"],
    }
    return {"run_root": str(run_root), "metrics": metrics, "totals": totals, "rows": rows}


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# ReDD Analogous LLM Run Summary",
        "",
        f"Run root: `{summary['run_root']}`",
        "",
        "## Metrics",
        "",
        f"- Queries: {metrics['queries']}",
        f"- Can answer: {metrics['can_answer']}",
        f"- Table recall: {metrics['table_recall']}",
        f"- Cell recall: {metrics['cell_recall']}",
        f"- Answer recall: {metrics['answer_recall']}",
        "",
        "## Rows",
        "",
        "| Query | Table recall | Cell recall | Answer recall | Path |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row.get('query_id')} | {row.get('table_recall')} | {row.get('cell_recall')} | {row.get('answer_recall')} | `{row.get('path')}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def _analogous_result(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    full = (
        metrics.get("queries", 0) > 0
        and metrics.get("table_recall") == 1.0
        and metrics.get("cell_recall") == 1.0
        and metrics.get("answer_recall") == 1.0
    )
    return {
        "experiment_id": "table2_data_population_accuracy",
        "status": "analogous_supported" if full else "partial",
        "observed": (
            f"Analogous LLM extraction run: queries={metrics.get('queries')}, "
            f"table_recall={metrics.get('table_recall')}, cell_recall={metrics.get('cell_recall')}, "
            f"answer_recall={metrics.get('answer_recall')}; summary={summary['run_root']}/reports/redd_paper_analogous_summary.json"
        ),
        "comparison": (
            "Paper-like Table 2 smoke evidence for LLM data population on the current demo dataset; "
            "not exact paper Table 2 reproduction."
        ),
        "next_action": "Scale the analogous run to the target datasets and add correction/cost variants.",
    }


def _merge_analogous_result(path: Path, result: dict[str, Any]) -> None:
    payload = _read_json(path) if path.exists() else {}
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
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--paper-output-root", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = summarize_run(run_root)
    summary_path = reports / "redd_paper_analogous_summary.json"
    summary_md_path = reports / "redd_paper_analogous_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary_md_path, summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {summary_md_path}")

    if args.paper_output_root:
        paper_reports = Path(args.paper_output_root) / "reports"
        paper_reports.mkdir(parents=True, exist_ok=True)
        path = paper_reports / "redd_paper_analogous_results.json"
        _merge_analogous_result(path, _analogous_result(summary))
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
