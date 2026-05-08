#!/usr/bin/env python3
"""Summarize a paper-like analogous LLM schema discovery run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text.endswith("s") and len(text) > 3:
        text = text[:-1]
    return text


def _gt_items(dataset_root: Path) -> tuple[set[str], set[str]]:
    schema = _read_json(dataset_root / "metadata" / "schema.json", {})
    tables: set[str] = set()
    attrs: set[str] = set()
    for table in schema.get("tables", []) if isinstance(schema, dict) else []:
        table_name = _norm(table.get("name") or table.get("table_id"))
        if table_name:
            tables.add(table_name)
        for column in table.get("columns", []) or []:
            attr = _norm(column.get("name") or column.get("column_id"))
            if attr:
                attrs.add(attr)
    return tables, attrs


def _generated_schema_files(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("**/preprocessing/**/res_*.json"))


def _usage_models(run_root: Path) -> list[str]:
    labels: list[str] = []
    for path in sorted(run_root.glob("**/llm_usage*.jsonl")):
        with path.open() as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                provider = str(item.get("provider") or "unknown")
                model = str(item.get("response_model") or item.get("configured_model") or "unknown")
                label = f"{provider}:{model}"
                if label not in labels:
                    labels.append(label)
    return labels


def _generated_items(run_root: Path) -> tuple[set[str], set[str], dict[str, list[str]], list[dict[str, Any]], list[str]]:
    tables: set[str] = set()
    attrs: set[str] = set()
    attr_descriptions: dict[str, list[str]] = {}
    rows = []
    files = []
    for path in _generated_schema_files(run_root):
        files.append(str(path))
        payload = _read_json(path, {})
        if not isinstance(payload, dict):
            continue
        for doc_id, doc_result in payload.items():
            if not isinstance(doc_result, dict):
                continue
            assignment = doc_result.get("res")
            if assignment:
                tables.add(_norm(assignment))
            for table in doc_result.get("log", []) or []:
                if not isinstance(table, dict):
                    continue
                table_name = table.get("Schema Name") or table.get("schema") or table.get("table")
                if table_name:
                    tables.add(_norm(table_name))
                for attr_obj in table.get("Attributes", []) or []:
                    if isinstance(attr_obj, dict):
                        attr_name = attr_obj.get("Attribute Name") or next(iter(attr_obj), None)
                        attr_desc = attr_obj.get("Description") or attr_obj.get("description")
                        if attr_desc is None and attr_name in attr_obj:
                            attr_desc = attr_obj.get(attr_name)
                    else:
                        attr_name = attr_obj
                        attr_desc = ""
                    attr = _norm(attr_name)
                    if attr:
                        attrs.add(attr)
                        attr_descriptions.setdefault(attr, []).append(str(attr_desc or ""))
                rows.append(
                    {
                        "path": str(path),
                        "doc_id": doc_id,
                        "assignment": assignment,
                        "table": table_name,
                        "attributes": table.get("Attributes", []),
                    }
                )
    return tables, attrs, attr_descriptions, rows, files


def _semantic_attr_matches(
    gt_attrs: set[str],
    pred_attrs: set[str],
    pred_descriptions: dict[str, list[str]],
) -> tuple[set[str], set[str], dict[str, str]]:
    gt_to_pred: dict[str, str] = {}
    pred_to_gt: dict[str, str] = {}
    unmatched_pred = set(pred_attrs)
    for attr in sorted(gt_attrs & pred_attrs):
        gt_to_pred[attr] = attr
        pred_to_gt[attr] = attr
        unmatched_pred.discard(attr)

    aliases = {
        "appelation": {"appellation", "appellation_region", "region", "wine_region", "designation", "wine_name"},
        "appellation": {"appelation", "region", "wine_region", "designation", "wine_name"},
        "winery": {"winery_name", "producer", "producer_name"},
    }
    for gt_attr in sorted(gt_attrs - set(gt_to_pred)):
        candidates = aliases.get(gt_attr, set())
        for pred_attr in sorted(unmatched_pred):
            desc_text = " ".join(pred_descriptions.get(pred_attr, [])).lower()
            pred_tokens = set(filter(None, pred_attr.split("_")))
            if pred_attr in candidates or candidates & pred_tokens or any(alias in desc_text for alias in candidates):
                gt_to_pred[gt_attr] = pred_attr
                pred_to_gt[pred_attr] = gt_attr
                unmatched_pred.discard(pred_attr)
                break
    return set(gt_to_pred), set(pred_to_gt), gt_to_pred


def _ratio(covered: int, total: int) -> float | None:
    return covered / total if total else None


def summarize_run(run_root: Path, dataset_root: Path) -> dict[str, Any]:
    gt_tables, gt_attrs = _gt_items(dataset_root)
    pred_tables, pred_attrs, pred_descriptions, rows, files = _generated_items(run_root)
    table_hits = gt_tables & pred_tables
    attr_hits = gt_attrs & pred_attrs
    semantic_gt_hits, semantic_pred_hits, semantic_map = _semantic_attr_matches(
        gt_attrs,
        pred_attrs,
        pred_descriptions,
    )
    metrics = {
        "schema_files": len(files),
        "table_recall": _ratio(len(table_hits), len(gt_tables)),
        "table_precision": _ratio(len(table_hits), len(pred_tables)),
        "attribute_recall": _ratio(len(attr_hits), len(gt_attrs)),
        "attribute_precision": _ratio(len(attr_hits), len(pred_attrs)),
        "semantic_attribute_recall": _ratio(len(semantic_gt_hits), len(gt_attrs)),
        "semantic_attribute_precision": _ratio(len(semantic_pred_hits), len(pred_attrs)),
        "gt_tables": sorted(gt_tables),
        "pred_tables": sorted(pred_tables),
        "matched_tables": sorted(table_hits),
        "gt_attributes": sorted(gt_attrs),
        "pred_attributes": sorted(pred_attrs),
        "matched_attributes": sorted(attr_hits),
        "missing_attributes": sorted(gt_attrs - pred_attrs),
        "extra_attributes": sorted(pred_attrs - gt_attrs),
        "semantic_matched_attributes": sorted(semantic_gt_hits),
        "semantic_attribute_map": semantic_map,
        "semantic_missing_attributes": sorted(gt_attrs - semantic_gt_hits),
        "semantic_extra_attributes": sorted(pred_attrs - semantic_pred_hits),
    }
    return {
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "models": _usage_models(run_root),
        "metrics": metrics,
        "files": files,
        "rows": rows,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# ReDD Analogous Schema Run Summary",
        "",
        f"Run root: `{summary['run_root']}`",
        f"Dataset root: `{summary['dataset_root']}`",
        "",
        "## Metrics",
        "",
        f"- Schema files: {metrics['schema_files']}",
        f"- Table recall: {metrics['table_recall']}",
        f"- Table precision: {metrics['table_precision']}",
        f"- Attribute recall: {metrics['attribute_recall']}",
        f"- Attribute precision: {metrics['attribute_precision']}",
        f"- Semantic attribute recall: {metrics['semantic_attribute_recall']}",
        f"- Semantic attribute precision: {metrics['semantic_attribute_precision']}",
        f"- Missing attributes: {metrics['missing_attributes']}",
        f"- Extra attributes: {metrics['extra_attributes']}",
        f"- Semantic missing attributes: {metrics['semantic_missing_attributes']}",
        f"- Semantic extra attributes: {metrics['semantic_extra_attributes']}",
        "",
    ]
    path.write_text("\n".join(lines))


def _analogous_result(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    model_label = ", ".join(summary.get("models") or ["unknown model"])
    full = (
        metrics.get("schema_files", 0) > 0
        and metrics.get("table_recall") == 1.0
        and metrics.get("table_precision") == 1.0
        and metrics.get("semantic_attribute_recall") == 1.0
        and metrics.get("semantic_attribute_precision") == 1.0
    )
    return {
        "experiment_id": "table4_schema_discovery",
        "status": "analogous_supported" if full else "partial",
        "observed": (
            f"Analogous LLM schema run ({model_label}): table_recall={metrics.get('table_recall')}, "
            f"table_precision={metrics.get('table_precision')}, "
            f"attribute_recall={metrics.get('attribute_recall')}, "
            f"attribute_precision={metrics.get('attribute_precision')}, "
            f"semantic_attribute_recall={metrics.get('semantic_attribute_recall')}, "
            f"semantic_attribute_precision={metrics.get('semantic_attribute_precision')}, "
            f"semantic_attribute_map={metrics.get('semantic_attribute_map')}; "
            f"summary={summary['run_root']}/reports/redd_paper_analogous_schema_summary.json"
        ),
        "comparison": (
            "Paper-like Table 4 smoke evidence for LLM schema discovery on the current demo dataset; "
            "not exact paper Table 4 reproduction."
        ),
        "next_action": "Scale the analogous schema discovery run and add Phase I/II/repair variants.",
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
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--paper-output-root", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    dataset_root = Path(args.dataset_root)
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = summarize_run(run_root, dataset_root)
    summary_path = reports / "redd_paper_analogous_schema_summary.json"
    summary_md_path = reports / "redd_paper_analogous_schema_summary.md"
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
