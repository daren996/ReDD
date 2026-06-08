#!/usr/bin/env python3
"""Build paper-oriented experiment summaries from available ReDD artifacts.

This script does not synthesize SCAPE evidence. When hidden states, classifier
models, or correction outputs are missing, the corresponding paper experiments
are marked blocked and the generated report explains the missing prerequisite.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PAPER_TABLE2 = {
    "spider": {"no_correction": 0.938, "scape": 0.991, "scape_hyb": 0.993},
    "bird": {"no_correction": 0.949, "scape": 0.992, "scape_hyb": 0.994},
    "galois": {"evaporate": 0.475, "palimpzest": 0.867, "no_correction": 0.873, "scape": 0.989, "scape_hyb": 0.989},
    "fda": {"evaporate": 0.516, "palimpzest": 0.924, "no_correction": 0.965, "scape": 0.988, "scape_hyb": 0.990},
    "cuad": {"evaporate": 0.209, "palimpzest": 0.613, "no_correction": 0.661, "scape": 0.724, "scape_hyb": 0.983},
}

PAPER_TABLE3 = {
    "spider": {"scape": 0.063, "scape_hyb": 0.038},
    "bird": {"scape": 0.054, "scape_hyb": 0.039},
    "galois": {"scape": 0.044, "scape_hyb": 0.027},
    "fda": {"scape": 0.051, "scape_hyb": 0.032},
    "cuad": {"scape": 0.114, "scape_hyb": 0.072},
}

PAPER_TABLE4 = {
    "phase_i_only": {"invalids": 2, "schema_recall": 0.989, "schema_precision": 0.522},
    "phase_ii_only": {"invalids": 12, "schema_recall": 0.951, "schema_precision": 0.968},
    "phase_i_ii": {"invalids": 1, "schema_recall": 0.991, "schema_precision": 0.956},
    "phase_i_ii_repair": {"invalids": 0, "schema_recall": 1.000, "schema_precision": 0.956},
}

PAPER_RUNTIME = {
    "scape_hyb_per_query_hours_upper_bound": 4.0,
    "schema_discovery_minutes": 40,
    "data_extraction_hours": 3,
    "correction_minutes": 20,
    "galois_tokens_per_query": 18_000_000,
}

PAPER_TABLE1_QUERY_COUNTS = {
    "spider": 86,
    "bird": 36,
    "galois": 10,
    "fda": 6,
    "cuad": 15,
}


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    paper_ref: str
    status: str
    paper_claim: str
    observed: str
    comparison: str
    next_action: str


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _avg(values: Iterable[float]) -> float | None:
    items = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return mean(items) if items else None


def _status_counts(results: list[ExperimentResult]) -> dict[str, int]:
    return dict(Counter(r.status for r in results))


def _env_status() -> dict[str, Any]:
    cuda = False
    torch_importable = False
    try:
        import torch  # type: ignore

        torch_importable = True
        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    api_keys = [
        "OPENAI_API_KEY",
        "SILICONFLOW_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
    ]
    return {
        "platform": platform.platform(),
        "torch_importable": torch_importable,
        "cuda_available": cuda,
        "api_keys_present": sorted(k for k in api_keys if os.environ.get(k)),
    }


def _artifact_counts(output_root: Path) -> dict[str, int]:
    return {
        "hidden_state_files": len(list(output_root.glob("**/hidden_states*/**/*.pt"))),
        "classifier_models": len(list(output_root.glob("**/classifiers/**/*.pt"))),
        "classifier_eval_json": len(list(output_root.glob("**/eval_classifiers*/**/*.json"))),
        "correction_eval_json": len(list(output_root.glob("**/eval_correction*.json"))),
        "scape_named_files": len(
            [
                p
                for p in output_root.glob("**/*")
                if p.is_file() and ("scape" in p.name.lower() or "hyb" in p.name.lower())
            ]
        ),
        "schema_original_files": len(list(output_root.glob("**/*.original.json"))),
        "schema_current_files": len([p for p in output_root.glob("**/schema_refinement/**/*.json") if ".original" not in p.name]),
        "runtime_rows": len(_read_json(output_root / "current_sweep_ranked.json", [])),
    }


def _global_artifact_counts(repo_root: Path) -> dict[str, int]:
    return {
        "hidden_state_files": len(list(repo_root.glob("**/hidden_states*/**/*.pt"))),
        "classifier_models": len(list(repo_root.glob("**/classifiers/**/*.pt"))),
        "classifier_eval_json": len(list(repo_root.glob("**/eval_classifiers*/**/*.json"))),
        "correction_eval_json": len(list(repo_root.glob("**/eval_correction*.json"))),
        "scape_or_hyb_json": len(
            [
                p
                for p in repo_root.glob("**/*.json")
                if "scape" in p.name.lower() or "hyb" in p.name.lower()
            ]
        ),
    }


def _metric_values_from_obj(obj: Any) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            if key == "threshold2correctedaccuracy":
                for item in value.values():
                    if isinstance(item, (int, float)):
                        metrics["corrected_accuracy"].append(float(item))
                return
            if key == "threshold2extracostrate":
                for item in value.values():
                    if isinstance(item, (int, float)):
                        metrics["extra_cost_rate"].append(float(item))
                return
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, (int, float)):
            if key in {
                "original_accuracy",
                "corrected_accuracy",
                "extra_cost_rate",
                "residual_error_rate",
                "error_rate",
                "coverage",
                "abstain",
                "precision",
                "recall",
                "f1",
            }:
                metrics[key].append(float(value))

    visit(obj)
    return dict(metrics)


def _summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _correction_artifact_summary(output_root: Path) -> dict[str, Any]:
    paths = sorted(set(output_root.glob("**/eval_classifiers*/**/*.json")) | set(output_root.glob("**/eval_correction*.json")))
    metrics: dict[str, list[float]] = defaultdict(list)
    files_with_metrics = 0
    parse_errors = 0
    for path in paths:
        try:
            data = _read_json(path, None)
            file_metrics = _metric_values_from_obj(data)
        except Exception:
            parse_errors += 1
            continue
        if file_metrics:
            files_with_metrics += 1
        for key, values in file_metrics.items():
            metrics[key].extend(values)
    return {
        "files": len(paths),
        "files_with_metrics": files_with_metrics,
        "parse_errors": parse_errors,
        "metrics": {key: _summarize_values(values) for key, values in sorted(metrics.items()) if values},
    }


def _load_sweep_rows(output_root: Path) -> list[dict[str, Any]]:
    data = _read_json(output_root / "current_sweep_ranked.json", [])
    return data if isinstance(data, list) else []


def _baseline_rows(sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in sweep_rows if str(row.get("artifact_id", "")).endswith("baseline")), None)
    rows = baseline.get("rows", []) if baseline else []
    return rows if isinstance(rows, list) else []


def _canonical_query_counts(dataset_root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in dataset_root.glob("*/metadata/queries.json"):
        dataset_id = path.parent.parent.name
        data = _read_json(path, {})
        query_count = len(data) if isinstance(data, (list, dict)) else 0
        counts[dataset_id.split(".")[0]] += query_count
    return dict(sorted(counts.items()))


def _dataset_setup_result(sweep_rows: list[dict[str, Any]], dataset_root: Path) -> ExperimentResult:
    rows = _baseline_rows(sweep_rows)
    canonical_counts = _canonical_query_counts(dataset_root)
    if not rows:
        return ExperimentResult(
            "table1_dataset_setup",
            "Table 1 / Sec. 6.1",
            "missing",
            f"Expected paper query counts: {PAPER_TABLE1_QUERY_COUNTS}",
            f"No baseline rows found in current_sweep_ranked.json. canonical_query_counts={canonical_counts}",
            "Cannot compare current output to paper setup.",
            "Run a sweep that emits baseline rows.",
        )
    counts = Counter(str(r.get("dataset", "")).split(".")[0] for r in rows)
    paper_families = set(PAPER_TABLE1_QUERY_COUNTS)
    observed_families = {k for k in counts if k in paper_families}
    exact = all(counts.get(k, 0) == v for k, v in PAPER_TABLE1_QUERY_COUNTS.items()) and set(counts) <= paper_families
    status = "supported" if exact else "partial"
    comparison = "exact paper setup" if exact else "current output is a broader/different benchmark than the paper setup"
    return ExperimentResult(
        "table1_dataset_setup",
        "Table 1 / Sec. 6.1",
        status,
        f"Expected paper query counts: {PAPER_TABLE1_QUERY_COUNTS}",
        f"Observed output families={dict(sorted(counts.items()))}; canonical_query_counts={canonical_counts}; paper families covered={sorted(observed_families)}; total query rows={len(rows)}",
        comparison,
        "Provide the exact paper dataset split/artifacts; current canonical query counts are not sufficient to reconstruct Table 1 verbatim.",
    )


def _fig1_pipeline_result() -> ExperimentResult:
    return ExperimentResult(
        "fig1_pipeline_overview",
        "Figure 1",
        "not_experimental",
        "Figure 1 is a conceptual overview of raw documents, ISD, TDP, and correction.",
        "No numeric output artifact is required; this is a design/architecture figure.",
        "No experiment can validate a conceptual pipeline diagram directly.",
        "Keep implementation documentation aligned with the pipeline stages.",
    )


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("dataset", "")).split(".")[0]
        grouped[family].append(row)
    summary = {}
    for family, items in grouped.items():
        summary[family] = {
            "queries": len(items),
            "answer_recall": _avg(float(item.get("answer_recall", 0)) for item in items) or 0.0,
            "cell_recall": _avg((item.get("cell", [0, 0])[0] / item.get("cell", [0, 1])[1]) for item in items if item.get("cell", [0, 0])[1]) or 0.0,
        }
    return dict(sorted(summary.items()))


def _table2_surrogate_result(output_root: Path, sweep_rows: list[dict[str, Any]], counts: dict[str, int]) -> ExperimentResult:
    summary = _correction_artifact_summary(output_root)
    if summary["files_with_metrics"] and (
        "corrected_accuracy" in summary["metrics"] or "original_accuracy" in summary["metrics"]
    ):
        status = "partial"
        observed = json.dumps(summary, sort_keys=True)
        next_action = "Map artifact file names/config metadata to No Correction, SCAPE, and SCAPE-Hyb rows per dataset."
        comparison = (
            "Correction metrics are parseable, but the suite still needs method/dataset labels to reproduce Table 2 exactly."
        )
    else:
        status = "blocked"
        baseline = next((row for row in sweep_rows if str(row.get("artifact_id", "")).endswith("baseline")), None)
        observed_summary = _family_summary(baseline.get("rows", [])) if baseline else {}
        observed = (
            "No SCAPE/SCAPE-Hyb classifier or correction outputs found. "
            f"Oracle baseline family summary={observed_summary}"
        )
        next_action = "Run real LLM extraction, hidden-state generation, classifier training, and SCAPE/SCAPE-Hyb correction evaluation."
        comparison = "Current oracle extraction recall is not Table 2 evidence because Table 2 measures real extraction errors and correction."
    return ExperimentResult(
        "table2_data_extraction_accuracy",
        "Table 2",
        status,
        f"Paper ACCpop targets: {PAPER_TABLE2}",
        observed,
        comparison,
        next_action,
    )


def _table3_fpr_result(output_root: Path, counts: dict[str, int]) -> ExperimentResult:
    summary = _correction_artifact_summary(output_root)
    if summary["files_with_metrics"] and "extra_cost_rate" in summary["metrics"]:
        status = "partial"
        observed = json.dumps(summary, sort_keys=True)
        comparison = "Extra-cost metrics are parseable, but exact FPRpop requires method/dataset labels and fp/tn counts."
    else:
        status = "blocked"
        observed = (
            f"correction_eval_json={counts['correction_eval_json']}; "
            f"classifier_eval_json={counts['classifier_eval_json']}"
        )
        comparison = "Cannot compute FPRpop without correction predictions and labels."
    return ExperimentResult(
        "table3_false_positive_overhead",
        "Table 3",
        status,
        f"Paper FPRpop targets: {PAPER_TABLE3}",
        observed,
        comparison,
        "Run SCAPE and SCAPE-Hyb correction evaluation and emit per-dataset fp/tn or FPRpop.",
    )


def _alpha_threshold_effect_result(output_root: Path, counts: dict[str, int]) -> ExperimentResult:
    summary = _correction_artifact_summary(output_root)
    claim = (
        "Paper Sec. 6.2.3: decreasing alpha increases ACCpop and increases review/FPR cost; "
        "Spider examples include alpha=0.5 -> ACCpop 0.947 with 4.9% reviewed, "
        "alpha=0.3 -> 0.975 with 9.2% reviewed, alpha=0.01 -> 0.998 with 31% reviewed."
    )
    if summary["files_with_metrics"] and {"coverage", "extra_cost_rate"} & set(summary["metrics"]):
        return ExperimentResult(
            "sec623_alpha_threshold_effect",
            "Sec. 6.2.3 / Figure 2",
            "partial",
            claim,
            json.dumps(summary, sort_keys=True),
            "Alpha-related metrics are parseable, but exact alpha-point labels and Spider review counts require sweep metadata.",
            "Emit alpha, dataset, reviewed_count, false_positive_count, ACCpop, and FPRpop for each SCAPE-Hyb sweep point.",
        )
    return ExperimentResult(
        "sec623_alpha_threshold_effect",
        "Sec. 6.2.3 / Figure 2",
        "blocked",
        claim,
        (
            f"classifier_eval_json={counts['classifier_eval_json']}; "
            f"correction_eval_json={counts['correction_eval_json']}"
        ),
        "No alpha sweep correction artifacts are present.",
        "Run SCAPE-Hyb alpha sweeps with per-alpha ACCpop/FPRpop/review-count accounting.",
    )


def _cuad_chunk_merge_result(counts: dict[str, int]) -> ExperimentResult:
    return ExperimentResult(
        "table2_cuad_chunk_merge",
        "Table 2 / Sec. 6.2.1",
        "blocked",
        "Paper claim: CUAD needs map-reduce chunk merge; SCAPE-Hyb reaches ACCpop=0.983 versus SCAPE=0.724.",
        f"correction_eval_json={counts['correction_eval_json']}; scape_named_files={counts['scape_named_files']}",
        "Current output has no CUAD map-reduce chunk-merge correction comparison.",
        "Run CUAD long-document chunking, chunk merge, and SCAPE/SCAPE-Hyb correction evaluation.",
    )


def _fig2_to_fig6_results(output_root: Path, counts: dict[str, int], env: dict[str, Any]) -> list[ExperimentResult]:
    artifact_line = (
        f"hidden_state_files={counts['hidden_state_files']}, classifier_models={counts['classifier_models']}, "
        f"classifier_eval_json={counts['classifier_eval_json']}, cuda={env['cuda_available']}"
    )
    correction_summary = _correction_artifact_summary(output_root)
    specs = [
        (
            "fig2_accuracy_cost_tradeoff",
            "Figure 2",
            "SCAPE-Hyb should dominate MV, CF, IndivConformal, and SCAPE across alpha/cost tradeoffs.",
            "Run alpha sweeps for MV, CF, IndivConformal, SCAPE, and SCAPE-Hyb.",
        ),
        (
            "fig3_lambda_sweep",
            "Figure 3",
            "SCAPE-Hyb accuracy should peak around the reported lambda values under calibration-size settings.",
            "Run lambda sweeps at calibration sizes 30, 150, and 300.",
        ),
        (
            "fig4_calibration_size_sweep",
            "Figure 4",
            "SCAPE/SCAPE-Hyb should improve and plateau as calibration size grows.",
            "Run Ncal-base sweeps and summarize ACCpop/FPRpop.",
        ),
        (
            "fig5_training_size_sweep",
            "Figure 5",
            "Classifier accuracy should exceed 0.99 with about 50 training examples.",
            "Run classifier training-size sweeps.",
        ),
        (
            "fig6_human_vs_llm_labels",
            "Figure 6",
            "LLM committee labels should be comparable to human labels within about 1%.",
            "Run paired human-label and LLM-committee-label correction evaluation.",
        ),
    ]
    results = []
    for experiment_id, paper_ref, claim, action in specs:
        if correction_summary["files_with_metrics"]:
            status = "partial"
            observed = json.dumps(correction_summary, sort_keys=True)
            comparison = "Classifier/correction metrics are parseable, but figure-specific sweep metadata is not complete."
        else:
            status = "blocked"
            observed = artifact_line
            comparison = "Current output has no classifier/correction artifacts for this figure."
        results.append(ExperimentResult(experiment_id, paper_ref, status, claim, observed, comparison, action))
    return results



def _schema_items(data: Any) -> set[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    if isinstance(data, dict):
        iterator = data.items()
        for table, attrs in iterator:
            if isinstance(attrs, list):
                for attr in attrs:
                    if isinstance(attr, dict):
                        name = attr.get("Attribute Name") or attr.get("name") or attr.get("attribute")
                    else:
                        name = str(attr)
                    if name:
                        items.add((str(table).strip().lower(), str(name).strip().lower()))
    elif isinstance(data, list):
        for table_obj in data:
            if not isinstance(table_obj, dict):
                continue
            table = table_obj.get("Schema Name") or table_obj.get("schema") or table_obj.get("table")
            if not table:
                continue
            for attr in table_obj.get("Attributes", []) or []:
                if isinstance(attr, dict):
                    name = attr.get("Attribute Name") or attr.get("name") or attr.get("attribute")
                else:
                    name = str(attr)
                if name:
                    items.add((str(table).strip().lower(), str(name).strip().lower()))
    return items


def _schema_surrogate(output_root: Path) -> dict[str, Any]:
    recalls = []
    precisions = []
    invalids = 0
    pairs = 0
    for original_path in output_root.glob("**/*.original.json"):
        current_path = original_path.with_name(original_path.name.replace(".original.json", ".json"))
        if not current_path.exists():
            invalids += 1
            continue
        try:
            original = _schema_items(_read_json(original_path, None))
            current = _schema_items(_read_json(current_path, None))
        except Exception:
            invalids += 1
            continue
        if not original or not current:
            invalids += 1
            continue
        pairs += 1
        inter = original & current
        recalls.append(len(inter) / len(original))
        precisions.append(len(inter) / len(current))
    return {
        "paired_schema_files": pairs,
        "invalid_or_unpaired": invalids,
        "schema_recall": _avg(recalls),
        "schema_precision": _avg(precisions),
    }


def _table4_schema_result(output_root: Path) -> ExperimentResult:
    observed = _schema_surrogate(output_root)
    if observed["paired_schema_files"] == 0:
        status = "blocked"
        comparison = "No comparable schema artifact pairs were found."
    else:
        status = "surrogate_only"
        comparison = (
            "These are oracle/adaptive schema artifact pair metrics, not the paper's GPT schema discovery "
            "Phase I/Phase II/Repair experiment."
        )
    return ExperimentResult(
        "table4_schema_discovery",
        "Table 4",
        status,
        f"Paper schema targets: {PAPER_TABLE4}",
        json.dumps(observed, sort_keys=True),
        comparison,
        "Run Phase I Only, Phase II Only, Phase I+II, and Phase I+II+Repair with true LLM schema discovery artifacts.",
    )


def _read_manifest(dataset_dir: Path) -> dict[str, str]:
    manifest = dataset_dir / "manifest.yaml"
    paths: dict[str, str] = {}
    if not manifest.exists():
        return paths
    current_section = None
    for line in manifest.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            continue
        if current_section == "paths" and ":" in line:
            key, value = line.split(":", 1)
            paths[key.strip()] = value.strip().strip("'\"")
    return paths


def _dataset_density(dataset_root: Path, dataset_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        return {"__error__": {"error": f"pandas/parquet support unavailable: {exc}"}}

    results = {}
    for dataset_id in sorted(set(dataset_ids)):
        dataset_dir = dataset_root / dataset_id
        paths = _read_manifest(dataset_dir)
        docs_path = dataset_dir / paths.get("documents", "data/documents.parquet")
        gt_path = dataset_dir / paths.get("ground_truth", "data/ground_truth.parquet")
        if not docs_path.exists() or not gt_path.exists():
            continue
        try:
            docs = pd.read_parquet(docs_path)
            gt = pd.read_parquet(gt_path)
        except Exception as exc:
            results[dataset_id] = {"error": str(exc)}
            continue
        doc_count = int(len(docs))
        cell_count = int(len(gt))
        token_count = int(docs.get("doc_text", "").fillna("").map(lambda value: len(str(value).split())).sum())
        chunked_docs = int(docs.get("is_chunked", False).fillna(False).astype(bool).sum()) if "is_chunked" in docs else 0
        multi_table_docs = 0
        if "doc_id" in gt and "table_id" in gt:
            table_counts = gt.groupby("doc_id")["table_id"].nunique()
            multi_table_docs = int((table_counts > 1).sum())
        results[dataset_id] = {
            "documents": doc_count,
            "ground_truth_cells": cell_count,
            "tokens": token_count,
            "cells_per_doc": cell_count / doc_count if doc_count else None,
            "cells_per_1k_tokens": (cell_count / token_count * 1000) if token_count else None,
            "chunked_docs": chunked_docs,
            "multi_table_docs": multi_table_docs,
        }
    return results


def _density_result(output_root: Path, dataset_root: Path, sweep_rows: list[dict[str, Any]]) -> ExperimentResult:
    rows = _baseline_rows(sweep_rows)
    dataset_ids = [str(row.get("dataset")) for row in rows if row.get("dataset")]
    density = _dataset_density(dataset_root, dataset_ids)
    if "__error__" in density:
        return ExperimentResult(
            "density_sweep",
            "Figure 8 / Sec. 6.4.2",
            "blocked",
            "Paper claim: DocDensity and DatasetDensity have minimal impact on ReDD accuracy.",
            density["__error__"]["error"],
            "Cannot compute local density summary.",
            "Install parquet dependencies or provide density sweep artifacts.",
        )
    if not density:
        return ExperimentResult(
            "density_sweep",
            "Figure 8 / Sec. 6.4.2",
            "missing",
            "Paper claim: DocDensity and DatasetDensity have minimal impact on ReDD accuracy.",
            "No canonical dataset density inputs found.",
            "No density comparison can be made.",
            "Provide canonical dataset manifests or generated density variants.",
        )
    cell_density_values = [v["cells_per_doc"] for v in density.values() if isinstance(v, dict) and v.get("cells_per_doc") is not None]
    token_density_values = [v["cells_per_1k_tokens"] for v in density.values() if isinstance(v, dict) and v.get("cells_per_1k_tokens") is not None]
    observed = {
        "datasets_measured": len(density),
        "avg_cells_per_doc": _avg(cell_density_values),
        "min_cells_per_doc": min(cell_density_values) if cell_density_values else None,
        "max_cells_per_doc": max(cell_density_values) if cell_density_values else None,
        "avg_cells_per_1k_tokens": _avg(token_density_values),
    }
    return ExperimentResult(
        "density_sweep",
        "Figure 8 / Sec. 6.4.2",
        "surrogate_only",
        "Paper claim: DocDensity and DatasetDensity have minimal impact on ReDD accuracy.",
        json.dumps(observed, sort_keys=True),
        "This measures current dataset density, but does not run controlled density variants or real ReDD accuracy under those variants.",
        "Add DocDensity/DatasetDensity variant generation and rerun real extraction/correction metrics per bin.",
    )


def _one_to_many_result(output_root: Path, dataset_root: Path, sweep_rows: list[dict[str, Any]]) -> ExperimentResult:
    rows = _baseline_rows(sweep_rows)
    density = _dataset_density(dataset_root, [str(row.get("dataset")) for row in rows if row.get("dataset")])
    total_multi = sum(int(v.get("multi_table_docs", 0)) for v in density.values() if isinstance(v, dict))
    total_chunked = sum(int(v.get("chunked_docs", 0)) for v in density.values() if isinstance(v, dict))
    status = "surrogate_only" if total_multi or total_chunked else "missing"
    return ExperimentResult(
        "one_to_many_chunk_to_table",
        "Figure 7 / Sec. 6.4.1",
        status,
        "Paper claim: one-to-many chunk-to-table keeps schema discovery unchanged and drops data extraction accuracy by only 0.13%.",
        f"current canonical inputs: multi_table_docs={total_multi}, chunked_docs={total_chunked}, output_root={output_root}",
        "Current output does not contain a controlled one-to-many variant comparison.",
        "Create/run a one-to-many chunk dataset preset and compare schema/data-extraction metrics against the default preset.",
    )


def _runtime_token_result(sweep_rows: list[dict[str, Any]]) -> ExperimentResult:
    if not sweep_rows:
        return ExperimentResult(
            "runtime_token_accounting",
            "Sec. 6.4.3",
            "missing",
            f"Paper runtime targets: {PAPER_RUNTIME}",
            "No sweep rows found.",
            "Cannot compute runtime surrogate.",
            "Run paper pipeline with phase timing and token accounting.",
        )
    elapsed = [float(row.get("elapsed_sec", 0)) for row in sweep_rows]
    llm_docs = [int(row.get("llm_docs", 0)) for row in sweep_rows]
    observed = {
        "variants": len(sweep_rows),
        "total_elapsed_sec": sum(elapsed),
        "max_variant_elapsed_sec": max(elapsed),
        "min_llm_docs": min(llm_docs),
        "max_llm_docs": max(llm_docs),
    }
    return ExperimentResult(
        "runtime_token_accounting",
        "Sec. 6.4.3",
        "surrogate_only",
        f"Paper runtime targets: {PAPER_RUNTIME}",
        json.dumps(observed, sort_keys=True),
        "Current output has optimizer elapsed time and document-call counts only; it does not have phase timings or token accounting.",
        "Instrument schema discovery, extraction, hidden-state generation, correction, GPU resources, and provider token counts.",
    )


def _load_analogous_results(output_root: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(output_root / "reports" / "redd_paper_analogous_results.json", {})
    if not isinstance(payload, dict):
        return {}
    results = payload.get("results", [])
    if not isinstance(results, list):
        return {}
    mapped = {}
    for item in results:
        if isinstance(item, dict) and item.get("experiment_id"):
            mapped[str(item["experiment_id"])] = item
    return mapped


def _apply_analogous_results(output_root: Path, results: list[ExperimentResult]) -> list[ExperimentResult]:
    analogous = _load_analogous_results(output_root)
    if not analogous:
        return results
    updated = []
    for result in results:
        override = analogous.get(result.experiment_id)
        if not override:
            updated.append(result)
            continue
        status = str(override.get("status") or "analogous_supported")
        if status not in {"analogous_supported", "partial", "blocked", "missing", "unsupported"}:
            status = "partial"
        updated.append(
            ExperimentResult(
                result.experiment_id,
                result.paper_ref,
                status,
                str(override.get("paper_claim") or result.paper_claim),
                str(override.get("observed") or result.observed),
                str(
                    override.get("comparison")
                    or "Analogous paper-like experiment evidence supplied; not exact paper evidence."
                ),
                str(override.get("next_action") or result.next_action),
            )
        )
    return updated


def _render_markdown(
    output_root: Path,
    dataset_root: Path,
    env: dict[str, Any],
    counts: dict[str, int],
    global_counts: dict[str, int],
    results: list[ExperimentResult],
) -> str:
    lines = [
        "# ReDD Paper Experiment Suite",
        "",
        f"Output root: `{output_root}`",
        f"Dataset root: `{dataset_root}`",
        "",
        "## Environment",
        "",
        f"- Platform: `{env['platform']}`",
        f"- Torch importable: `{env['torch_importable']}`",
        f"- CUDA available: `{env['cuda_available']}`",
        f"- API keys present: `{env['api_keys_present']}`",
        "",
        "## Artifact Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Global Artifact Counts", ""])
    for key, value in sorted(global_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Status Summary", ""])
    for status, count in sorted(_status_counts(results).items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Paper Claim Comparison",
            "",
            "| Experiment | Paper ref | Status | Observed | Comparison | Next action |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for result in results:
        observed = result.observed.replace("|", "\\|")
        comparison = result.comparison.replace("|", "\\|")
        next_action = result.next_action.replace("|", "\\|")
        lines.append(
            f"| `{result.experiment_id}` | {result.paper_ref} | {result.status} | {observed} | {comparison} | {next_action} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_suite(output_root: Path, dataset_root: Path) -> dict[str, Any]:
    sweep_rows = _load_sweep_rows(output_root)
    counts = _artifact_counts(output_root)
    global_counts = _global_artifact_counts(Path("."))
    env = _env_status()
    results = [
        _fig1_pipeline_result(),
        _dataset_setup_result(sweep_rows, dataset_root),
        _table2_surrogate_result(output_root, sweep_rows, counts),
        _cuad_chunk_merge_result(counts),
        _table3_fpr_result(output_root, counts),
        _alpha_threshold_effect_result(output_root, counts),
        *_fig2_to_fig6_results(output_root, counts, env),
        _table4_schema_result(output_root),
        _density_result(output_root, dataset_root, sweep_rows),
        _one_to_many_result(output_root, dataset_root, sweep_rows),
        _runtime_token_result(sweep_rows),
    ]
    results = _apply_analogous_results(output_root, results)
    return {
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "env": env,
        "artifact_counts": counts,
        "global_artifact_counts": global_counts,
        "status_summary": _status_counts(results),
        "paper_targets": {
            "table1_query_counts": PAPER_TABLE1_QUERY_COUNTS,
            "table2_accpop": PAPER_TABLE2,
            "table3_fprpop": PAPER_TABLE3,
            "table4_schema": PAPER_TABLE4,
            "runtime": PAPER_RUNTIME,
        },
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/paper_claim_run_hash_train100_v2")
    parser.add_argument("--dataset-root", default="dataset/canonical")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    dataset_root = Path(args.dataset_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = run_suite(output_root, dataset_root)
    json_path = reports_dir / "redd_paper_experiment_suite.json"
    md_path = reports_dir / "redd_paper_experiment_suite.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(
        _render_markdown(
            output_root,
            dataset_root,
            payload["env"],
            payload["artifact_counts"],
            payload["global_artifact_counts"],
            [ExperimentResult(**item) for item in payload["results"]],
        )
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Status summary: {payload['status_summary']}")
    if any(result["status"] in {"blocked", "missing"} for result in payload["results"]):
        return 2
    if any(result["status"] in {"partial", "surrogate_only", "implemented_needs_aggregation"} for result in payload["results"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
