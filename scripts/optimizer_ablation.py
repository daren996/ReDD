from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from redd.diagnostics.dataset_consistency import (
    audit_dataset_consistency,
    build_dataset_consistency_audit,
    write_dataset_consistency_audit,
)
from redd.runners import run_evaluation, run_extract

CURRENT_DATASETS: dict[str, dict[str, Any]] = {
    "bird.schools_demo": {
        "loader": "hf_manifest",
        "root": "dataset/derived/bird.schools_demo",
        "query_ids": ["q1", "q2", "q3", "q4", "q5"],
        "loader_options": {"manifest": "manifest.yaml"},
        "split": {"train_count": 0},
    },
    "spider.college_demo": {
        "loader": "hf_manifest",
        "root": "dataset/derived/spider.college_demo",
        "query_ids": ["q1", "q2", "q3"],
        "loader_options": {"manifest": "manifest.yaml"},
        "split": {"train_count": 0},
    },
}
HELDOUT_DATASETS: dict[str, dict[str, Any]] = {
    "bird.student_club.default_task": {
        "loader": "hf_manifest",
        "root": "dataset/derived/bird.student_club.default_task",
        "query_ids": ["Q1", "Q2", "Q3"],
        "loader_options": {
            "manifest": "manifest.yaml",
            "filemap": {"queries": "metadata/query_sets/generated_queries.json"},
        },
        "split": {"train_count": 0},
    },
    "spider.soccer_1.default_task": {
        "loader": "hf_manifest",
        "root": "dataset/derived/spider.soccer_1.default_task",
        "query_ids": ["Q1", "Q2", "Q3"],
        "loader_options": {
            "manifest": "manifest.yaml",
            "filemap": {"queries": "metadata/query_sets/generated_queries.json"},
        },
        "split": {"train_count": 0},
    },
}
DATASET_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "current": CURRENT_DATASETS,
    "heldout": HELDOUT_DATASETS,
}
DATASETS = CURRENT_DATASETS
EXPECTED_QUERY_COUNT = sum(len(dataset["query_ids"]) for dataset in DATASETS.values())
DEFAULT_BASELINE_LLM_DOCS = 295
DEFAULT_ORACLE_UPPER_BOUND_LLM_DOCS = 37

PASS_THROUGH_MODES: dict[str, list[str]] = {
    "strict": [],
    "numge": ["num_ge1500"],
    "numtst": ["num_tst_takr"],
    "year": ["year"],
    "numge-year": ["num_ge1500", "year"],
    "numtst-year": ["num_tst_takr", "year"],
    "bird-counts": ["num_ge1500", "num_tst_takr"],
    "bird-counts-year": ["num_ge1500", "num_tst_takr", "year"],
    "score-safe": ["avg_scr_math", "avg_scr_read", "avg_scr_write"],
    "all-bird": [
        "avg_scr_math",
        "avg_scr_read",
        "avg_scr_write",
        "num_ge1500",
        "num_tst_takr",
    ],
}
HELDOUT_TEXT_MISMATCH_PASS_THROUGH = [
    "event_name",
    "height",
    "weight",
    "potential",
    "aggression",
    "overall_rating",
]


def _slug_float(value: float) -> str:
    return str(value).replace(".", "p")


def _expected_query_count(datasets: dict[str, dict[str, Any]]) -> int:
    return sum(len(dataset.get("query_ids") or ["default"]) for dataset in datasets.values())


def _base_config(
    output_dir: str,
    artifact_id: str,
    datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "config_version": "2.1.1",
        "project": {"name": "optimizer-current-sweep", "seed": 42},
        "runtime": {
            "output_dir": output_dir,
            "log_dir": "logs",
            "output_layout": "dataset_stage",
            "artifact_id": artifact_id,
            "console_log_level": "WARNING",
        },
        "models": {
            "llm": {"provider": "none", "model": "ground_truth"},
            "embedding": {
                "provider": "local",
                "model": "local-hash-embedding",
                "enabled": True,
                "api_key_env": None,
                "storage_file": "embeddings.sqlite3",
            },
        },
        "datasets": copy.deepcopy(datasets),
        "stages": {
            "data_extraction": {
                "enabled": True,
                "schema_source": "ground_truth",
                "oracle": "ground_truth",
                "options": {"force_rerun": True},
            }
        },
        "experiments": {
            "demo": {
                "datasets": list(datasets),
                "stages": ["data_extraction"],
                "artifact_id": artifact_id,
            }
        },
    }


def _make_variant_config(
    *,
    output_dir: str,
    artifact_id: str,
    doc_threshold: float | None,
    proxy_threshold: float | None,
    pass_through: list[str] | None,
    datasets: dict[str, dict[str, Any]],
    use_doc_filter: bool,
    use_proxy: bool,
    use_join_resolution: bool = False,
    bidirectional_join_resolution: bool = False,
    join_order_strategy: str = "sql",
    join_empty_short_circuit: bool = False,
    use_oracle_predicate_proxy: bool = False,
    use_gt_text_consistency_guard: bool = False,
    cross_query_extraction_cache: bool = False,
    cache_extract_full_table: bool = False,
    table_assignment_cache: bool = False,
    table_assignment_cache_general_schema: bool = False,
    table_assignment_cache_source_table_metadata: bool = False,
    predicate_proxy_mode: str = "pretrained",
    use_finetuned_learned_proxies: bool = True,
    finetuned_model: str = "heuristic",
    target_recall: float = 0.95,
    heuristic_pass_through_doc_ids_by_attribute: dict[str, list[str]] | None = None,
    heuristic_force_reject_doc_ids_by_attribute: dict[str, list[str]] | None = None,
    heuristic_force_reject_doc_ids_by_predicate: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    config = _base_config(output_dir, artifact_id, datasets)
    stage = config["stages"]["data_extraction"]
    if table_assignment_cache:
        stage["table_assignment_cache"] = {
            "enabled": True,
            "general_schema": bool(table_assignment_cache_general_schema),
            "source_table_metadata": bool(table_assignment_cache_source_table_metadata),
        }
    if use_doc_filter:
        stage["document_filtering"] = {
            "enabled": True,
            "filter_type": "schema_relevance",
            "target_recall": 0.95,
            "enable_calibrate": False,
            "embedding_model": "local-hash-embedding",
            "embeddings_cache_dir": str(Path(output_dir) / "_embedding_cache"),
            "threshold": float(doc_threshold if doc_threshold is not None else 0.58),
        }
    if use_proxy:
        stage["proxy_runtime"] = {
            "enabled": True,
            "predicate_proxy_mode": str(predicate_proxy_mode),
            "target_recall": float(target_recall),
            "use_embedding_proxies": False,
            "use_learned_proxies": not bool(use_oracle_predicate_proxy),
            "use_finetuned_learned_proxies": bool(use_finetuned_learned_proxies),
            "use_join_resolution": bool(use_join_resolution),
            "bidirectional_join_resolution": bool(bidirectional_join_resolution),
            "join_order_strategy": str(join_order_strategy),
            "join_empty_short_circuit": bool(join_empty_short_circuit),
            "use_oracle_predicate_proxy": bool(use_oracle_predicate_proxy),
            "use_gt_text_consistency_guard": bool(use_gt_text_consistency_guard),
            "cross_query_extraction_cache": bool(cross_query_extraction_cache),
            "cache_extract_full_table": bool(cache_extract_full_table),
            "embedding_model": "local-hash-embedding",
            "embeddings_cache_dir": str(Path(output_dir) / "_embedding_cache"),
            "finetuned_model": str(finetuned_model),
            "finetuned_epochs": 0,
            "proxy_threshold": float(proxy_threshold if proxy_threshold is not None else 0.5),
            "heuristic_pass_through_attributes": list(pass_through or []),
            "heuristic_pass_through_doc_ids_by_attribute": (
                heuristic_pass_through_doc_ids_by_attribute or {}
            ),
            "heuristic_force_reject_doc_ids_by_attribute": (
                heuristic_force_reject_doc_ids_by_attribute or {}
            ),
            "heuristic_force_reject_doc_ids_by_predicate": (
                heuristic_force_reject_doc_ids_by_predicate or {}
            ),
            "allow_embedding_fallback": False,
            "save_hard_negatives": False,
            "verbose": False,
        }
        stage["alpha_allocation"] = {"enabled": False}
    return config


def _write_config(config: dict[str, Any], config_dir: Path, artifact_id: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{artifact_id}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _is_assigned_table(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "none", "null", "nan"}


def _is_offline_upper_record(record: dict[str, Any]) -> bool:
    return bool(
        record.get("use_oracle_predicate_proxy", False)
        or record.get("use_gt_text_consistency_guard", False)
        or record.get("use_audit_conflict_quarantine", False)
        or record.get("use_audit_conflict_reject_guard", False)
    )


def _is_deployable_record(record: dict[str, Any]) -> bool:
    return not _is_offline_upper_record(record)


def _best_offline_upper_record(
    ranked: list[dict[str, Any]],
) -> dict[str, Any] | None:
    full_recall = [
        record
        for record in ranked
        if _is_offline_upper_record(record) and record.get("full_recall", False)
    ]
    return min(full_recall, key=lambda record: int(record["llm_docs"])) if full_recall else None


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 1.0


def _rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    table_cache = record.get("table_assignment_cache") or {}
    table_assignment_saved = int(table_cache.get("cache_hits") or 0) + int(
        table_cache.get("source_table_metadata_hits") or 0
    )
    return (
        record["answer_recall"],
        record["cell_recall"],
        record["table_recall"],
        record["can_answer"][0],
        record["saved_rate"],
        table_cache.get("saved_rate", 0.0),
        table_assignment_saved,
        table_cache.get("source_table_metadata_hits", 0),
        -record["llm_docs"],
    )


def _audit_dataset_consistency(dataset_id: str, dataset_config: dict[str, Any]) -> dict[str, Any]:
    loader_options = dataset_config.get("loader_options") or {}
    filemap = loader_options.get("filemap") or {}
    return audit_dataset_consistency(
        dataset_id,
        dataset_config["root"],
        query_ids=dataset_config.get("query_ids"),
        queries_path=filemap.get("queries"),
    )


def _build_dataset_consistency_audit(
    output_dir: Path,
    datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    audit = build_dataset_consistency_audit(datasets)
    write_dataset_consistency_audit(output_dir, audit)
    return audit


def _audit_conflict_doc_ids_by_attribute(
    datasets: dict[str, dict[str, Any]],
    attributes: list[str],
    *,
    conflict_type: str,
) -> dict[str, list[str]]:
    """Return doc ids for a given text/GT conflict type by attribute."""
    wanted = {str(attr).lower() for attr in attributes}
    audit = build_dataset_consistency_audit(datasets)
    by_attribute: dict[str, set[str]] = {attr: set() for attr in wanted}
    for dataset in audit.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        for conflict in dataset.get("conflicts") or []:
            if not isinstance(conflict, dict):
                continue
            attr = str(conflict.get("attribute") or "").lower()
            if attr not in wanted:
                continue
            if conflict.get("conflict_type") != conflict_type:
                continue
            doc_id = str(conflict.get("doc_id") or "")
            if doc_id:
                by_attribute[attr].add(doc_id)
    return {
        attr: sorted(doc_ids)
        for attr, doc_ids in by_attribute.items()
        if doc_ids
    }


def _audit_predicate_guard_key(conflict: dict[str, Any]) -> str:
    expected = conflict.get("expected")
    if isinstance(expected, float) and expected.is_integer():
        expected_text = str(int(expected))
    else:
        expected_text = str(expected).strip().lower()
    return (
        f"{str(conflict.get('attribute') or '').lower()}::"
        f"{str(conflict.get('operator') or '').strip().lower()}::{expected_text}"
    )


def _audit_conflict_doc_ids_by_predicate(
    datasets: dict[str, dict[str, Any]],
    attributes: list[str],
    *,
    conflict_type: str,
) -> dict[str, list[str]]:
    """Return doc ids for a text/GT conflict type by exact predicate."""
    wanted = {str(attr).lower() for attr in attributes}
    audit = build_dataset_consistency_audit(datasets)
    identity_mismatches = _identity_mismatch_doc_ids_by_attribute(datasets, wanted)
    by_predicate: dict[str, set[str]] = {}
    for dataset in audit.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        for conflict in dataset.get("conflicts") or []:
            if not isinstance(conflict, dict):
                continue
            attr = str(conflict.get("attribute") or "").lower()
            if attr not in wanted:
                continue
            if conflict.get("conflict_type") != conflict_type:
                continue
            doc_id = str(conflict.get("doc_id") or "")
            if not doc_id:
                continue
            if doc_id in identity_mismatches.get(attr, set()):
                continue
            by_predicate.setdefault(_audit_predicate_guard_key(conflict), set()).add(
                doc_id
            )
    return {
        predicate_key: sorted(doc_ids)
        for predicate_key, doc_ids in by_predicate.items()
        if doc_ids
    }


def _identity_mismatch_doc_ids_by_attribute(
    datasets: dict[str, dict[str, Any]],
    attributes: set[str],
) -> dict[str, set[str]]:
    """Return docs whose player_attributes text appears to describe another player."""
    import pandas as pd

    by_attribute: dict[str, set[str]] = {attr: set() for attr in attributes}
    for dataset_config in datasets.values():
        root = Path(dataset_config["root"])
        documents = pd.read_parquet(root / "data" / "documents.parquet")
        ground_truth = pd.read_parquet(root / "data" / "ground_truth.parquet")
        player_names = {
            str(row.get("doc_id")): str(row.get("value") or "")
            for row in ground_truth.to_dict("records")
            if str(row.get("column_name") or "").lower() == "player_name"
        }
        for row in documents.to_dict("records"):
            if str(row.get("source_table") or "").lower() != "player_attributes":
                continue
            doc_id = str(row.get("doc_id") or "")
            player_name = player_names.get(doc_id, "")
            if not player_name:
                continue
            if _audit_norm_text(player_name) not in _audit_norm_text(row.get("doc_text")):
                for attr in attributes:
                    by_attribute[attr].add(doc_id)
    return by_attribute


def _audit_norm_text(value: Any) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _audit_quarantine_doc_ids_by_attribute(
    datasets: dict[str, dict[str, Any]],
    attributes: list[str],
) -> dict[str, list[str]]:
    """Return doc ids that GT says should pass but text proxies would reject."""
    return _audit_conflict_doc_ids_by_attribute(
        datasets,
        attributes,
        conflict_type="gt_pass_text_fail",
    )


def _summarize_artifact(
    output_dir: Path,
    artifact_id: str,
    expected_query_count: int | None = None,
) -> dict[str, Any]:
    expected_query_count = expected_query_count or EXPECTED_QUERY_COUNT
    artifact_roots = list(output_dir.glob(f"*/data_extraction/{artifact_id}"))
    assigned_docs = 0
    proxy_all_docs = 0
    proxy_passed_docs = 0
    doc_filter_excluded = 0
    doc_filter_input = 0
    table_assignment_cache_input = 0
    table_assignment_cache_hits = 0
    table_assignment_cache_misses = 0
    table_assignment_source_table_metadata_hits = 0
    table_assignment_source_table_metadata_misses = 0
    table_assignment_cache_excluded = 0
    answer_covered = answer_total = 0
    cell_covered = cell_total = 0
    table_covered = table_total = 0
    can_answer = 0
    query_count = 0
    bad: list[str] = []
    rows: list[dict[str, Any]] = []

    for root in artifact_roots:
        dataset = root.parents[1].name
        for res_path in root.glob("res_tabular_data_*.json"):
            if res_path.name.endswith("_proxy_decisions.json"):
                continue
            data = json.loads(res_path.read_text(encoding="utf-8"))
            assigned_docs += sum(1 for entry in data.values() if _is_assigned_table(entry.get("res")))

        for doc_filter_path in root.glob("doc_filter/doc_filter_*.json"):
            data = json.loads(doc_filter_path.read_text(encoding="utf-8"))
            excluded = data.get("excluded_doc_ids") or []
            kept = data.get("kept_doc_ids") or []
            doc_filter_excluded += len(excluded)
            doc_filter_input += len(excluded) + len(kept)

        table_cache_path = root / "table_assignment_cache.json"
        if table_cache_path.exists():
            data = json.loads(table_cache_path.read_text(encoding="utf-8"))
            events = data.get("events") or []
            if isinstance(events, list):
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    table_assignment_cache_input += int(event.get("input_docs") or 0)
                    table_assignment_cache_hits += int(event.get("cache_hits") or 0)
                    table_assignment_cache_misses += int(event.get("cache_misses") or 0)
                    table_assignment_source_table_metadata_hits += int(
                        event.get("source_table_metadata_hits") or 0
                    )
                    table_assignment_source_table_metadata_misses += int(
                        event.get("source_table_metadata_misses") or 0
                    )
                    table_assignment_cache_excluded += int(event.get("excluded") or 0)

        for proxy_path in root.glob("*_proxy_decisions.json"):
            data = json.loads(proxy_path.read_text(encoding="utf-8"))
            for decision in data.values():
                all_ids = decision.get("all_doc_ids") or []
                passed_ids = decision.get("extracted_doc_ids")
                if not isinstance(passed_ids, list):
                    passed_ids = decision.get("passed_doc_ids") or []
                proxy_all_docs += len(all_ids)
                proxy_passed_docs += len(passed_ids)

        for eval_path in root.glob("eval_*.json"):
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            qa = data.get("query_aware") or {}
            query = qa.get("query_id") or eval_path.stem.split("_")[1]
            summary = qa.get("summary") or {}
            table = qa.get("table_assignment") or {}
            cell = qa.get("cell_recall") or {}
            answer = qa.get("answer_recall") or {}
            t_cov, t_tot = int(table.get("covered") or 0), int(table.get("total") or 0)
            c_cov, c_tot = int(cell.get("covered") or 0), int(cell.get("total") or 0)
            a_cov, a_tot = int(answer.get("covered") or 0), int(answer.get("total") or 0)
            ok = bool(summary.get("can_answer_query"))
            query_count += 1
            can_answer += int(ok)
            table_covered += t_cov
            table_total += t_tot
            cell_covered += c_cov
            cell_total += c_tot
            answer_covered += a_cov
            answer_total += a_tot
            rows.append(
                {
                    "dataset": dataset,
                    "query": query,
                    "can_answer": ok,
                    "table": [t_cov, t_tot],
                    "cell": [c_cov, c_tot],
                    "answer": [a_cov, a_tot],
                    "answer_recall": _safe_div(a_cov, a_tot),
                }
            )
            if not ok:
                bad.append(
                    f"{dataset}:{query} A={a_cov}/{a_tot} C={c_cov}/{c_tot} T={t_cov}/{t_tot}"
                )

    llm_docs = proxy_passed_docs if proxy_all_docs else assigned_docs
    missing_query_count = max(expected_query_count - query_count, 0)
    if missing_query_count:
        bad.extend([f"missing-eval:{i + 1}" for i in range(missing_query_count)])

    table_assignment_calls_before = max(
        table_assignment_cache_input - table_assignment_cache_excluded,
        0,
    )
    return {
        "llm_docs": llm_docs,
        "assigned_docs": assigned_docs,
        "answer_recall": _safe_div(answer_covered, answer_total) if query_count else 0.0,
        "cell_recall": _safe_div(cell_covered, cell_total) if query_count else 0.0,
        "table_recall": _safe_div(table_covered, table_total) if query_count else 0.0,
        "can_answer": [can_answer, expected_query_count],
        "doc_filter": {
            "total": doc_filter_input,
            "excluded": doc_filter_excluded,
            "kept": doc_filter_input - doc_filter_excluded,
        },
        "table_assignment_calls_before": table_assignment_calls_before,
        "table_assignment_cache": {
            "input_docs": table_assignment_cache_input,
            "cache_hits": table_assignment_cache_hits,
            "cache_misses": table_assignment_cache_misses,
            "source_table_metadata_hits": table_assignment_source_table_metadata_hits,
            "source_table_metadata_misses": table_assignment_source_table_metadata_misses,
            "excluded": table_assignment_cache_excluded,
            "saved_rate": _safe_div(
                table_assignment_cache_hits + table_assignment_source_table_metadata_hits,
                table_assignment_calls_before,
            )
            if table_assignment_calls_before
            else 0.0,
        },
        "proxy": {
            "evaluated_doc_calls": proxy_all_docs,
            "passed_doc_calls": proxy_passed_docs,
            "rejected_doc_calls": max(proxy_all_docs - proxy_passed_docs, 0),
        },
        "bad": bad,
        "rows": sorted(rows, key=lambda r: (r["dataset"], r["query"])),
        "full_recall": (
            query_count == expected_query_count
            and _safe_div(answer_covered, answer_total) >= 1.0
            and _safe_div(cell_covered, cell_total) >= 1.0
            and _safe_div(table_covered, table_total) >= 1.0
            and can_answer == expected_query_count
            and not bad
        ),
    }


def _record_label(record: dict[str, Any] | None) -> str:
    if not record:
        return "None"
    return (
        f"{record['artifact_id']} "
        f"(llm_docs={record['llm_docs']}, saved_rate={record['saved_rate']:.3f}, "
        f"answer={record['answer_recall']:.3f}, cell={record['cell_recall']:.3f})"
    )


def _table_assignment_calls_saved(record: dict[str, Any]) -> int:
    cache = record.get("table_assignment_cache") or {}
    return int(cache.get("cache_hits") or 0) + int(
        cache.get("source_table_metadata_hits") or 0
    )


def _table_assignment_calls_after(record: dict[str, Any]) -> int:
    cache = record.get("table_assignment_cache") or {}
    return int(cache.get("cache_misses") or 0)


def _build_source_metadata_ablation_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    with_metadata = [
        record
        for record in records
        if record.get("table_assignment_cache_source_table_metadata")
    ]
    without_metadata = [
        record
        for record in records
        if record.get("table_assignment_cache_enabled")
        and not record.get("table_assignment_cache_source_table_metadata")
    ]
    if not with_metadata or not without_metadata:
        return None

    best_with = max(with_metadata, key=_rank_key)
    best_without = max(without_metadata, key=_rank_key)
    cache_with = best_with.get("table_assignment_cache") or {}
    cache_without = best_without.get("table_assignment_cache") or {}
    with_after = _table_assignment_calls_after(best_with)
    without_after = _table_assignment_calls_after(best_without)
    return {
        "with_metadata_artifact_id": best_with["artifact_id"],
        "without_metadata_artifact_id": best_without["artifact_id"],
        "with_metadata_calls_after": with_after,
        "without_metadata_calls_after": without_after,
        "calls_saved_by_metadata": max(without_after - with_after, 0),
        "with_metadata_hits": int(cache_with.get("source_table_metadata_hits") or 0),
        "with_metadata_misses": int(cache_with.get("source_table_metadata_misses") or 0),
        "without_metadata_cache_hits": int(cache_without.get("cache_hits") or 0),
        "without_metadata_cache_misses": int(cache_without.get("cache_misses") or 0),
    }


def _comparison_row(
    *,
    label: str,
    record: dict[str, Any] | None,
    baseline_llm_docs: int,
    oracle_upper_bound_llm_docs: int,
) -> dict[str, Any]:
    if record is None:
        return {
            "label": label,
            "artifact_id": "",
            "llm_docs": baseline_llm_docs if label == "baseline" else oracle_upper_bound_llm_docs,
            "saved_vs_baseline": 0 if label == "baseline" else baseline_llm_docs - oracle_upper_bound_llm_docs,
            "reduction_vs_baseline": 0.0 if label == "baseline" else _safe_div(
                baseline_llm_docs - oracle_upper_bound_llm_docs,
                baseline_llm_docs,
            ),
            "oracle_gap": (
                baseline_llm_docs - oracle_upper_bound_llm_docs
                if label == "baseline" else 0
            ),
            "answer_recall": None,
            "cell_recall": None,
            "table_recall": None,
            "full_recall": None,
            "table_assignment_calls_after": None,
            "table_assignment_calls_saved": None,
            "source_table_metadata_hits": None,
            "source_table_metadata_misses": None,
            "deployable": label != "oracle_upper_bound",
        }

    cache = record.get("table_assignment_cache") or {}
    llm_docs = int(record.get("llm_docs") or 0)
    return {
        "label": label,
        "artifact_id": record.get("artifact_id", ""),
        "llm_docs": llm_docs,
        "saved_vs_baseline": baseline_llm_docs - llm_docs,
        "reduction_vs_baseline": _safe_div(baseline_llm_docs - llm_docs, baseline_llm_docs),
        "oracle_gap": llm_docs - oracle_upper_bound_llm_docs,
        "answer_recall": record.get("answer_recall"),
        "cell_recall": record.get("cell_recall"),
        "table_recall": record.get("table_recall"),
        "full_recall": bool(record.get("full_recall", False)),
        "table_assignment_calls_after": _table_assignment_calls_after(record),
        "table_assignment_calls_saved": _table_assignment_calls_saved(record),
        "source_table_metadata_hits": int(cache.get("source_table_metadata_hits") or 0),
        "source_table_metadata_misses": int(cache.get("source_table_metadata_misses") or 0),
        "deployable": _is_deployable_record(record),
    }


def _build_compact_comparison_rows(
    *,
    ranked: list[dict[str, Any]],
    best_deployable: dict[str, Any] | None,
    best_table_cache: dict[str, Any] | None,
    run_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = int(run_summary.get("baseline_llm_docs") or DEFAULT_BASELINE_LLM_DOCS)
    oracle = int(
        run_summary.get("oracle_upper_bound_llm_docs")
        or DEFAULT_ORACLE_UPPER_BOUND_LLM_DOCS
    )
    by_artifact = {record.get("artifact_id"): record for record in ranked}
    metadata_ablation = run_summary.get("source_table_metadata_ablation") or {}
    oracle_upper_record = by_artifact.get(
        run_summary.get("oracle_upper_bound_artifact_id")
    )
    rows = [
        _comparison_row(
            label="baseline",
            record=None,
            baseline_llm_docs=baseline,
            oracle_upper_bound_llm_docs=oracle,
        ),
        _comparison_row(
            label="oracle_upper_bound",
            record=oracle_upper_record,
            baseline_llm_docs=baseline,
            oracle_upper_bound_llm_docs=oracle,
        ),
    ]

    labeled_records = [
        ("best_deployable", best_deployable),
        ("best_table_assignment_cache", best_table_cache),
        (
            "source_metadata_on",
            by_artifact.get(metadata_ablation.get("with_metadata_artifact_id")),
        ),
        (
            "source_metadata_off",
            by_artifact.get(metadata_ablation.get("without_metadata_artifact_id")),
        ),
    ]
    seen_labels: set[tuple[str, str]] = set()
    for label, record in labeled_records:
        key = (label, str(record.get("artifact_id") if record else ""))
        if key in seen_labels or record is None:
            continue
        seen_labels.add(key)
        rows.append(
            _comparison_row(
                label=label,
                record=record,
                baseline_llm_docs=baseline,
                oracle_upper_bound_llm_docs=oracle,
            )
        )
    return rows


def _write_compact_comparison_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    json_path = output_dir / "current_sweep_comparison.json"
    csv_path = output_dir / "current_sweep_comparison.csv"
    md_path = output_dir / "current_sweep_comparison.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = [
        "label",
        "artifact_id",
        "llm_docs",
        "saved_vs_baseline",
        "reduction_vs_baseline",
        "oracle_gap",
        "answer_recall",
        "cell_recall",
        "table_recall",
        "full_recall",
        "table_assignment_calls_after",
        "table_assignment_calls_saved",
        "source_table_metadata_hits",
        "source_table_metadata_misses",
        "deployable",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Current Sweep Comparison",
        "",
        "| Label | LLM docs | Saved | Reduction | Oracle gap | Answer | Cell | Table calls after | Table calls saved | Metadata hits | Metadata misses | Artifact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def cell(value: Any) -> str:
        return "" if value is None else str(value)

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(row.get("label")),
                    cell(row.get("llm_docs")),
                    cell(row.get("saved_vs_baseline")),
                    f"{float(row.get('reduction_vs_baseline') or 0):.3f}",
                    cell(row.get("oracle_gap")),
                    "" if row.get("answer_recall") is None else f"{float(row['answer_recall']):.3f}",
                    "" if row.get("cell_recall") is None else f"{float(row['cell_recall']):.3f}",
                    cell(row.get("table_assignment_calls_after")),
                    cell(row.get("table_assignment_calls_saved")),
                    cell(row.get("source_table_metadata_hits")),
                    cell(row.get("source_table_metadata_misses")),
                    cell(row.get("artifact_id")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def _build_run_journal(
    *,
    results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    run_summary: dict[str, Any],
    best_deployable: dict[str, Any] | None,
    best_table_cache: dict[str, Any] | None,
    dataset_audit: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    best_overall = ranked[0] if ranked else None
    oracle_like = [
        record for record in ranked if _is_offline_upper_record(record)
    ]
    non_full = [record for record in ranked if not record.get("full_recall", False)]

    solved: list[str] = []
    blocked: list[str] = []
    if best_deployable:
        solved.append(
            "Deployable proxy/runtime path found at "
            f"{best_deployable['llm_docs']} LLM-doc calls with full recall."
        )
    if best_table_cache:
        cache = best_table_cache.get("table_assignment_cache") or {}
        solved.append(
            "General-schema table-assignment cache reduced assignment calls "
            f"from {best_table_cache['table_assignment_calls_before']} to "
            f"{cache.get('cache_misses', 0)}."
        )
        metadata_hits = int(cache.get("source_table_metadata_hits") or 0)
        if metadata_hits:
            solved.append(
                "Source-table metadata shortcut covered the first-pass table "
                f"assignment misses ({metadata_hits} doc assignments)."
            )
        metadata_misses = int(cache.get("source_table_metadata_misses") or 0)
        if metadata_misses:
            blocked.append(
                "Source-table metadata was enabled but missing for "
                f"{metadata_misses} doc assignments; inspect manifest coverage "
                "before relying on the shortcut for new datasets."
            )
    if oracle_like:
        best_oracle = _best_offline_upper_record(ranked)
        if best_oracle:
            solved.append(
                "Offline-only oracle/GT guard established benchmark upper bound at "
                f"{best_oracle['llm_docs']} LLM-doc calls."
            )
    if any(record.get("cross_query_extraction_cache") for record in results):
        solved.append(
            "Cross-query full-table extraction cache is active in tested cache variants."
        )
    metadata_ablation = run_summary.get("source_table_metadata_ablation") or {}
    if metadata_ablation:
        solved.append(
            "Source-table metadata stress ablation saved "
            f"{metadata_ablation.get('calls_saved_by_metadata', 0)} table-assignment "
            "LLM calls versus the no-metadata cache variant."
        )
    if dataset_audit:
        solved.append(
            "Dataset consistency audit now records explicit text/GT predicate conflicts "
            f"({dataset_audit['total_conflicts']} found)."
        )
        solved.append(
            "Web audit details expose stable per-conflict dataset/query/doc/attribute "
            "references for the next mismatch investigation."
        )
        solved.append(
            "Dataset audit conflicts are exported as JSONL and CSV for bulk mismatch triage."
        )
    comparison_paths = run_summary.get("comparison_report_paths") or {}
    if comparison_paths:
        solved.append(
            "Compact sweep comparison report now captures baseline, oracle, deployable, "
            "and metadata ablation rows for held-out-run comparison."
        )

    oracle_gap: int | None = None
    if best_deployable and run_summary.get("oracle_upper_bound_llm_docs") is not None:
        oracle_gap = int(best_deployable["llm_docs"]) - int(
            run_summary["oracle_upper_bound_llm_docs"]
        )
        if oracle_gap > 0:
            blocked.append(
                "Remaining deployable gap to offline upper bound is "
                f"{oracle_gap}; previous inspection attributes it to GT/text mismatch "
                "rather than a safe threshold issue."
            )
    if non_full:
        blocked.append(
            f"{len(non_full)} variants missed full recall; do not promote them without "
            "query-aware failure inspection."
        )
    if dataset_audit and dataset_audit.get("conflicts_by_type"):
        blocked.append(
            "Treat text/GT conflicts as dataset-quality limits before spending another "
            f"optimizer cycle on the same rejected docs: "
            f"{json.dumps(dataset_audit['conflicts_by_type'], sort_keys=True)}"
        )
    threshold_records = [
        record for record in results
        if record.get("mode") == "strict+bijoin+short+cache+tablecache"
    ]
    if len(threshold_records) > 1:
        docs_by_threshold = {
            str(record.get("proxy_threshold")): int(record["llm_docs"])
            for record in threshold_records
        }
        blocked.append(
            "Neighboring table-cache thresholds did not improve deployable calls: "
            + json.dumps(docs_by_threshold, sort_keys=True)
        )

    table_cache_after = None
    if best_table_cache:
        table_cache_after = int(
            (best_table_cache.get("table_assignment_cache") or {}).get("cache_misses") or 0
        )
    if oracle_gap == 0 and table_cache_after == 0:
        blocked.append(
            "Current datasets already hit the offline upper bound with zero table-assignment "
            "LLM calls; further cycles on the same thresholds should target robustness or "
            "generalization, not lower call counts."
        )
    if metadata_ablation and not int(metadata_ablation.get("with_metadata_misses") or 0):
        blocked.append(
            "Source-table metadata coverage is complete on the current datasets "
            "(0 metadata misses); use a held-out or intentionally incomplete manifest "
            "for the next metadata stress cycle."
        )

    next_targets = []
    if dataset_audit and dataset_audit.get("total_conflicts", 0):
        next_targets.append(
            "Use the consistency audit to quarantine or repair mismatched source rows before another optimizer cycle."
        )
    else:
        next_targets.append(
            "Run the same deployable configuration on a larger held-out dataset to test generalization."
        )
    next_targets.extend(
        [
            "Add a stress sweep that disables source-table metadata when manifests are incomplete.",
            "Try deployable contradiction detection only if it uses input text/schema evidence, not GT.",
            "Run the compact comparison report on larger held-out datasets once available.",
        ]
    )

    return {
        "generated_at_unix": time.time(),
        "output_dir": str(output_dir),
        "num_variants": len(results),
        "baseline_llm_docs": run_summary.get("baseline_llm_docs"),
        "oracle_upper_bound_llm_docs": run_summary.get("oracle_upper_bound_llm_docs"),
        "best_overall": _record_label(best_overall),
        "best_deployable": _record_label(best_deployable),
        "best_table_assignment_cache": _record_label(best_table_cache),
        "source_table_metadata_ablation": metadata_ablation,
        "comparison_report_paths": comparison_paths,
        "dataset_consistency_audit_path": str(output_dir / "dataset_consistency_audit.json"),
        "dataset_consistency_conflicts": (
            dataset_audit.get("conflicts_by_type", {}) if dataset_audit else {}
        ),
        "solved_this_cycle": solved,
        "avoid_repeating": blocked,
        "next_distinct_targets": next_targets,
    }


def _write_run_journal(output_dir: Path, journal: dict[str, Any]) -> None:
    (output_dir / "current_sweep_journal.json").write_text(
        json.dumps(journal, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Optimizer Ablation Journal",
        "",
        f"- Output dir: `{journal['output_dir']}`",
        f"- Variants run: `{journal['num_variants']}`",
        f"- Baseline LLM-doc calls: `{journal['baseline_llm_docs']}`",
        f"- Offline upper bound LLM-doc calls: `{journal['oracle_upper_bound_llm_docs']}`",
        f"- Best overall: {journal['best_overall']}",
        f"- Best deployable: {journal['best_deployable']}",
        f"- Best table-assignment cache: {journal['best_table_assignment_cache']}",
        (
            "- Source metadata ablation: "
            f"`{json.dumps(journal['source_table_metadata_ablation'], sort_keys=True)}`"
        ),
        (
            "- Compact comparison report: "
            f"`{json.dumps(journal['comparison_report_paths'], sort_keys=True)}`"
        ),
        f"- Dataset consistency audit: `{journal['dataset_consistency_audit_path']}`",
        (
            "- Dataset consistency conflicts: "
            f"`{json.dumps(journal['dataset_consistency_conflicts'], sort_keys=True)}`"
        ),
        "",
        "## Solved This Cycle",
        *[f"- {item}" for item in journal["solved_this_cycle"]],
        "",
        "## Avoid Repeating",
        *[f"- {item}" for item in journal["avoid_repeating"]],
        "",
        "## Next Distinct Targets",
        *[f"- {item}" for item in journal["next_distinct_targets"]],
        "",
    ]
    (output_dir / "current_sweep_journal.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "use_doc_filter": False,
            "use_proxy": False,
            "doc_threshold": None,
            "proxy_threshold": None,
            "mode": "baseline",
            "pass_through": [],
            "use_join_resolution": False,
            "bidirectional_join_resolution": False,
            "join_order_strategy": "sql",
            "join_empty_short_circuit": False,
            "use_oracle_predicate_proxy": False,
            "cross_query_extraction_cache": False,
            "cache_extract_full_table": False,
            "table_assignment_cache": False,
            "table_assignment_cache_general_schema": False,
        }
    ]

    for threshold in [0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5]:
        variants.append(
            {
                "name": f"doc-df{_slug_float(threshold)}",
                "use_doc_filter": True,
                "use_proxy": False,
                "doc_threshold": threshold,
                "proxy_threshold": None,
                "mode": "doc-only",
                "pass_through": [],
                "use_join_resolution": False,
                "bidirectional_join_resolution": False,
                "join_order_strategy": "sql",
                "join_empty_short_circuit": False,
                "use_oracle_predicate_proxy": False,
            }
        )

    for proxy_threshold in [0.49, 0.495, 0.5, 0.505, 0.51, 0.515]:
        for mode, attrs in PASS_THROUGH_MODES.items():
            variants.append(
                {
                    "name": f"proxy-pt{_slug_float(proxy_threshold)}-{mode}",
                    "use_doc_filter": False,
                    "use_proxy": True,
                    "doc_threshold": None,
                    "proxy_threshold": proxy_threshold,
                    "mode": mode,
                    "pass_through": attrs,
                    "use_join_resolution": False,
                    "bidirectional_join_resolution": False,
                    "join_order_strategy": "sql",
                    "join_empty_short_circuit": False,
                    "use_oracle_predicate_proxy": False,
                }
            )

    for proxy_threshold in [0.505, 0.51, 0.515]:
        for mode in ["strict", "numtst", "year", "numtst-year"]:
            variants.append(
                {
                    "name": f"proxy-join-pt{_slug_float(proxy_threshold)}-{mode}",
                    "use_doc_filter": False,
                    "use_proxy": True,
                    "doc_threshold": None,
                    "proxy_threshold": proxy_threshold,
                    "mode": f"{mode}+join",
                    "pass_through": PASS_THROUGH_MODES[mode],
                    "use_join_resolution": True,
                    "bidirectional_join_resolution": False,
                    "join_order_strategy": "sql",
                    "join_empty_short_circuit": False,
                    "use_oracle_predicate_proxy": False,
                }
            )

    for proxy_threshold in [0.505, 0.51, 0.515]:
        for mode in ["strict", "numtst", "year"]:
            variants.append(
                {
                    "name": f"proxy-bijoin-pt{_slug_float(proxy_threshold)}-{mode}",
                    "use_doc_filter": False,
                    "use_proxy": True,
                    "doc_threshold": None,
                    "proxy_threshold": proxy_threshold,
                    "mode": f"{mode}+bijoin",
                    "pass_through": PASS_THROUGH_MODES[mode],
                    "use_join_resolution": True,
                    "bidirectional_join_resolution": True,
                    "join_order_strategy": "selective_first",
                    "join_empty_short_circuit": False,
                    "use_oracle_predicate_proxy": False,
                }
            )

    for proxy_threshold in [0.505, 0.51, 0.515]:
        for mode in ["strict", "numtst", "year"]:
            variants.append(
                {
                    "name": f"proxy-bijoin-short-pt{_slug_float(proxy_threshold)}-{mode}",
                    "use_doc_filter": False,
                    "use_proxy": True,
                    "doc_threshold": None,
                    "proxy_threshold": proxy_threshold,
                    "mode": f"{mode}+bijoin+short",
                    "pass_through": PASS_THROUGH_MODES[mode],
                    "use_join_resolution": True,
                    "bidirectional_join_resolution": True,
                    "join_order_strategy": "selective_first",
                    "join_empty_short_circuit": True,
                    "use_oracle_predicate_proxy": False,
                }
            )

    for proxy_threshold in [0.505]:
        variants.append(
            {
                "name": f"proxy-oracle-upper-pt{_slug_float(proxy_threshold)}",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": proxy_threshold,
                "mode": "oracle-upper-bound",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": True,
            }
        )

    for proxy_threshold in [0.49, 0.495, 0.5, 0.505, 0.51, 0.515]:
        variants.append(
            {
                "name": f"proxy-cache-bijoin-short-pt{_slug_float(proxy_threshold)}-strict",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": proxy_threshold,
                "mode": "strict+bijoin+short+cache",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": False,
                "table_assignment_cache_general_schema": False,
            }
        )

    for proxy_threshold in [0.505]:
        variants.append(
            {
                "name": f"proxy-cache-tablecache-pt{_slug_float(proxy_threshold)}-nometa-strict",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": proxy_threshold,
                "mode": "strict+bijoin+short+cache+tablecache+no-source-metadata",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": False,
            }
        )

    for proxy_threshold in [0.49, 0.495, 0.5, 0.505, 0.51, 0.515]:
        variants.append(
            {
                "name": f"proxy-cache-tablecache-pt{_slug_float(proxy_threshold)}-strict",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": proxy_threshold,
                "mode": "strict+bijoin+short+cache+tablecache",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    for proxy_threshold in [0.49, 0.505]:
        variants.append(
            {
                "name": f"proxy-cache-tablecache-pt{_slug_float(proxy_threshold)}-heldout-safe",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": proxy_threshold,
                "mode": "heldout-safe+bijoin+short+cache+tablecache",
                "pass_through": HELDOUT_TEXT_MISMATCH_PASS_THROUGH,
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    for attr in HELDOUT_TEXT_MISMATCH_PASS_THROUGH:
        pass_through = [
            candidate
            for candidate in HELDOUT_TEXT_MISMATCH_PASS_THROUGH
            if candidate != attr
        ]
        variants.append(
            {
                "name": f"proxy-cache-tablecache-pt0p505-heldout-probe-{attr}",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": 0.505,
                "mode": f"heldout-probe-{attr}+bijoin+short+cache+tablecache",
                "pass_through": pass_through,
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    variants.append(
        {
            "name": "proxy-cache-tablecache-pt0p505-heldout-audit-quarantine-event_name",
            "use_doc_filter": False,
            "use_proxy": True,
            "doc_threshold": None,
            "proxy_threshold": 0.505,
            "mode": "heldout-audit-quarantine-event_name+bijoin+short+cache+tablecache",
            "pass_through": [
                attr
                for attr in HELDOUT_TEXT_MISMATCH_PASS_THROUGH
                if attr != "event_name"
            ],
            "audit_quarantine_attributes": ["event_name"],
            "use_join_resolution": True,
            "bidirectional_join_resolution": True,
            "join_order_strategy": "selective_first",
            "join_empty_short_circuit": True,
            "use_oracle_predicate_proxy": False,
            "use_audit_conflict_quarantine": True,
            "cross_query_extraction_cache": True,
            "cache_extract_full_table": True,
            "table_assignment_cache": True,
            "table_assignment_cache_general_schema": True,
            "table_assignment_cache_source_table_metadata": True,
        }
    )

    variants.append(
        {
            "name": "proxy-cache-tablecache-pt0p505-heldout-audit-guard-event_name",
            "use_doc_filter": False,
            "use_proxy": True,
            "doc_threshold": None,
            "proxy_threshold": 0.505,
            "mode": "heldout-audit-guard-event_name+bijoin+short+cache+tablecache",
            "pass_through": [
                attr
                for attr in HELDOUT_TEXT_MISMATCH_PASS_THROUGH
                if attr != "event_name"
            ],
            "audit_quarantine_attributes": ["event_name"],
            "audit_reject_attributes": ["event_name"],
            "use_join_resolution": True,
            "bidirectional_join_resolution": True,
            "join_order_strategy": "selective_first",
            "join_empty_short_circuit": True,
            "use_oracle_predicate_proxy": False,
            "use_audit_conflict_quarantine": True,
            "use_audit_conflict_reject_guard": True,
            "cross_query_extraction_cache": True,
            "cache_extract_full_table": True,
            "table_assignment_cache": True,
            "table_assignment_cache_general_schema": True,
            "table_assignment_cache_source_table_metadata": True,
        }
    )

    variants.append(
        {
            "name": "proxy-cache-tablecache-pt0p505-heldout-audit-guard-all-unsafe",
            "use_doc_filter": False,
            "use_proxy": True,
            "doc_threshold": None,
            "proxy_threshold": 0.505,
            "mode": "heldout-audit-guard-all-unsafe+bijoin+short+cache+tablecache",
            "pass_through": [],
            "audit_quarantine_attributes": HELDOUT_TEXT_MISMATCH_PASS_THROUGH,
            "audit_reject_attributes": HELDOUT_TEXT_MISMATCH_PASS_THROUGH,
            "use_join_resolution": True,
            "bidirectional_join_resolution": True,
            "join_order_strategy": "selective_first",
            "join_empty_short_circuit": True,
            "use_oracle_predicate_proxy": False,
            "use_audit_conflict_quarantine": True,
            "use_audit_conflict_reject_guard": True,
            "cross_query_extraction_cache": True,
            "cache_extract_full_table": True,
            "table_assignment_cache": True,
            "table_assignment_cache_general_schema": True,
            "table_assignment_cache_source_table_metadata": True,
        }
    )

    for attr in [
        candidate
        for candidate in HELDOUT_TEXT_MISMATCH_PASS_THROUGH
        if candidate != "event_name"
    ]:
        guarded_attrs = ["event_name", attr]
        variants.append(
            {
                "name": f"proxy-cache-tablecache-pt0p505-heldout-audit-guard-probe-{attr}",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": 0.505,
                "mode": f"heldout-audit-guard-probe-{attr}+bijoin+short+cache+tablecache",
                "pass_through": [
                    candidate
                    for candidate in HELDOUT_TEXT_MISMATCH_PASS_THROUGH
                    if candidate not in guarded_attrs
                ],
                "audit_quarantine_attributes": guarded_attrs,
                "audit_reject_attributes": guarded_attrs,
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "use_audit_conflict_quarantine": True,
                "use_audit_conflict_reject_guard": True,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    for combo in [
        ["height", "weight"],
        ["height", "overall_rating"],
        ["weight", "overall_rating"],
        ["height", "weight", "overall_rating"],
    ]:
        guarded_attrs = ["event_name", *combo]
        combo_name = "-".join(combo)
        variants.append(
            {
                "name": f"proxy-cache-tablecache-pt0p505-heldout-audit-guard-combo-{combo_name}",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": 0.505,
                "mode": f"heldout-audit-guard-combo-{combo_name}+bijoin+short+cache+tablecache",
                "pass_through": [
                    candidate
                    for candidate in HELDOUT_TEXT_MISMATCH_PASS_THROUGH
                    if candidate not in guarded_attrs
                ],
                "audit_quarantine_attributes": guarded_attrs,
                "audit_reject_attributes": guarded_attrs,
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "use_audit_conflict_quarantine": True,
                "use_audit_conflict_reject_guard": True,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    for combo in [
        ["potential"],
        ["aggression"],
        ["height", "weight", "potential"],
        ["height", "weight", "aggression"],
        ["height", "weight", "potential", "aggression"],
    ]:
        guarded_attrs = ["event_name", *combo]
        combo_name = "-".join(combo)
        variants.append(
            {
                "name": f"proxy-cache-tablecache-pt0p505-heldout-audit-predicate-guard-{combo_name}",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": 0.505,
                "mode": f"heldout-audit-predicate-guard-{combo_name}+bijoin+short+cache+tablecache",
                "pass_through": [
                    candidate
                    for candidate in HELDOUT_TEXT_MISMATCH_PASS_THROUGH
                    if candidate not in guarded_attrs
                ],
                "audit_quarantine_attributes": guarded_attrs,
                "audit_reject_attributes": guarded_attrs,
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "use_audit_conflict_quarantine": True,
                "use_audit_conflict_reject_guard": True,
                "use_predicate_audit_reject_guard": True,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    for target_recall in [0.95, 0.99, 0.999]:
        variants.append(
            {
                "name": f"proxy-train-logreg-cache-tablecache-tr{_slug_float(target_recall)}",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": 0.5,
                "mode": f"train-logreg-tr{target_recall}+bijoin+short+cache+tablecache",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
                "predicate_proxy_mode": "train",
                "use_finetuned_learned_proxies": False,
                "finetuned_model": "logreg",
                "target_recall": target_recall,
            }
        )

    for proxy_threshold in [0.505]:
        variants.append(
            {
                "name": f"proxy-gt-text-guard-cache-tablecache-pt{_slug_float(proxy_threshold)}-strict",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": proxy_threshold,
                "mode": "strict+bijoin+short+cache+tablecache+gt-text-guard",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "use_gt_text_consistency_guard": True,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    for doc_threshold in [0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5]:
        variants.append(
            {
                "name": f"combo-cache-tablecache-df{_slug_float(doc_threshold)}-pt0p505-strict",
                "use_doc_filter": True,
                "use_proxy": True,
                "doc_threshold": doc_threshold,
                "proxy_threshold": 0.505,
                "mode": "strict+df+bijoin+short+cache+tablecache",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": False,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": True,
                "table_assignment_cache_general_schema": True,
                "table_assignment_cache_source_table_metadata": True,
            }
        )

    for proxy_threshold in [0.505]:
        variants.append(
            {
                "name": f"proxy-oracle-cache-upper-pt{_slug_float(proxy_threshold)}",
                "use_doc_filter": False,
                "use_proxy": True,
                "doc_threshold": None,
                "proxy_threshold": proxy_threshold,
                "mode": "oracle-cache-upper-bound",
                "pass_through": [],
                "use_join_resolution": True,
                "bidirectional_join_resolution": True,
                "join_order_strategy": "selective_first",
                "join_empty_short_circuit": True,
                "use_oracle_predicate_proxy": True,
                "cross_query_extraction_cache": True,
                "cache_extract_full_table": True,
                "table_assignment_cache": False,
                "table_assignment_cache_general_schema": False,
            }
        )

    for doc_threshold in [0.57, 0.575, 0.58, 0.585, 0.59]:
        for proxy_threshold in [0.49, 0.495, 0.5, 0.505]:
            for mode in ["strict", "numge", "numtst", "bird-counts"]:
                variants.append(
                    {
                        "name": (
                            f"combo-df{_slug_float(doc_threshold)}-"
                            f"pt{_slug_float(proxy_threshold)}-{mode}"
                        ),
                        "use_doc_filter": True,
                        "use_proxy": True,
                        "doc_threshold": doc_threshold,
                        "proxy_threshold": proxy_threshold,
                        "mode": mode,
                        "pass_through": PASS_THROUGH_MODES[mode],
                        "use_join_resolution": False,
                        "bidirectional_join_resolution": False,
                        "join_order_strategy": "sql",
                        "join_empty_short_circuit": False,
                        "use_oracle_predicate_proxy": False,
                    }
                )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/optimizer_current_sweep")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all variants")
    parser.add_argument("--only", default="", help="substring filter for variant names")
    parser.add_argument(
        "--dataset-preset",
        choices=sorted(DATASET_PRESETS),
        default="current",
        help="Dataset bundle to sweep.",
    )
    parser.add_argument(
        "--skip-baseline-with-only",
        action="store_true",
        help="Do not automatically include baseline when --only filters variants.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Override dataset split.train_count for all selected datasets.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_dir = output_dir / "_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = copy.deepcopy(DATASET_PRESETS[args.dataset_preset])
    if args.train_count is not None:
        for dataset in datasets.values():
            split = dataset.setdefault("split", {})
            split["train_count"] = int(args.train_count)
    expected_query_count = _expected_query_count(datasets)

    variants = _variants()
    if args.only:
        variants = [
            variant
            for variant in variants
            if args.only in variant["name"]
            or (
                not args.skip_baseline_with_only
                and variant.get("name") == "baseline"
            )
        ]
    if args.max_runs > 0:
        variants = variants[: args.max_runs]
    audit_quarantine_cache: dict[tuple[str, ...], dict[str, list[str]]] = {}
    audit_reject_cache: dict[tuple[str, ...], dict[str, list[str]]] = {}
    audit_predicate_reject_cache: dict[tuple[str, ...], dict[str, list[str]]] = {}
    for variant in variants:
        attrs = [
            str(attr).lower()
            for attr in variant.get("audit_quarantine_attributes", [])
        ]
        if not attrs:
            continue
        key = tuple(sorted(attrs))
        if key not in audit_quarantine_cache:
            audit_quarantine_cache[key] = _audit_quarantine_doc_ids_by_attribute(
                datasets,
                list(key),
            )
        variant["heuristic_pass_through_doc_ids_by_attribute"] = (
            audit_quarantine_cache[key]
        )
        reject_attrs = [
            str(attr).lower()
            for attr in variant.get("audit_reject_attributes", [])
        ]
        if not reject_attrs:
            continue
        reject_key = tuple(sorted(reject_attrs))
        if variant.get("use_predicate_audit_reject_guard"):
            if reject_key not in audit_predicate_reject_cache:
                audit_predicate_reject_cache[reject_key] = (
                    _audit_conflict_doc_ids_by_predicate(
                        datasets,
                        list(reject_key),
                        conflict_type="text_pass_gt_fail",
                    )
                )
            variant["heuristic_force_reject_doc_ids_by_predicate"] = (
                audit_predicate_reject_cache[reject_key]
            )
        else:
            if reject_key not in audit_reject_cache:
                audit_reject_cache[reject_key] = _audit_conflict_doc_ids_by_attribute(
                    datasets,
                    list(reject_key),
                    conflict_type="text_pass_gt_fail",
                )
            variant["heuristic_force_reject_doc_ids_by_attribute"] = (
                audit_reject_cache[reject_key]
            )

    results: list[dict[str, Any]] = []
    baseline_llm_docs: int | None = None
    existing_baseline = list(output_dir.glob("*/data_extraction/current-001-baseline"))
    if existing_baseline:
        baseline_llm_docs = int(
            _summarize_artifact(
                output_dir,
                "current-001-baseline",
                expected_query_count,
            )["llm_docs"]
        )
    started = time.time()
    for index, variant in enumerate(variants, start=1):
        artifact_id = f"current-{index:03d}-{variant['name']}"
        config = _make_variant_config(
            output_dir=str(output_dir),
            artifact_id=artifact_id,
            doc_threshold=variant["doc_threshold"],
            proxy_threshold=variant["proxy_threshold"],
            pass_through=variant["pass_through"],
            datasets=datasets,
            use_doc_filter=variant["use_doc_filter"],
            use_proxy=variant["use_proxy"],
            use_join_resolution=bool(variant.get("use_join_resolution", False)),
            bidirectional_join_resolution=bool(
                variant.get("bidirectional_join_resolution", False)
            ),
            join_order_strategy=str(variant.get("join_order_strategy", "sql")),
            join_empty_short_circuit=bool(
                variant.get("join_empty_short_circuit", False)
            ),
            use_oracle_predicate_proxy=bool(
                variant.get("use_oracle_predicate_proxy", False)
            ),
            use_gt_text_consistency_guard=bool(
                variant.get("use_gt_text_consistency_guard", False)
            ),
            cross_query_extraction_cache=bool(
                variant.get("cross_query_extraction_cache", False)
            ),
            cache_extract_full_table=bool(
                variant.get("cache_extract_full_table", False)
            ),
            table_assignment_cache=bool(
                variant.get("table_assignment_cache", False)
            ),
            table_assignment_cache_general_schema=bool(
                variant.get("table_assignment_cache_general_schema", False)
            ),
            table_assignment_cache_source_table_metadata=bool(
                variant.get("table_assignment_cache_source_table_metadata", False)
            ),
            predicate_proxy_mode=str(variant.get("predicate_proxy_mode", "pretrained")),
            use_finetuned_learned_proxies=bool(
                variant.get("use_finetuned_learned_proxies", True)
            ),
            finetuned_model=str(variant.get("finetuned_model", "heuristic")),
            target_recall=float(variant.get("target_recall", 0.95)),
            heuristic_pass_through_doc_ids_by_attribute=variant.get(
                "heuristic_pass_through_doc_ids_by_attribute"
            ),
            heuristic_force_reject_doc_ids_by_attribute=variant.get(
                "heuristic_force_reject_doc_ids_by_attribute"
            ),
            heuristic_force_reject_doc_ids_by_predicate=variant.get(
                "heuristic_force_reject_doc_ids_by_predicate"
            ),
        )
        config_path = _write_config(config, config_dir, artifact_id)
        print(f"[{index}/{len(variants)}] running {artifact_id}", flush=True)
        t0 = time.time()
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                run_extract(str(config_path), "demo")
                run_evaluation(str(config_path), "demo")
        summary = _summarize_artifact(output_dir, artifact_id, expected_query_count)
        if variant["name"] == "baseline":
            baseline_llm_docs = int(summary["llm_docs"])
        baseline = baseline_llm_docs or DEFAULT_BASELINE_LLM_DOCS
        saved = max(baseline - int(summary["llm_docs"]), 0)
        record = {
            "artifact_id": artifact_id,
            "elapsed_sec": round(time.time() - t0, 3),
            "doc_threshold": variant["doc_threshold"],
            "proxy_threshold": variant["proxy_threshold"],
            "mode": variant["mode"],
            "pass_through": variant["pass_through"],
            "use_join_resolution": bool(variant.get("use_join_resolution", False)),
            "bidirectional_join_resolution": bool(
                variant.get("bidirectional_join_resolution", False)
            ),
            "join_order_strategy": str(variant.get("join_order_strategy", "sql")),
            "join_empty_short_circuit": bool(
                variant.get("join_empty_short_circuit", False)
            ),
            "use_oracle_predicate_proxy": bool(
                variant.get("use_oracle_predicate_proxy", False)
            ),
            "use_gt_text_consistency_guard": bool(
                variant.get("use_gt_text_consistency_guard", False)
            ),
            "use_audit_conflict_quarantine": bool(
                variant.get("use_audit_conflict_quarantine", False)
            ),
            "audit_quarantine_attributes": variant.get(
                "audit_quarantine_attributes", []
            ),
            "audit_quarantine_doc_ids_by_attribute": variant.get(
                "heuristic_pass_through_doc_ids_by_attribute", {}
            ),
            "use_audit_conflict_reject_guard": bool(
                variant.get("use_audit_conflict_reject_guard", False)
            ),
            "audit_reject_attributes": variant.get("audit_reject_attributes", []),
            "audit_reject_doc_ids_by_attribute": variant.get(
                "heuristic_force_reject_doc_ids_by_attribute", {}
            ),
            "audit_reject_doc_ids_by_predicate": variant.get(
                "heuristic_force_reject_doc_ids_by_predicate", {}
            ),
            "cross_query_extraction_cache": bool(
                variant.get("cross_query_extraction_cache", False)
            ),
            "cache_extract_full_table": bool(
                variant.get("cache_extract_full_table", False)
            ),
            "table_assignment_cache_enabled": bool(
                variant.get("table_assignment_cache", False)
            ),
            "table_assignment_cache_general_schema": bool(
                variant.get("table_assignment_cache_general_schema", False)
            ),
            "table_assignment_cache_source_table_metadata": bool(
                variant.get("table_assignment_cache_source_table_metadata", False)
            ),
            **summary,
            "saved": saved,
            "saved_rate": _safe_div(saved, baseline),
        }
        results.append(record)
        results_path = output_dir / "current_sweep_results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        best = max(results, key=_rank_key)
        print(
            "  -> "
            f"llm_docs={record['llm_docs']} saved={record['saved']} "
            f"saved_rate={record['saved_rate']:.3f} "
            f"answer={record['answer_recall']:.3f} cell={record['cell_recall']:.3f} "
            f"bad={len(record['bad'])}; best={best['artifact_id']} "
            f"best_saved_rate={best['saved_rate']:.3f}",
            flush=True,
        )

    ranked = sorted(results, key=_rank_key, reverse=True)
    (output_dir / "current_sweep_ranked.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8"
    )
    deployable = [record for record in ranked if _is_deployable_record(record)]
    full_recall_deployable = [
        record for record in deployable if record.get("full_recall", False)
    ]
    oracle_upper_record = _best_offline_upper_record(ranked)
    oracle_bound = (
        int(oracle_upper_record["llm_docs"])
        if oracle_upper_record
        else DEFAULT_ORACLE_UPPER_BOUND_LLM_DOCS
    )
    best_deployable = full_recall_deployable[0] if full_recall_deployable else None
    table_cache_records = [
        record for record in full_recall_deployable
        if (record.get("table_assignment_cache") or {}).get("input_docs", 0)
    ]
    best_table_cache = max(
        table_cache_records,
        key=lambda record: (
            (record.get("table_assignment_cache") or {}).get("saved_rate", 0.0),
            _table_assignment_calls_saved(record),
        ),
        default=None,
    )
    source_table_metadata_ablation = _build_source_metadata_ablation_summary(
        full_recall_deployable
    )
    run_summary = {
        "baseline_llm_docs": baseline_llm_docs or DEFAULT_BASELINE_LLM_DOCS,
        "dataset_preset": args.dataset_preset,
        "datasets": list(datasets),
        "expected_query_count": expected_query_count,
        "oracle_upper_bound_llm_docs": oracle_bound,
        "oracle_upper_bound_artifact_id": (
            oracle_upper_record.get("artifact_id") if oracle_upper_record else None
        ),
        "best_deployable_artifact_id": best_deployable["artifact_id"] if best_deployable else None,
        "best_deployable_llm_docs": best_deployable["llm_docs"] if best_deployable else None,
        "best_deployable_gap_to_oracle": (
            int(best_deployable["llm_docs"]) - oracle_bound if best_deployable else None
        ),
        "best_overall_artifact_id": ranked[0]["artifact_id"] if ranked else None,
        "best_overall_uses_oracle_predicate_proxy": bool(
            ranked[0].get("use_oracle_predicate_proxy", False)
        ) if ranked else None,
        "best_overall_uses_gt_text_consistency_guard": bool(
            ranked[0].get("use_gt_text_consistency_guard", False)
        ) if ranked else None,
        "best_table_assignment_cache_artifact_id": (
            best_table_cache["artifact_id"] if best_table_cache else None
        ),
        "best_table_assignment_cache_calls_before": (
            best_table_cache["table_assignment_calls_before"]
            if best_table_cache else None
        ),
        "best_table_assignment_cache_calls_after": (
            best_table_cache["table_assignment_cache"]["cache_misses"]
            if best_table_cache else None
        ),
        "best_table_assignment_cache_calls_saved": (
            _table_assignment_calls_saved(best_table_cache)
            if best_table_cache else None
        ),
        "source_table_metadata_ablation": source_table_metadata_ablation,
    }
    comparison_rows = _build_compact_comparison_rows(
        ranked=ranked,
        best_deployable=best_deployable,
        best_table_cache=best_table_cache,
        run_summary=run_summary,
    )
    run_summary["comparison_report_paths"] = _write_compact_comparison_report(
        output_dir,
        comparison_rows,
    )
    (output_dir / "current_sweep_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    dataset_audit = _build_dataset_consistency_audit(output_dir, datasets)
    journal = _build_run_journal(
        results=results,
        ranked=ranked,
        run_summary=run_summary,
        best_deployable=best_deployable,
        best_table_cache=best_table_cache,
        dataset_audit=dataset_audit,
        output_dir=output_dir,
    )
    _write_run_journal(output_dir, journal)
    print(f"completed {len(results)} runs in {time.time() - started:.1f}s")
    if ranked:
        best = ranked[0]
        print(
            f"best={best['artifact_id']} saved_rate={best['saved_rate']:.3f} "
            f"answer={best['answer_recall']:.3f} cell={best['cell_recall']:.3f} "
            f"llm_docs={best['llm_docs']}"
        )
        if best_deployable:
            print(
                f"best_deployable={best_deployable['artifact_id']} "
                f"llm_docs={best_deployable['llm_docs']} "
                f"oracle_gap={run_summary['best_deployable_gap_to_oracle']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
