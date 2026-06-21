"""Prepare and run DeepSeek experiments for curated query candidates."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_JSON = ROOT / "outputs" / "query_curation_expanded_v4" / "generated_query_candidates.json"
CONFIG_ROOT = ROOT / "configs" / "examples"
OUTPUT_ROOT = ROOT / "outputs" / "deepseek_query_curation_v4_aligned"
REPORT_ROOT = OUTPUT_ROOT / "reports"
LOG_ROOT = REPORT_ROOT / "command_logs"

VARIANTS = ("noalpha", "alpha")

DEFAULT_SELECTION: dict[str, tuple[dict[str, Any], ...]] = {
    "bird.schools_demo": (
        {"bucket": 0.03, "join": False},
        {"bucket": 0.20, "join": False},
        {"bucket": 0.80, "join": False},
    ),
    "fda.fda.no_chunk": (
        {"bucket": 0.03, "join": False},
    ),
    "galois.premierleague.default_task": (
        {"bucket": 0.80, "join": False},
    ),
    "spider.college_demo": (
        {"bucket": 0.03, "join": False},
        {"bucket": 0.20, "join": False},
        {"bucket": 0.40, "join": True},
        {"bucket": 0.60, "join": True},
    ),
}


@dataclass(frozen=True)
class PreparedDataset:
    source_dataset_id: str
    dataset_id: str
    slug: str
    root: Path
    query_ids: list[str]
    train_count: int


def _load_dotenv() -> None:
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()
    return text or "dataset"


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _candidate_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    candidates = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    return {
        (str(item.get("dataset_id")), str(item.get("query_id"))): item
        for item in candidates
        if item.get("dataset_id") and item.get("query_id")
    }


def _select_queries() -> list[dict[str, Any]]:
    candidates = [
        item
        for item in json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
        if item.get("dataset_id") and item.get("query_id")
        and (item.get("profile") or {}).get("curation_status") == "keep"
        and (item.get("profile") or {}).get("provenance_status") == "ok"
    ]
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for dataset_id, specs in DEFAULT_SELECTION.items():
        dataset_rows = [item for item in candidates if item.get("dataset_id") == dataset_id]
        for spec in specs:
            bucket = float(spec["bucket"])
            prefer_join = spec.get("join")
            rows = [
                item
                for item in dataset_rows
                if abs(float((item.get("meta") or {}).get("coverage_bucket", -1)) - bucket) < 1e-9
            ]
            if prefer_join is not None:
                preferred = [
                    item
                    for item in rows
                    if bool((item.get("meta") or {}).get("is_join")) == bool(prefer_join)
                ]
                if preferred:
                    rows = preferred
            if not rows:
                continue
            rows.sort(
                key=lambda item: (
                    int((item.get("profile") or {}).get("eval_required_cells") or 10**9),
                    int((item.get("profile") or {}).get("eval_relevant_docs") or 10**9),
                    str(item.get("query_id")),
                )
            )
            candidate = dict(rows[0])
            key = (dataset_id, str(candidate["query_id"]))
            if key in used:
                continue
            used.add(key)
            selected.append(candidate)
    return selected


def _source_manifest(candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(candidate["profile"]["dataset_root"])
    manifest = _read_yaml(root / "manifest.yaml")
    return root, manifest


def _write_dataset(source_dataset_id: str, queries: list[dict[str, Any]]) -> PreparedDataset:
    source_root, manifest = _source_manifest(queries[0])
    slug = _slug(source_dataset_id)
    return PreparedDataset(
        source_dataset_id=source_dataset_id,
        dataset_id=source_dataset_id,
        slug=slug,
        root=source_root,
        query_ids=[str(item["query_id"]) for item in queries],
        train_count=25,
    )


def prepare_datasets() -> list[PreparedDataset]:
    selected = _select_queries()
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in selected:
        by_dataset.setdefault(str(item["dataset_id"]), []).append(item)
    return [
        _write_dataset(dataset_id, queries)
        for dataset_id, queries in sorted(by_dataset.items())
    ]


def _base_config(prepared: list[PreparedDataset], variant: str) -> dict[str, Any]:
    artifact_id = f"deepseek-query-curation-v4-aligned-{variant}-v1"
    datasets = {
        item.dataset_id: {
            "loader": "hf_manifest",
            "root": str(item.root.relative_to(ROOT)),
            "query_ids": item.query_ids,
            "loader_options": {"manifest": "manifest.yaml"},
            "split": {"train_count": item.train_count, "strategy": "prefix"},
        }
        for item in prepared
    }
    return {
        "config_version": "2.1.1",
        "project": {"name": f"deepseek-query-curation-{variant}", "seed": 42},
        "runtime": {
            "output_dir": str(OUTPUT_ROOT.relative_to(ROOT)),
            "log_dir": "logs",
            "output_layout": "dataset_stage",
            "artifact_id": artifact_id,
            "console_log_level": "WARNING",
            "force_rerun": True,
        },
        "models": {
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key_env": "DEEPSEEK_API_KEY",
                "structured_backend": "json",
                "max_retries": 2,
                "wait_time": 1,
                "temperature": 0,
                "max_tokens": 1024,
            },
            "embedding": {
                "provider": "none",
                "model": "none",
                "enabled": False,
                "api_key_env": None,
            },
        },
        "datasets": datasets,
        "stages": {
            "data_extraction": {
                "enabled": True,
                "schema_source": "ground_truth",
                "oracle": "llm",
                "prompts": {
                    "prompt_table": "data_extraction_table",
                    "prompt_attr": "data_extraction_attr",
                },
                "options": {
                    "force_rerun": True,
                    "result_save_interval": 1,
                    "eval": {
                        "mode": "deepseek",
                        "llm_model": "deepseek-chat",
                        "structured_backend": "json",
                        "max_retries": 2,
                        "wait_time": 1,
                        "prompts": {
                            "data_extraction_cmp_str": "data_extraction_cmp_str"
                        },
                    },
                },
            }
        },
        "experiments": {
            "demo": {
                "datasets": [item.dataset_id for item in prepared],
                "stages": ["data_extraction"],
                "artifact_id": artifact_id,
            }
        },
    }


def _enable_optimizations(config: dict[str, Any], *, alpha: bool) -> None:
    stage = config["stages"]["data_extraction"]
    stage["doc_filter"] = {
        "enabled": True,
        "filter_type": "schema_relevance",
        "target_recall": 0.95,
        "enable_calibrate": True,
        "embedding_model": "local-hash-embedding",
        "embeddings_cache_dir": str((Path("outputs") / "deepseek_query_curation" / "_embedding_cache")),
        "threshold": 0.585,
        "use_source_table_metadata": True,
        "source_table_metadata_only": False,
        "source_table_keep_unknown": True,
    }
    stage["table_assignment_cache"] = {
        "enabled": True,
        "source_table_metadata": True,
        "general_schema": True,
    }
    stage["proxy_runtime"] = {
        "enabled": True,
        "predicate_proxy_mode": "pretrained",
        "target_recall": 0.95,
        "use_embedding_proxies": False,
        "use_learned_proxies": True,
        "use_finetuned_learned_proxies": True,
        "finetuned_model": "heuristic",
        "finetuned_epochs": 0,
        "proxy_threshold": 0.51,
        "allow_embedding_fallback": False,
        "use_join_resolution": True,
        "bidirectional_join_resolution": True,
        "join_order_strategy": "selective_first",
        "join_empty_short_circuit": True,
        "cross_query_extraction_cache": True,
        "cache_extract_full_table": True,
        "save_hard_negatives": False,
        "verbose": False,
    }
    if alpha:
        stage["alpha_allocation"] = {
            "enabled": True,
            "target_recall": 0.95,
            "alpha_grid": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            "answer_recall_calibration": True,
            "answer_recall_calibration_allow_over_budget": True,
            "answer_recall_calibration_global": True,
        }
    else:
        stage["alpha_allocation"] = {"enabled": False}


def write_configs(prepared: list[PreparedDataset]) -> list[dict[str, Any]]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for variant in VARIANTS:
        config = _base_config(prepared, variant)
        if variant == "noalpha":
            _enable_optimizations(config, alpha=False)
        elif variant == "alpha":
            _enable_optimizations(config, alpha=True)
        path = CONFIG_ROOT / f"deepseek_query_curation_v4_aligned_{variant}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        written.append(
            {
                "variant": variant,
                "config": str(path.relative_to(ROOT)),
                "artifact": config["runtime"]["artifact_id"],
            }
        )
    return written


def prepare() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    prepared = prepare_datasets()
    configs = write_configs(prepared)
    manifest = {
        "datasets": [
            {
                "source_dataset_id": item.source_dataset_id,
                "dataset_id": item.dataset_id,
                "slug": item.slug,
                "root": str(item.root.relative_to(ROOT)),
                "query_ids": item.query_ids,
                "train_count": item.train_count,
            }
            for item in prepared
        ],
        "configs": configs,
    }
    (REPORT_ROOT / "query_curation_experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _load_manifest() -> dict[str, Any]:
    path = REPORT_ROOT / "query_curation_experiment_manifest.json"
    if not path.exists():
        return prepare()
    return json.loads(path.read_text(encoding="utf-8"))


def run_variant(variant: str) -> None:
    _load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    manifest = _load_manifest()
    config_row = next(row for row in manifest["configs"] if row["variant"] == variant)
    config_path = ROOT / config_row["config"]
    usage_log = REPORT_ROOT / f"llm_usage_{variant}.jsonl"
    if usage_log.exists():
        usage_log.unlink()
    os.environ["REDD_LLM_USAGE_LOG"] = str(usage_log)
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    sys.path.insert(0, str(ROOT / "src"))
    from redd.runners import run_evaluation, run_extract

    status_path = REPORT_ROOT / "query_curation_run_status.json"
    status = []
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = []
    start = time.time()
    record = {
        "variant": variant,
        "config": config_row["config"],
        "artifact": config_row["artifact"],
        "usage_log": str(usage_log.relative_to(ROOT)),
        "status": "running",
        "started_at": start,
    }
    print(f"[run] {variant}", flush=True)
    try:
        with (LOG_ROOT / f"{variant}_extract.log").open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                run_extract(str(config_path), "demo")
        with (LOG_ROOT / f"{variant}_evaluate.log").open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                run_evaluation(str(config_path), "demo")
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = repr(exc)
        print(f"[error] {variant}: {exc!r}", flush=True)
    record["elapsed_sec"] = round(time.time() - start, 2)
    status.append(record)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def semantic_evaluate_variant(variant: str) -> None:
    _load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    manifest = _load_manifest()
    config_row = next(row for row in manifest["configs"] if row["variant"] == variant)
    config_path = ROOT / config_row["config"]
    usage_log = REPORT_ROOT / f"semantic_llm_usage_{variant}.jsonl"
    env = os.environ.copy()
    env["REDD_LLM_USAGE_LOG"] = str(usage_log)
    env["LITELLM_LOG"] = "ERROR"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src")
        if not existing_pythonpath
        else str(ROOT / "src") + os.pathsep + existing_pythonpath
    )
    with (LOG_ROOT / f"{variant}_semantic_evaluate.log").open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from redd.runners import run_evaluation; "
                f"run_evaluation({str(config_path)!r}, 'demo')",
            ],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=handle,
            check=True,
        )


def _usage_summary(path: Path) -> dict[str, Any]:
    calls = prompt = completion = total = 0
    providers: set[str] = set()
    response_models: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            usage = item.get("usage") or {}
            calls += 1
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
            total += int(usage.get("total_tokens") or 0)
            if item.get("provider"):
                providers.add(str(item["provider"]))
            if item.get("response_model"):
                response_models.add(str(item["response_model"]))
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "providers": sorted(providers),
        "response_models": sorted(response_models),
    }


def _query_summary(eval_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    query = payload.get("query_aware") or {}
    summary = query.get("summary") or {}
    cell = query.get("cell_recall") or {}
    answer = query.get("answer_recall") or {}
    semantic = query.get("semantic_cell_accuracy") or {}
    return {
        "query_id": query.get("query_id") or eval_path.name,
        "evaluated_docs": summary.get("evaluated_docs"),
        "relevant_docs": summary.get("relevant_docs"),
        "prediction_docs": summary.get("prediction_docs"),
        "cell_recall": cell.get("recall"),
        "cell_covered": cell.get("covered"),
        "cell_total": cell.get("total"),
        "answer_recall": answer.get("recall"),
        "answer_precision": answer.get("precision"),
        "answer_covered": answer.get("covered"),
        "answer_total": answer.get("total"),
        "semantic_cell_accuracy": semantic.get("accuracy"),
        "semantic_correct": semantic.get("correct"),
        "semantic_total": semantic.get("total"),
        "semantic_llm_judged": semantic.get("llm_judged"),
    }


def summarize() -> dict[str, Any]:
    manifest = _load_manifest()
    selected_meta: dict[tuple[str, str], dict[str, Any]] = {}
    allowed_queries: dict[str, set[str]] = {}
    for dataset in manifest["datasets"]:
        allowed_queries[dataset["dataset_id"]] = {str(query_id) for query_id in dataset.get("query_ids") or []}
        queries_path = ROOT / dataset["root"] / "metadata" / "queries.json"
        payload = json.loads(queries_path.read_text(encoding="utf-8"))
        for query in payload.get("queries") or []:
            selected_meta[(dataset["dataset_id"], query["query_id"])] = {
                "source_dataset_id": dataset["source_dataset_id"],
                "coverage_bucket": (query.get("meta") or {}).get("coverage_bucket"),
                "actual_doc_coverage": (query.get("meta") or {}).get("actual_doc_coverage"),
                "text_support": (query.get("meta") or {}).get("text_support") or {},
                "sql": query.get("sql"),
                "question": query.get("question"),
            }

    rows: list[dict[str, Any]] = []
    for config in manifest["configs"]:
        variant = config["variant"]
        artifact = config["artifact"]
        usage = _usage_summary(REPORT_ROOT / f"llm_usage_{variant}.jsonl")
        for dataset in manifest["datasets"]:
            artifact_dir = OUTPUT_ROOT / dataset["dataset_id"] / "data_extraction" / artifact
            for eval_path in sorted(artifact_dir.glob("eval_*.json")):
                query = _query_summary(eval_path)
                if str(query["query_id"]) not in allowed_queries.get(dataset["dataset_id"], set()):
                    continue
                meta = selected_meta.get((dataset["dataset_id"], str(query["query_id"])), {})
                rows.append(
                    {
                        "variant": variant,
                        "dataset_id": dataset["dataset_id"],
                        "source_dataset_id": dataset["source_dataset_id"],
                        "artifact": artifact,
                        **query,
                        "coverage_bucket": meta.get("coverage_bucket"),
                        "actual_doc_coverage": meta.get("actual_doc_coverage"),
                        "output_support": (meta.get("text_support") or {}).get("output_support"),
                        "required_support": (meta.get("text_support") or {}).get("required_support"),
                        "sql": meta.get("sql"),
                        "question": meta.get("question"),
                        "variant_calls": usage["calls"],
                        "variant_total_tokens": usage["total_tokens"],
                        "variant_response_models": ",".join(usage["response_models"]),
                    }
                )
    df = pd.DataFrame(rows)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_ROOT / "query_curation_summary.csv", index=False)
    summary = {
        "rows": rows,
        "usage": {
            variant: _usage_summary(REPORT_ROOT / f"llm_usage_{variant}.jsonl")
            for variant in VARIANTS
        },
    }
    (REPORT_ROOT / "query_curation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if not df.empty:
        aggregate = (
            df.groupby("variant")
            .agg(
                queries=("query_id", "count"),
                semantic_correct=("semantic_correct", "sum"),
                semantic_total=("semantic_total", "sum"),
                answer_covered=("answer_covered", "sum"),
                answer_total=("answer_total", "sum"),
                calls=("variant_calls", "max"),
                tokens=("variant_total_tokens", "max"),
            )
            .reset_index()
        )
        aggregate["semantic_cell_accuracy"] = aggregate["semantic_correct"] / aggregate["semantic_total"]
        aggregate["answer_recall"] = aggregate["answer_covered"] / aggregate["answer_total"]
        aggregate.to_csv(REPORT_ROOT / "query_curation_aggregate.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="append", choices=VARIANTS)
    parser.add_argument("--semantic-evaluate", action="append", choices=VARIANTS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if not (args.prepare or args.run or args.semantic_evaluate or args.summarize):
        args.prepare = True
    if args.prepare:
        prepare()
    for variant in args.run or []:
        run_variant(variant)
    for variant in args.semantic_evaluate or []:
        semantic_evaluate_variant(variant)
    if args.summarize:
        summarize()


if __name__ == "__main__":
    main()
