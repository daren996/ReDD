from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DERIVED_ROOT = ROOT / "dataset" / "derived" / "deepseek_ablation_multi"
CONFIG_ROOT = ROOT / "configs" / "examples"
OUTPUT_ROOT = ROOT / "outputs" / "deepseek_ablation_multi"
REPORT_ROOT = OUTPUT_ROOT / "reports"
LOG_ROOT = REPORT_ROOT / "command_logs"


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    parent_id: str
    source_root: str
    table_id: str
    predicate_column: str
    predicate_name: str
    predicate_value: str
    output_columns: tuple[str, str]
    output_names: tuple[str, str]
    output_questions: tuple[str, str]
    positive_doc_id: str


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        slug="spider_soccer_1",
        parent_id="spider.soccer_1.default_task",
        source_root="dataset/derived/spider.soccer_1.default_task",
        table_id="player_attributes",
        predicate_column="player_attributes.defensive_work_rate",
        predicate_name="defensive_work_rate",
        predicate_value="high",
        output_columns=(
            "player_attributes.preferred_foot",
            "player_attributes.player_name",
        ),
        output_names=("preferred_foot", "player_name"),
        output_questions=(
            "Which preferred foot is recorded for the player with high defensive work rate?",
            "Which player has the high defensive work rate record?",
        ),
        positive_doc_id="200-0",
    ),
    DatasetSpec(
        slug="spider_wine_1",
        parent_id="spider.wine_1.wine_appellations",
        source_root="dataset/derived/spider.wine_1.wine_appellations",
        table_id="wines",
        predicate_column="wines.appellation",
        predicate_name="appellation",
        predicate_value="Santa Lucia Highlands",
        output_columns=("wines.wine_name", "wines.region"),
        output_names=("wine_name", "region"),
        output_questions=(
            "Which wine name is from the Santa Lucia Highlands appellation?",
            "Which region is recorded for the Santa Lucia Highlands appellation wine?",
        ),
        positive_doc_id="289-0",
    ),
    DatasetSpec(
        slug="bird_student_club",
        parent_id="bird.student_club.default_task",
        source_root="dataset/derived/bird.student_club.default_task",
        table_id="zip_code",
        predicate_column="zip_code.short_state",
        predicate_name="short_state",
        predicate_value="CT",
        output_columns=("zip_code.state", "zip_code.county"),
        output_names=("state", "county"),
        output_questions=(
            "Which state uses the short state CT?",
            "Which county is recorded for the CT zip code row?",
        ),
        positive_doc_id="422-0",
    ),
    DatasetSpec(
        slug="bird_debit_card",
        parent_id="bird.debit_card_specializing.default_task",
        source_root="dataset/derived/bird.debit_card_specializing.default_task",
        table_id="transactions_1k",
        predicate_column="transactions_1k.gas_station_id",
        predicate_name="gas_station_id",
        predicate_value="1119",
        output_columns=("transactions_1k.amount", "transactions_1k.customer_id"),
        output_names=("amount", "customer_id"),
        output_questions=(
            "What amount was recorded for the transaction at gas station 1119?",
            "Which customer id was recorded for the transaction at gas station 1119?",
        ),
        positive_doc_id="184-0",
    ),
    DatasetSpec(
        slug="bird_california_schools",
        parent_id="bird.california_schools.default_task",
        source_root="dataset/derived/bird.california_schools.default_task",
        table_id="frpm",
        predicate_column="frpm.educational_option_type",
        predicate_name="educational_option_type",
        predicate_value="Alternative School of Choice",
        output_columns=("frpm.district_type", "frpm.academic_year"),
        output_names=("district_type", "academic_year"),
        output_questions=(
            "Which district type is recorded for the Alternative School of Choice row?",
            "Which academic year is recorded for the Alternative School of Choice row?",
        ),
        positive_doc_id="1351-0",
    ),
]


VARIANT_ORDER = [
    "baseline",
    "table_metadata_cache",
    "doc_filter_metadata",
    "predicate_proxy",
    "full_stack_extraction_cache",
]

VARIANT_FEATURES = {
    "baseline": "none",
    "table_metadata_cache": "table metadata + cross-query table cache",
    "doc_filter_metadata": "metadata doc filter",
    "predicate_proxy": "heuristic predicate proxy",
    "full_stack_extraction_cache": (
        "doc filter + table metadata/cache + predicate proxy + extraction cache"
    ),
}


def _load_dotenv() -> None:
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _norm(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _value_variants(value: Any) -> list[str]:
    text = _norm(value)
    variants = {text}
    if text.endswith(".0"):
        variants.add(text[:-2])
    try:
        number = float(text)
        if number.is_integer():
            variants.add(str(int(number)))
    except Exception:
        pass
    return [variant for variant in variants if variant]


def _contains_exact(text: str, value: Any) -> bool:
    for variant in _value_variants(value):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(variant) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def _doc_table_map(ground_truth: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for doc_id, group in ground_truth.groupby("doc_id"):
        if not group.empty:
            mapping[str(doc_id)] = str(group["table_id"].iloc[0])
    return mapping


def _doc_value(
    ground_truth: pd.DataFrame,
    doc_id: str,
    column_id: str,
) -> str:
    rows = ground_truth[
        (ground_truth["doc_id"].astype(str) == doc_id)
        & (ground_truth["column_id"].astype(str) == column_id)
    ]
    if rows.empty:
        return ""
    return _norm(rows["value"].iloc[0])


def _select_negative_docs(
    docs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    spec: DatasetSpec,
    *,
    limit: int = 5,
) -> list[str]:
    text_by_doc = dict(zip(docs["doc_id"].astype(str), docs["doc_text"].astype(str)))
    selected: list[str] = []
    rows = ground_truth[
        (ground_truth["table_id"].astype(str) == spec.table_id)
        & (ground_truth["column_id"].astype(str) == spec.predicate_column)
    ].copy()
    for _, row in rows.sort_values("doc_id").iterrows():
        doc_id = str(row["doc_id"])
        value = _norm(row["value"])
        if doc_id == spec.positive_doc_id or value == spec.predicate_value:
            continue
        if doc_id not in text_by_doc:
            continue
        if _contains_exact(text_by_doc[doc_id], value):
            selected.append(doc_id)
        if len(selected) >= limit:
            return selected
    raise RuntimeError(f"{spec.slug}: only found {len(selected)} negative docs")


def _select_non_target_docs(
    docs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    spec: DatasetSpec,
    *,
    exclude_doc_ids: set[str],
    limit: int,
) -> list[str]:
    table_by_doc = _doc_table_map(ground_truth)
    selected: list[str] = []
    for doc_id in docs["doc_id"].astype(str).tolist():
        if doc_id in exclude_doc_ids:
            continue
        if table_by_doc.get(doc_id) == spec.table_id:
            continue
        selected.append(doc_id)
        if len(selected) >= limit:
            return selected
    raise RuntimeError(f"{spec.slug}: only found {len(selected)} non-target docs")


def _select_train_docs(
    docs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    exclude_doc_ids: set[str],
    limit: int = 3,
) -> list[str]:
    table_by_doc = _doc_table_map(ground_truth)
    selected: list[str] = []
    seen_tables: set[str] = set()
    doc_ids = docs["doc_id"].astype(str).tolist()
    for doc_id in doc_ids:
        if doc_id in exclude_doc_ids:
            continue
        table_id = table_by_doc.get(doc_id, "")
        if table_id in seen_tables:
            continue
        selected.append(doc_id)
        seen_tables.add(table_id)
        if len(selected) >= limit:
            return selected
    for doc_id in doc_ids:
        if doc_id in exclude_doc_ids or doc_id in selected:
            continue
        selected.append(doc_id)
        if len(selected) >= limit:
            return selected
    raise RuntimeError(f"only found {len(selected)} train docs")


def _rewrite_source_table(
    docs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    table_by_doc = _doc_table_map(ground_truth)
    out = docs.copy()
    out["source_table"] = out["doc_id"].astype(str).map(table_by_doc).fillna(out["source_table"])
    return out


def _query_payload(dataset_id: str, spec: DatasetSpec) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    for index, (column, name, question) in enumerate(
        zip(spec.output_columns, spec.output_names, spec.output_questions),
        start=1,
    ):
        query_id = f"Q_{spec.slug}_{index}"
        queries.append(
            {
                "query_id": query_id,
                "question": question,
                "sql": (
                    f"SELECT {name} FROM {spec.table_id} "
                    f"WHERE {spec.predicate_name} = '{spec.predicate_value}';"
                ),
                "required_tables": [spec.table_id],
                "required_columns": [spec.predicate_column, column],
                "output_columns": [column],
                "tags": ["ablation", "predicate", "multi_dataset"],
                "difficulty": "controlled",
            }
        )
    return {
        "schema_version": "redd.queries.v1",
        "dataset_id": dataset_id,
        "queries": queries,
    }


def prepare_datasets() -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in DATASET_SPECS:
        source_root = ROOT / spec.source_root
        docs = pd.read_parquet(source_root / "data" / "documents.parquet")
        ground_truth = pd.read_parquet(source_root / "data" / "ground_truth.parquet")
        text_by_doc = dict(zip(docs["doc_id"].astype(str), docs["doc_text"].astype(str)))

        if spec.positive_doc_id not in text_by_doc:
            raise RuntimeError(f"{spec.slug}: positive doc missing")
        for column in (spec.predicate_column, *spec.output_columns):
            value = (
                spec.predicate_value
                if column == spec.predicate_column
                else _doc_value(ground_truth, spec.positive_doc_id, column)
            )
            if not _contains_exact(text_by_doc[spec.positive_doc_id], value):
                raise RuntimeError(
                    f"{spec.slug}: positive doc {spec.positive_doc_id} "
                    f"does not contain {column}={value!r}"
                )

        negative_docs = _select_negative_docs(docs, ground_truth, spec, limit=5)
        reserved = {spec.positive_doc_id, *negative_docs}
        non_target_docs = _select_non_target_docs(
            docs,
            ground_truth,
            spec,
            exclude_doc_ids=reserved,
            limit=9,
        )
        train_docs = _select_train_docs(
            docs,
            ground_truth,
            exclude_doc_ids={*reserved, *non_target_docs},
            limit=3,
        )
        ordered_doc_ids = [*train_docs, spec.positive_doc_id, *negative_docs, *non_target_docs]

        dataset_id = f"deepseek_ablation_multi.{spec.slug}"
        dest = DERIVED_ROOT / spec.slug
        if dest.exists():
            shutil.rmtree(dest)
        (dest / "data").mkdir(parents=True, exist_ok=True)
        (dest / "metadata").mkdir(parents=True, exist_ok=True)

        docs_out = docs[docs["doc_id"].astype(str).isin(ordered_doc_ids)].copy()
        docs_out["_order"] = pd.Categorical(
            docs_out["doc_id"].astype(str),
            categories=ordered_doc_ids,
            ordered=True,
        )
        docs_out = docs_out.sort_values("_order").drop(columns=["_order"])
        docs_out["dataset_id"] = dataset_id
        docs_out["split"] = ["train"] * len(train_docs) + ["test"] * (len(ordered_doc_ids) - len(train_docs))
        docs_out = _rewrite_source_table(docs_out, ground_truth)

        gt_out = ground_truth[ground_truth["doc_id"].astype(str).isin(ordered_doc_ids)].copy()
        gt_out["dataset_id"] = dataset_id

        docs_out.to_parquet(dest / "data" / "documents.parquet", index=False)
        gt_out.to_parquet(dest / "data" / "ground_truth.parquet", index=False)

        schema = json.loads((source_root / "metadata" / "schema.json").read_text(encoding="utf-8"))
        schema["dataset_id"] = dataset_id
        (dest / "metadata" / "schema.json").write_text(
            json.dumps(schema, indent=2),
            encoding="utf-8",
        )
        queries = _query_payload(dataset_id, spec)
        (dest / "metadata" / "queries.json").write_text(
            json.dumps(queries, indent=2),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "redd.manifest.v1",
            "dataset_id": dataset_id,
            "kind": "derived",
            "version": "0.1.0",
            "parents": [spec.parent_id],
            "paths": {
                "documents": "data/documents.parquet",
                "ground_truth": "data/ground_truth.parquet",
                "schema": "metadata/schema.json",
                "queries": "metadata/queries.json",
            },
        }
        (dest / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

        prepared.append(
            {
                "slug": spec.slug,
                "dataset_id": dataset_id,
                "root": str(dest.relative_to(ROOT)),
                "query_ids": [f"Q_{spec.slug}_1", f"Q_{spec.slug}_2"],
                "positive_doc_id": spec.positive_doc_id,
                "negative_docs": negative_docs,
                "non_target_docs": non_target_docs,
                "train_docs": train_docs,
            }
        )
    return prepared


def _base_config(dataset: dict[str, Any], variant: str) -> dict[str, Any]:
    artifact_id = f"deepseek-ablation-multi-{dataset['slug']}-{variant}-v1"
    dataset_id = dataset["dataset_id"]
    return {
        "config_version": "2.1.1",
        "project": {"name": f"deepseek-ablation-multi-{dataset['slug']}-{variant}", "seed": 42},
        "runtime": {
            "output_dir": "outputs/deepseek_ablation_multi",
            "log_dir": "outputs/logs",
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
                "max_tokens": 256,
            },
            "embedding": {
                "provider": "none",
                "model": "none",
                "enabled": False,
                "api_key_env": None,
            },
        },
        "datasets": {
            dataset_id: {
                "loader": "hf_manifest",
                "root": dataset["root"],
                "query_ids": dataset["query_ids"],
                "loader_options": {"manifest": "manifest.yaml"},
                "split": {"train_count": len(dataset["train_docs"])},
            }
        },
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
                "datasets": [dataset_id],
                "stages": ["data_extraction"],
                "artifact_id": artifact_id,
            }
        },
    }


def _with_doc_filter(stage: dict[str, Any]) -> None:
    stage["doc_filter"] = {
        "enabled": True,
        "filter_type": "schema_relevance",
        "target_recall": 1.0,
        "use_source_table_metadata": True,
        "source_table_metadata_only": True,
        "source_table_keep_unknown": True,
    }


def _with_table_cache(stage: dict[str, Any]) -> None:
    stage["table_assignment_cache"] = {
        "enabled": True,
        "source_table_metadata": True,
        "general_schema": False,
    }


def _with_proxy(stage: dict[str, Any], *, extraction_cache: bool = False) -> None:
    proxy = {
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
        "use_join_resolution": False,
        "save_hard_negatives": False,
        "verbose": True,
    }
    if extraction_cache:
        proxy["cross_query_extraction_cache"] = True
        proxy["cache_extract_full_table"] = True
    stage["proxy_runtime"] = proxy


def make_config(dataset: dict[str, Any], variant: str) -> dict[str, Any]:
    config = _base_config(dataset, variant)
    stage = config["stages"]["data_extraction"]
    if variant == "table_metadata_cache":
        _with_table_cache(stage)
    elif variant == "doc_filter_metadata":
        _with_doc_filter(stage)
    elif variant == "predicate_proxy":
        _with_proxy(stage)
    elif variant == "full_stack_extraction_cache":
        _with_doc_filter(stage)
        _with_table_cache(stage)
        _with_proxy(stage, extraction_cache=True)
        stage["alpha_allocation"] = {"enabled": False}
    elif variant != "baseline":
        raise ValueError(f"unknown variant: {variant}")
    return config


def write_configs(prepared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    for dataset in prepared:
        for variant in VARIANT_ORDER:
            path = CONFIG_ROOT / f"deepseek_ablation_multi_{dataset['slug']}_{variant}.yaml"
            config = make_config(dataset, variant)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            written.append(
                {
                    "slug": dataset["slug"],
                    "dataset_id": dataset["dataset_id"],
                    "variant": variant,
                    "config": str(path.relative_to(ROOT)),
                    "artifact": config["runtime"]["artifact_id"],
                }
            )
    return written


def prepare() -> list[dict[str, Any]]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    prepared = prepare_datasets()
    configs = write_configs(prepared)
    manifest = {"datasets": prepared, "configs": configs}
    (REPORT_ROOT / "ablation_multi_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return configs


def _load_manifest() -> dict[str, Any]:
    path = REPORT_ROOT / "ablation_multi_manifest.json"
    if not path.exists():
        prepare()
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiments(*, variants: set[str] | None = None, datasets: set[str] | None = None) -> None:
    _load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    sys.path.insert(0, str(ROOT / "src"))
    from redd.runners import run_evaluation, run_extract

    manifest = _load_manifest()
    status_path = REPORT_ROOT / "ablation_multi_run_status.json"
    status: list[dict[str, Any]] = []
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = []
    done = {(row.get("slug"), row.get("variant")) for row in status if row.get("status") == "ok"}

    for row in manifest["configs"]:
        slug = row["slug"]
        variant = row["variant"]
        if variants and variant not in variants:
            continue
        if datasets and slug not in datasets:
            continue
        if (slug, variant) in done:
            continue
        config_path = str(ROOT / row["config"])
        usage_log = REPORT_ROOT / f"llm_usage_{slug}_{variant}.jsonl"
        if usage_log.exists():
            usage_log.unlink()
        os.environ["REDD_LLM_USAGE_LOG"] = str(usage_log)
        extract_log = LOG_ROOT / f"{slug}_{variant}_extract.log"
        eval_log = LOG_ROOT / f"{slug}_{variant}_evaluate.log"
        start = time.time()
        record = {
            "slug": slug,
            "dataset_id": row["dataset_id"],
            "variant": variant,
            "config": row["config"],
            "artifact": row["artifact"],
            "usage_log": str(usage_log.relative_to(ROOT)),
            "status": "running",
            "started_at": start,
        }
        print(f"[run] {slug} / {variant}", flush=True)
        try:
            with extract_log.open("w", encoding="utf-8") as handle:
                with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                    run_extract(config_path, "demo")
            with eval_log.open("w", encoding="utf-8") as handle:
                with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                    run_evaluation(config_path, "demo")
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = repr(exc)
            print(f"[error] {slug} / {variant}: {exc!r}", flush=True)
        record["elapsed_sec"] = round(time.time() - start, 2)
        status.append(record)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_semantic_evaluations(*, variants: set[str] | None = None, datasets: set[str] | None = None) -> None:
    _load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    sys.path.insert(0, str(ROOT / "src"))

    manifest = _load_manifest()
    status_path = REPORT_ROOT / "ablation_multi_semantic_eval_status.json"
    status: list[dict[str, Any]] = []
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = []
    done = {(row.get("slug"), row.get("variant")) for row in status if row.get("status") == "ok"}

    for row in manifest["configs"]:
        slug = row["slug"]
        variant = row["variant"]
        if variants and variant not in variants:
            continue
        if datasets and slug not in datasets:
            continue
        if (slug, variant) in done:
            continue
        config_path = str(ROOT / row["config"])
        usage_log = REPORT_ROOT / f"semantic_llm_usage_{slug}_{variant}.jsonl"
        os.environ["REDD_LLM_USAGE_LOG"] = str(usage_log)
        eval_log = LOG_ROOT / f"{slug}_{variant}_semantic_evaluate.log"
        start = time.time()
        record = {
            "slug": slug,
            "dataset_id": row["dataset_id"],
            "variant": variant,
            "config": row["config"],
            "artifact": row["artifact"],
            "usage_log": str(usage_log.relative_to(ROOT)),
            "status": "running",
            "started_at": start,
        }
        print(f"[semantic-eval] {slug} / {variant}", flush=True)
        try:
            with eval_log.open("w", encoding="utf-8") as handle:
                env = os.environ.copy()
                env["REDD_LLM_USAGE_LOG"] = str(usage_log)
                env["LITELLM_LOG"] = "ERROR"
                existing_pythonpath = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    str(ROOT / "src")
                    if not existing_pythonpath
                    else str(ROOT / "src") + os.pathsep + existing_pythonpath
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "from redd.runners import run_evaluation; "
                            f"run_evaluation({config_path!r}, 'demo')"
                        ),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=handle,
                    check=True,
                )
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = repr(exc)
            print(f"[error] semantic {slug} / {variant}: {exc!r}", flush=True)
        record["elapsed_sec"] = round(time.time() - start, 2)
        status.append(record)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _usage_summary(path: Path) -> dict[str, Any]:
    calls = 0
    prompt = completion = total = 0
    providers: set[str] = set()
    request_models: set[str] = set()
    response_models: set[str] = set()
    if not path.exists():
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "providers": [],
            "request_models": [],
            "response_models": [],
        }
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
        if item.get("request_model"):
            request_models.add(str(item["request_model"]))
        if item.get("response_model"):
            response_models.add(str(item["response_model"]))
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "providers": sorted(providers),
        "request_models": sorted(request_models),
        "response_models": sorted(response_models),
    }


def _query_eval_summary(artifact_dir: Path) -> dict[str, Any]:
    table_cov = table_total = 0
    cell_cov = cell_total = 0
    answer_cov = answer_total = 0
    precision_values: list[float] = []
    legacy_attr_true = legacy_attr_total = 0
    legacy_final_true = legacy_final_total = 0
    query_semantic_true = query_semantic_total = 0
    query_semantic_llm_judged = 0
    semantic_attr_true = semantic_attr_total = 0
    semantic_final_true = semantic_final_total = 0
    semantic_llm_judged = 0
    evaluated_docs = 0
    for path in artifact_dir.glob("eval_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        query = payload.get("query_aware") or {}
        table = query.get("table_assignment") or {}
        cells = query.get("cell_recall") or {}
        answer = query.get("answer_recall") or {}
        summary = query.get("summary") or {}
        table_cov += int(table.get("covered") or 0)
        table_total += int(table.get("total") or 0)
        cell_cov += int(cells.get("covered") or 0)
        cell_total += int(cells.get("total") or 0)
        answer_cov += int(answer.get("covered") or 0)
        answer_total += int(answer.get("total") or 0)
        if answer.get("precision") is not None:
            precision_values.append(float(answer.get("precision")))
        evaluated_docs += int(summary.get("evaluated_docs") or 0)
        query_semantic = query.get("semantic_cell_accuracy") or {}
        query_semantic_true += int(query_semantic.get("correct") or 0)
        query_semantic_total += int(query_semantic.get("total") or 0)
        query_semantic_llm_judged += int(query_semantic.get("llm_judged") or 0)
        legacy = payload.get("legacy") or {}
        for doc in (legacy.get("doc_stats") or {}).values():
            attrs = doc.get("attr") or {}
            legacy_attr_true += sum(1 for value in attrs.values() if value)
            legacy_attr_total += len(attrs)
            legacy_final_true += 1 if doc.get("final") else 0
            legacy_final_total += 1
        semantic = payload.get("full_table_semantic") or payload.get("semantic") or {}
        semantic_summary = semantic.get("summary") or {}
        semantic_attr_true += int(semantic_summary.get("attr_correct") or 0)
        semantic_attr_total += int(semantic_summary.get("attr_total") or 0)
        semantic_final_true += int(semantic_summary.get("doc_final_true") or 0)
        semantic_final_total += int(semantic_summary.get("doc_total") or 0)
        semantic_llm_judged += int(semantic_summary.get("llm_judged") or 0)
    return {
        "table_recall": table_cov / table_total if table_total else None,
        "cell_recall": cell_cov / cell_total if cell_total else None,
        "answer_recall": answer_cov / answer_total if answer_total else None,
        "answer_precision": (
            sum(precision_values) / len(precision_values) if precision_values else None
        ),
        "legacy_attr_accuracy": (
            legacy_attr_true / legacy_attr_total if legacy_attr_total else None
        ),
        "legacy_attr_correct": legacy_attr_true,
        "legacy_attr_total": legacy_attr_total,
        "legacy_doc_final_accuracy": (
            legacy_final_true / legacy_final_total if legacy_final_total else None
        ),
        "legacy_doc_final_true": legacy_final_true,
        "legacy_doc_total": legacy_final_total,
        "query_semantic_cell_accuracy": (
            query_semantic_true / query_semantic_total if query_semantic_total else None
        ),
        "query_semantic_cell_correct": query_semantic_true,
        "query_semantic_cell_total": query_semantic_total,
        "query_semantic_llm_judged": query_semantic_llm_judged,
        "semantic_attr_accuracy": (
            semantic_attr_true / semantic_attr_total if semantic_attr_total else None
        ),
        "semantic_attr_correct": semantic_attr_true,
        "semantic_attr_total": semantic_attr_total,
        "semantic_doc_final_accuracy": (
            semantic_final_true / semantic_final_total if semantic_final_total else None
        ),
        "semantic_doc_final_true": semantic_final_true,
        "semantic_doc_total": semantic_final_total,
        "semantic_llm_judged": semantic_llm_judged,
        "full_table_semantic_attr_accuracy": (
            semantic_attr_true / semantic_attr_total if semantic_attr_total else None
        ),
        "full_table_semantic_attr_correct": semantic_attr_true,
        "full_table_semantic_attr_total": semantic_attr_total,
        "full_table_semantic_llm_judged": semantic_llm_judged,
        "evaluated_docs_across_queries": evaluated_docs,
    }


def _doc_filter_summary(artifact_dir: Path) -> dict[str, int]:
    excluded = kept = 0
    for path in (artifact_dir / "doc_filter").glob("doc_filter_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded += len(payload.get("excluded_doc_ids") or [])
        kept += len(payload.get("kept_doc_ids") or [])
    return {"doc_filter_excluded": excluded, "doc_filter_kept": kept}


def _table_cache_summary(artifact_dir: Path) -> dict[str, int]:
    path = artifact_dir / "table_assignment_cache.json"
    if not path.exists():
        return {
            "table_saved": 0,
            "table_metadata_hits": 0,
            "table_cache_hits": 0,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    totals = payload.get("totals") or {}
    metadata_hits = int(totals.get("source_table_metadata_hits") or 0)
    cache_hits = int(totals.get("cache_hits") or 0)
    return {
        "table_saved": metadata_hits + cache_hits,
        "table_metadata_hits": metadata_hits,
        "table_cache_hits": cache_hits,
    }


def _proxy_summary(artifact_dir: Path) -> dict[str, Any]:
    candidates = passed = rejected = extracted = cache_hits = 0
    recalls: list[float] = []
    for path in artifact_dir.glob("*_proxy_decisions.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for table_payload in payload.values():
            all_doc_ids = table_payload.get("all_doc_ids") or []
            passed_doc_ids = table_payload.get("passed_doc_ids") or []
            extracted_doc_ids = table_payload.get("extracted_doc_ids") or []
            cache_hit_doc_ids = table_payload.get("cache_hit_doc_ids") or []
            proxy_rejected = table_payload.get("proxy_rejected_doc_ids") or {}
            candidates += len(all_doc_ids)
            passed += len(passed_doc_ids)
            extracted += len(extracted_doc_ids)
            cache_hits += len(cache_hit_doc_ids)
            rejected += sum(len(ids or []) for ids in proxy_rejected.values())
            for recall_payload in (table_payload.get("proxy_recalls") or {}).values():
                if recall_payload.get("recall") is not None:
                    recalls.append(float(recall_payload["recall"]))
    return {
        "proxy_candidates": candidates,
        "proxy_passed": passed,
        "proxy_rejected": rejected,
        "proxy_extracted": extracted,
        "proxy_cache_hits": cache_hits,
        "proxy_recall": sum(recalls) / len(recalls) if recalls else None,
    }


def _format_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}%"


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def summarize() -> dict[str, Any]:
    manifest = _load_manifest()
    rows: list[dict[str, Any]] = []
    for row in manifest["configs"]:
        slug = row["slug"]
        dataset_id = row["dataset_id"]
        variant = row["variant"]
        artifact = row["artifact"]
        artifact_dir = OUTPUT_ROOT / dataset_id / "data_extraction" / artifact
        usage = _usage_summary(REPORT_ROOT / f"llm_usage_{slug}_{variant}.jsonl")
        summary = {
            "dataset": slug,
            "dataset_id": dataset_id,
            "variant": variant,
            "features": VARIANT_FEATURES[variant],
            "artifact": artifact,
            **usage,
            **_query_eval_summary(artifact_dir),
            **_doc_filter_summary(artifact_dir),
            **_table_cache_summary(artifact_dir),
            **_proxy_summary(artifact_dir),
        }
        rows.append(summary)

    baselines = {
        row["dataset"]: row for row in rows if row["variant"] == "baseline"
    }
    for row in rows:
        baseline = baselines.get(row["dataset"]) or {}
        base_calls = baseline.get("calls") or 0
        base_tokens = baseline.get("total_tokens") or 0
        row["call_reduction_pct"] = (
            (base_calls - row["calls"]) / base_calls * 100 if base_calls else None
        )
        row["token_reduction_pct"] = (
            (base_tokens - row["total_tokens"]) / base_tokens * 100 if base_tokens else None
        )

    report = {"rows": rows}
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "ablation_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(REPORT_ROOT / "ablation_summary.csv", index=False)

    lines = [
        "# DeepSeek Multi-Dataset Data Extraction Ablation",
        "",
        "Controlled variants use only test rows whose predicate and query-output values appear exactly in the document text. Query semantic cell accuracy scores only SQL-required answer-provenance cells; full-table semantic attribute accuracy still scores every ground-truth column and is included as a stricter data-population diagnostic.",
        "",
        "| Dataset | Variant | DeepSeek Calls | Call Red. | Tokens | Token Red. | Query Answer R/P | Query Cell R | Query Semantic Cell Acc | Full-table Strict Attr Acc | Full-table Semantic Attr Acc | Strict Final Doc Acc | Query/Full Semantic LLM Judged | DocFilter Excl/Keep | Table Saved | Proxy Pass/Reject/Cache | Provider |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for slug in [spec.slug for spec in DATASET_SPECS]:
        for variant in VARIANT_ORDER:
            row = next(r for r in rows if r["dataset"] == slug and r["variant"] == variant)
            provider = ",".join(row.get("providers") or [])
            lines.append(
                "| {dataset} | {variant} | {calls} | {call_red} | {tokens} | {token_red} | {ar}/{ap} | {cell} | {query_semantic} | {legacy_attr} | {semantic_attr} | {legacy_final} | {query_semantic_judged}/{semantic_judged} | {df_ex}/{df_keep} | {table_saved} | {pp}/{pr}/{pc} | {provider} |".format(
                    dataset=slug,
                    variant=variant,
                    calls=row["calls"],
                    call_red=_format_pct(row["call_reduction_pct"]),
                    tokens=row["total_tokens"],
                    token_red=_format_pct(row["token_reduction_pct"]),
                    ar=_format_float(row["answer_recall"]),
                    ap=_format_float(row["answer_precision"]),
                    cell=_format_float(row["cell_recall"]),
                    query_semantic=_format_float(row["query_semantic_cell_accuracy"]),
                    legacy_attr=_format_float(row["legacy_attr_accuracy"]),
                    semantic_attr=_format_float(row["full_table_semantic_attr_accuracy"]),
                    legacy_final=_format_float(row["legacy_doc_final_accuracy"]),
                    query_semantic_judged=row["query_semantic_llm_judged"],
                    semantic_judged=row["full_table_semantic_llm_judged"],
                    df_ex=row["doc_filter_excluded"],
                    df_keep=row["doc_filter_kept"],
                    table_saved=row["table_saved"],
                    pp=row["proxy_passed"],
                    pr=row["proxy_rejected"],
                    pc=row["proxy_cache_hits"],
                    provider=provider,
                )
            )
        lines.append("|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Full-Stack Extraction Cache vs Baseline",
            "",
        "| Dataset | Baseline Calls | Full-Stack Calls | Call Red. | Baseline Tokens | Full-Stack Tokens | Token Red. | Full-Stack Query Answer R/P | Full-Stack Query Semantic Cell Acc | Full-Stack Full-table Strict Attr Acc | Full-Stack Full-table Semantic Attr Acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for slug in [spec.slug for spec in DATASET_SPECS]:
        base = next(r for r in rows if r["dataset"] == slug and r["variant"] == "baseline")
        full = next(
            r for r in rows
            if r["dataset"] == slug and r["variant"] == "full_stack_extraction_cache"
        )
        lines.append(
            "| {dataset} | {bc} | {fc} | {cr} | {bt} | {ft} | {tr} | {ar}/{ap} | {query_semantic} | {legacy} | {semantic} |".format(
                dataset=slug,
                bc=base["calls"],
                fc=full["calls"],
                cr=_format_pct(full["call_reduction_pct"]),
                bt=base["total_tokens"],
                ft=full["total_tokens"],
                tr=_format_pct(full["token_reduction_pct"]),
                ar=_format_float(full["answer_recall"]),
                ap=_format_float(full["answer_precision"]),
                query_semantic=_format_float(full["query_semantic_cell_accuracy"]),
                legacy=_format_float(full["legacy_attr_accuracy"]),
                semantic=_format_float(full["full_table_semantic_attr_accuracy"]),
            )
        )
    (REPORT_ROOT / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--semantic-evaluate", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--variant", action="append", choices=VARIANT_ORDER)
    parser.add_argument("--dataset", action="append", choices=[spec.slug for spec in DATASET_SPECS])
    args = parser.parse_args()

    if not (args.prepare or args.run or args.semantic_evaluate or args.summarize):
        args.prepare = args.run = args.summarize = True
    if args.prepare:
        prepare()
    if args.run:
        run_experiments(
            variants=set(args.variant) if args.variant else None,
            datasets=set(args.dataset) if args.dataset else None,
        )
    if args.semantic_evaluate:
        run_semantic_evaluations(
            variants=set(args.variant) if args.variant else None,
            datasets=set(args.dataset) if args.dataset else None,
        )
    if args.summarize:
        summarize()


if __name__ == "__main__":
    main()
