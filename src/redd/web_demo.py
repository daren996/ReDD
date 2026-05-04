"""Web-demo orchestration for ReDD."""

from __future__ import annotations

import html
import importlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .api import (
    DATA_EXTRACTION,
    PREPROCESSING,
    SCHEMA_REFINEMENT,
    DataPopulator,
    SchemaGenerator,
    run_pipeline,
)
from .config import (
    API_KEY_ENV_VARS,
    PROJECT_ROOT,
    DatasetRuntimeConfig,
    DatasetSplitConfig,
    ExperimentRuntime,
    ReDDConfig,
    StageName,
    load_experiment_runtime,
    load_redd_config,
    resolve_repo_path,
    select_experiment,
)
from .core.data_loader.data_loader_hf_manifest import build_default_query_records
from .core.utils.progress import progress_event_sink
from .diagnostics.dataset_consistency import (
    build_dataset_consistency_audit,
    write_dataset_consistency_audit,
)
from .model_catalog import get_model_catalog

DEFAULT_WEB_DEMO_CONFIG = "configs/demo/demo_datasets.yaml"
DEFAULT_WEB_DEMO_EXPERIMENT = "demo"
DEFAULT_DATASET_REGISTRY = "dataset/manifest.yaml"
DEFAULT_OUTPUTS_DIR = "outputs"
DATASET_SEARCH_GROUP_ORDER = (
    "example",
    "examples",
    "spider",
    "bird",
    "galois",
    "quest",
    "fda",
    "cuad",
)
WEB_DEMO_STAGES = [
    PREPROCESSING.value,
    SCHEMA_REFINEMENT.value,
    DATA_EXTRACTION.value,
]
WEB_DEMO_STAGE_SET = set(WEB_DEMO_STAGES)
WEB_DEMO_PAPERS = {
    "2025_ReDD.pdf": "papers/2025_ReDD.pdf",
    "2025_ReDD_TechReport.pdf": "papers/2025_ReDD_TechReport.pdf",
}
WEB_DEMO_API_KEY_ENV_VARS = tuple(dict.fromkeys(API_KEY_ENV_VARS.values()))
DEFAULT_WEB_DEMO_MODELS = {
    "llm": {
        "provider": "none",
        "model": "ground_truth",
        "api_key_env": None,
        "base_url": None,
        "structured_backend": "auto",
        "max_retries": 5,
        "wait_time": 10,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "local_model_path": None,
    },
    "embedding": {
        "provider": "local",
        "model": "local-hash-embedding",
        "enabled": True,
        "api_key_env": None,
        "base_url": None,
        "batch_size": 100,
        "storage_file": "embeddings.sqlite3",
    },
}

__all__ = [
    "collect_web_evaluation_summary",
    "create_web_demo_app",
    "run_web_demo",
    "serve_web_demo",
]


def run_web_demo(
    config_path: str | Path,
    experiment: str,
    *,
    config: dict[str, Any] | None = None,
    stages: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
    api_key: str | None = None,
    force_rerun: bool = False,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a package-backed workflow suitable for embedding in a web demo.

    This function intentionally does not start a server or depend on a web
    framework. A UI layer can call it and serialize the returned dictionary.
    """
    _emit_web_run_event(event_sink, "step_started", "materialize_config", "Loading runtime configuration.")
    if config is None:
        runtime, _ = load_experiment_runtime(config_path, experiment)
    else:
        runtime = select_experiment(ReDDConfig.model_validate(config), experiment)
    force_rerun = force_rerun or bool(getattr(runtime.runtime, "force_rerun", False))
    _emit_web_run_event(
        event_sink,
        "step_completed",
        "materialize_config",
        f"Experiment `{runtime.id}` loaded.",
        experiment=runtime.id,
    )

    requested_stages = list(stages or runtime.stage_order)
    requested_datasets = list(datasets or runtime.dataset_ids())
    requested_query_ids = list(query_ids or [])

    _emit_web_run_event(
        event_sink,
        "step_started",
        "resolve_datasets",
        "Resolving selected datasets and queries.",
        datasets=requested_datasets,
        query_ids=requested_query_ids,
    )
    runtime = _runtime_with_selected_datasets(runtime, requested_datasets, requested_query_ids)
    _emit_web_run_event(
        event_sink,
        "step_completed",
        "resolve_datasets",
        f"{len(requested_datasets)} dataset(s) ready.",
        datasets=requested_datasets,
        query_ids=requested_query_ids,
    )

    _emit_web_run_event(event_sink, "step_started", "prepare_runtime", "Preparing stage runtimes.")
    needs_schema_generator = any(
        stage in {PREPROCESSING.value, SCHEMA_REFINEMENT.value}
        for stage in requested_stages
    )
    needs_data_populator = DATA_EXTRACTION.value in requested_stages

    schema_generator = (
        SchemaGenerator(
            preprocessing_config=_stage_runtime_or_none(runtime, "preprocessing"),
            refinement_config=_stage_runtime_or_none(runtime, "schema_refinement"),
            api_key=api_key,
            configure_logging=False,
        )
        if needs_schema_generator
        else None
    )
    data_populator = (
        DataPopulator(
            _data_extraction_runtime(
                runtime,
                requested_query_ids,
                force_rerun=force_rerun,
            ),
            api_key=api_key,
            configure_logging=False,
        )
        if needs_data_populator
        else None
    )
    _emit_web_run_event(
        event_sink,
        "step_completed",
        "prepare_runtime",
        "Stage runtimes prepared.",
        stages=requested_stages,
    )

    if event_sink is None:
        result = run_pipeline(
            schema_generator=schema_generator,
            data_populator=data_populator,
            stages=requested_stages,
            datasets=requested_datasets,
        )
    else:
        result: dict[str, Any] = {}
        with progress_event_sink(event_sink):
            for stage in requested_stages:
                stage_id = str(stage)
                _emit_web_run_event(
                    event_sink,
                    "step_started",
                    stage_id,
                    f"Running {stage_id}.",
                )
                if stage_id == PREPROCESSING.value:
                    if schema_generator is None:
                        raise ValueError("PREPROCESSING requires `schema_generator=`.")
                    stage_result = schema_generator.preprocessing(datasets=requested_datasets)
                elif stage_id == SCHEMA_REFINEMENT.value:
                    if schema_generator is None:
                        raise ValueError("SCHEMA REFINEMENT requires `schema_generator=`.")
                    stage_result = schema_generator.schema_refinement(datasets=requested_datasets)
                elif stage_id == DATA_EXTRACTION.value:
                    if data_populator is None:
                        raise ValueError("DATA EXTRACTION requires `data_populator=`.")
                    stage_result = data_populator.data_extraction(
                        datasets=requested_datasets,
                        schema_generator=schema_generator,
                    )
                else:
                    raise ValueError(f"Unsupported stage `{stage_id}`.")

                result[stage_id] = stage_result
                _emit_web_run_event(
                    event_sink,
                    "step_completed",
                    stage_id,
                    f"{stage_id} completed.",
                    summary=_stage_result_summary(stage_result),
                )
                for metric in collect_web_optimization_metrics(
                    {
                        "experiment": runtime.id,
                        "datasets": requested_datasets,
                        "dataset_roots": _dataset_audit_inputs(runtime),
                        "query_ids": requested_query_ids,
                        "stages": requested_stages,
                        "result": result,
                    }
                ):
                    _emit_web_run_event(
                        event_sink,
                        "optimization_update",
                        "collect_optimization_metrics",
                        metric.get("message", ""),
                        optimization=metric,
                    )
    return {
        "experiment": runtime.id,
        "datasets": requested_datasets,
        "dataset_roots": _dataset_audit_inputs(runtime),
        "query_ids": requested_query_ids,
        "stages": requested_stages,
        "result": result,
    }


def _emit_web_run_event(
    event_sink: Callable[[dict[str, Any]], None] | None,
    event_type: str,
    step: str | None,
    message: str,
    **extra: Any,
) -> None:
    if event_sink is None:
        return
    event_sink(
        {
            "type": event_type,
            "step": step,
            "message": message,
            **extra,
        }
    )


def _stage_result_summary(value: Any) -> dict[str, Any]:
    return {
        "items": len(value) if isinstance(value, (list, dict)) else None,
        "kind": "list" if isinstance(value, list) else type(value).__name__,
    }


def _dataset_audit_inputs(runtime: ExperimentRuntime) -> dict[str, dict[str, Any]]:
    return {
        dataset_id: {
            "root": str(dataset.root),
            "query_ids": list(dataset.query_ids or []),
        }
        for dataset_id, dataset in runtime.datasets.items()
    }


def inspect_web_config(
    config_path: str | Path,
    experiment: str,
) -> dict[str, Any]:
    """Return config and selected experiment details for the web UI."""
    config, resolved_path = load_redd_config(config_path)
    experiment_id = _resolve_web_experiment_id(config, experiment)
    runtime = select_experiment(config, experiment_id)
    experiments = [
        {
            "id": experiment_id,
            "datasets": list(experiment_config.datasets),
            "stages": list(experiment_config.stages),
        }
        for experiment_id, experiment_config in config.experiments.items()
    ]
    datasets = [
        {
            "id": dataset_id,
            "root": str(dataset.root),
            "loader": dataset.loader,
            "query_ids": dataset.query_ids,
            "split": dataset.split.model_dump(mode="json", exclude_none=True),
            "loader_options": dataset.loader_options,
        }
        for dataset_id, dataset in runtime.datasets.items()
    ]
    return {
        "config_path": str(config_path),
        "resolved_config_path": str(resolved_path),
        "experiment": runtime.id,
        "project": config.project.model_dump(mode="json", exclude_none=True),
        "runtime": config.runtime.model_dump(mode="json", exclude_none=True),
        "models": _sanitize_models_for_web(config),
        "api_key_status": _api_key_status(config),
        "stage_configs": {
            stage_id: stage_config.model_dump(mode="json", exclude_none=True)
            for stage_id, stage_config in config.stages.items()
        },
        "experiments": experiments,
        "datasets": datasets,
        "dataset_ids": runtime.dataset_ids(),
        "stages": WEB_DEMO_STAGES,
        "default_stages": list(runtime.stage_order),
        "model_catalog": get_model_catalog(),
    }


def _resolve_web_experiment_id(config: ReDDConfig, experiment: str) -> str:
    requested_experiment = str(experiment or "").strip()
    if requested_experiment in config.experiments:
        return requested_experiment
    try:
        return next(iter(config.experiments))
    except StopIteration as exc:
        raise KeyError("No experiments found in config") from exc


def _load_web_dotenv() -> None:
    """Load local .env values for the web demo without exposing them to clients."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _sanitize_models_for_web(config: ReDDConfig) -> dict[str, Any]:
    models = config.models.model_dump(mode="json", exclude_none=False)
    for section in ("llm", "embedding"):
        model_config = models.get(section)
        if isinstance(model_config, dict):
            model_config.pop("api_key", None)
    return models


def _api_key_status(config: ReDDConfig | None = None) -> dict[str, Any]:
    model_sources: list[dict[str, Any]] = []
    configured = False

    if config is not None:
        model_pairs = (
            ("llm", config.models.llm),
            ("embedding", config.models.embedding),
        )
        for kind, model_config in model_pairs:
            if model_config is None or model_config.provider in {"local", "none"}:
                continue

            provider = str(model_config.provider)
            source = None
            if model_config.api_key:
                source = f"config.models.{kind}.api_key"
            elif model_config.api_key_env and os.getenv(model_config.api_key_env):
                source = model_config.api_key_env
            else:
                provider_env = API_KEY_ENV_VARS.get(provider)
                if provider_env and os.getenv(provider_env):
                    source = provider_env

            model_sources.append(
                {
                    "kind": kind,
                    "provider": provider,
                    "configured": source is not None,
                    "source": source,
                    "masked": "****" if source else None,
                }
            )
            configured = configured or source is not None

    env_sources = [
        {"name": env_var, "configured": bool(os.getenv(env_var)), "masked": "****"}
        for env_var in WEB_DEMO_API_KEY_ENV_VARS
    ]
    configured = configured or any(source["configured"] for source in env_sources)
    primary = next((source for source in model_sources if source["configured"]), None)
    if primary is None:
        primary = next((source for source in env_sources if source["configured"]), None)

    return {
        "configured": configured,
        "masked": "****" if configured else None,
        "source": primary.get("source") or primary.get("name") if primary else None,
        "models": model_sources,
        "environment": env_sources,
    }


def list_registry_datasets(
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> dict[str, Any]:
    registry, resolved_path = _load_dataset_registry(registry_path)
    datasets = [
        _dataset_summary(dataset_id, entry, resolved_path)
        for dataset_id, entry in sorted(
            (registry.get("datasets") or {}).items(),
            key=lambda item: _dataset_search_sort_key(item[0]),
        )
        if isinstance(entry, dict)
    ]
    return {
        "registry": str(resolved_path),
        "datasets": datasets,
    }


def get_registry_dataset(
    dataset_id: str,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> dict[str, Any]:
    registry, resolved_path = _load_dataset_registry(registry_path)
    entry = _registry_entry(registry, dataset_id)
    manifest_path = _dataset_manifest_path(entry, resolved_path)
    manifest = _read_yaml(manifest_path)
    paths = manifest.get("paths") or {}
    root = manifest_path.parent
    resources_payload = {
        key: {
            "path": str((root / str(value)).resolve()),
            "exists": (root / str(value)).resolve().exists(),
        }
        for key, value in paths.items()
    }
    return {
        **_dataset_summary(dataset_id, entry, resolved_path),
        "manifest": manifest,
        "resources": resources_payload,
    }


def get_registry_dataset_documents(
    dataset_id: str,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    manifest_path = _registry_dataset_manifest_path(dataset_id, registry_path)
    path = _dataset_resource_path(manifest_path, "documents")
    return {
        "dataset_id": dataset_id,
        "path": str(path),
        "documents": _read_parquet_records(path, limit=limit),
    }


def get_registry_dataset_schema(
    dataset_id: str,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> dict[str, Any]:
    manifest_path = _registry_dataset_manifest_path(dataset_id, registry_path)
    path = _dataset_resource_path(manifest_path, "schema")
    schema = _read_json(path)
    return {
        "dataset_id": dataset_id,
        "path": str(path),
        "schema": schema,
    }


def get_registry_dataset_queries(
    dataset_id: str,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> dict[str, Any]:
    manifest_path = _registry_dataset_manifest_path(dataset_id, registry_path)
    queries_path = _dataset_resource_path(manifest_path, "queries")
    schema_path = _dataset_resource_path(manifest_path, "schema")
    queries = _read_json(queries_path)
    schema = _read_json(schema_path)
    records = queries.get("queries", []) if isinstance(queries, dict) else []
    if not records:
        records = build_default_query_records(schema, dataset_id=dataset_id)
    return {
        "dataset_id": dataset_id,
        "path": str(queries_path),
        "schema_path": str(schema_path),
        "queries": records,
        "default_extraction": bool(records) and all(
            bool(record.get("default_extraction"))
            for record in records
            if isinstance(record, dict)
        ),
    }


def list_output_results(
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    """Return summaries of persisted JSON artifacts under the project outputs directory."""
    root = resolve_repo_path(outputs_dir)
    if limit < 1:
        raise ValueError("`limit` must be greater than 0.")
    if not root.exists():
        return {"outputs_dir": str(root), "results": []}
    if not root.is_dir():
        raise ValueError(f"Outputs path is not a directory: {root}")

    result_paths = sorted(
        (path for path in root.rglob("*.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    return {
        "outputs_dir": str(root),
        "results": [_output_result_summary(path, root) for path in result_paths],
    }


def delete_output_result(
    relative_path: str,
    outputs_dir: str | Path = DEFAULT_OUTPUTS_DIR,
) -> dict[str, Any]:
    root = resolve_repo_path(outputs_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Outputs directory not found: {root}")

    raw_path = Path(str(relative_path or ""))
    if raw_path.is_absolute() or not str(relative_path or "").strip():
        raise ValueError("Result path must be a relative path under outputs/.")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Result path must stay under outputs/.") from exc
    if candidate.suffix.lower() != ".json":
        raise ValueError("Only JSON result files can be deleted.")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Result file not found: {relative_path}")

    summary = _output_result_summary(candidate, root)
    candidate.unlink()
    return {"deleted": True, "result": summary}


class _WebRunRecord:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status = "queued"
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.events: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.error: dict[str, Any] | None = None
        self.condition = threading.Condition()

    def append(self, event: dict[str, Any]) -> None:
        with self.condition:
            payload = {
                "run_id": self.run_id,
                "sequence": len(self.events),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **event,
            }
            self.events.append(payload)
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "started_at": (
                    datetime.fromtimestamp(self.started_at, timezone.utc).isoformat()
                    if self.started_at
                    else None
                ),
                "ended_at": (
                    datetime.fromtimestamp(self.ended_at, timezone.utc).isoformat()
                    if self.ended_at
                    else None
                ),
                "elapsed_seconds": self.elapsed_seconds(),
                "result": self.result,
                "error": self.error,
                "events": list(self.events),
            }

    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        ended = self.ended_at or time.time()
        return round(ended - self.started_at, 4)


class _WebRunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, _WebRunRecord] = {}
        self._lock = threading.Lock()

    def create(self) -> _WebRunRecord:
        record = _WebRunRecord(uuid.uuid4().hex[:12])
        with self._lock:
            self._runs[record.run_id] = record
        record.append(
            {
                "type": "run_queued",
                "step": None,
                "message": "Run queued.",
                "status": "queued",
            }
        )
        return record

    def get(self, run_id: str) -> _WebRunRecord | None:
        with self._lock:
            return self._runs.get(run_id)


def _run_request_from_payload(
    payload: dict[str, Any],
    *,
    default_config_text: str,
    default_experiment: str,
) -> tuple[str, str, dict[str, Any]]:
    config_path = str(payload.get("config_path") or default_config_text).strip()
    experiment = str(payload.get("experiment") or default_experiment).strip()
    stages = _clean_optional_list(payload.get("stages"))
    datasets = _clean_optional_list(payload.get("datasets"))
    query_ids = _clean_optional_list(payload.get("query_ids"))
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        config_payload = None
    api_key = payload.get("api_key") or None
    if api_key is not None:
        api_key = str(api_key)
    force_rerun = bool(payload.get("force_rerun", False))

    run_kwargs: dict[str, Any] = {
        "stages": stages,
        "datasets": datasets,
        "query_ids": query_ids,
        "api_key": api_key,
    }
    if force_rerun:
        run_kwargs["force_rerun"] = True
    if config_payload is not None:
        run_kwargs["config"] = config_payload
    return config_path, experiment, run_kwargs


def _run_web_demo_async(
    record: _WebRunRecord,
    payload: dict[str, Any],
    *,
    default_config_text: str,
    default_experiment: str,
) -> None:
    record.started_at = time.time()
    record.status = "running"
    record.append(
        {
            "type": "run_started",
            "step": None,
            "message": "Run started.",
            "status": "running",
        }
    )
    started = time.perf_counter()

    def sink(event: dict[str, Any]) -> None:
        record.append(event)

    try:
        sink(
            {
                "type": "step_started",
                "step": "validate_request",
                "message": "Validating run request.",
            }
        )
        config_path, experiment, run_kwargs = _run_request_from_payload(
            payload,
            default_config_text=default_config_text,
            default_experiment=default_experiment,
        )
        sink(
            {
                "type": "step_completed",
                "step": "validate_request",
                "message": "Run request validated.",
                "stages": run_kwargs.get("stages"),
                "datasets": run_kwargs.get("datasets"),
            }
        )

        with progress_event_sink(sink):
            result = run_web_demo(
                config_path,
                experiment,
                **run_kwargs,
                event_sink=sink,
            )

        sink(
            {
                "type": "step_started",
                "step": "evaluation",
                "message": "Running evaluation for completed extraction results.",
            }
        )
        evaluation_summary = _run_web_evaluation(
            config_path,
            experiment,
            run_kwargs,
            result,
        )
        if evaluation_summary.get("status") != "not_enabled":
            sink(
                {
                    "type": "evaluation_update",
                    "step": "evaluation",
                    "message": evaluation_summary.get("message", "Evaluation updated."),
                    "evaluation": evaluation_summary,
                }
            )
        sink(
            {
                "type": "step_completed",
                "step": "evaluation",
                "message": evaluation_summary.get("message", "Evaluation completed."),
            }
        )

        sink(
            {
                "type": "step_started",
                "step": "collect_optimization_metrics",
                "message": "Collecting optimization metrics from emitted artifacts.",
            }
        )
        optimization_metrics = collect_web_optimization_metrics(result)
        for metric in optimization_metrics:
            sink(
                {
                    "type": "optimization_update",
                    "step": "collect_optimization_metrics",
                    "message": metric.get("message", ""),
                    "optimization": metric,
                }
            )
        sink(
            {
                "type": "step_completed",
                "step": "collect_optimization_metrics",
                "message": "Optimization metrics collected.",
            }
        )

        result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        result["optimization_metrics"] = optimization_metrics
        result["evaluation"] = evaluation_summary
        record.result = result
        record.status = "completed"
        record.ended_at = time.time()
        record.append(
            {
                "type": "run_completed",
                "step": None,
                "message": "Run completed.",
                "status": "completed",
                "elapsed_seconds": result["elapsed_seconds"],
                "payload": result,
            }
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        _fail_web_run_record(record, exc, status_code=400)
    except Exception as exc:
        _fail_web_run_record(record, exc, status_code=500)


def _run_web_evaluation(
    config_path: str | Path,
    experiment: str,
    run_kwargs: dict[str, Any],
    run_payload: dict[str, Any],
) -> dict[str, Any]:
    if DATA_EXTRACTION.value not in run_payload.get("stages", []):
        return {
            "id": "evaluation",
            "status": "not_enabled",
            "message": "Evaluation skipped because data_extraction is disabled.",
            "summary": {},
            "queries": [],
        }

    config_payload = run_kwargs.get("config")
    if config_payload is None:
        runtime, _ = load_experiment_runtime(config_path, experiment)
    else:
        runtime = select_experiment(ReDDConfig.model_validate(config_payload), experiment)

    datasets = [str(dataset) for dataset in run_payload.get("datasets", [])]
    query_ids = [str(query_id) for query_id in (run_kwargs.get("query_ids") or [])]
    runtime = _runtime_with_selected_datasets(runtime, datasets, query_ids)

    from redd.exp.evaluation import EvalDataExtraction

    evaluator = EvalDataExtraction(
        runtime.stage_runtime_dict(DATA_EXTRACTION.value),
        api_key=run_kwargs.get("api_key"),
    )
    evaluator(datasets)
    return collect_web_evaluation_summary(run_payload)


def _fail_web_run_record(
    record: _WebRunRecord,
    exc: Exception,
    *,
    status_code: int,
) -> None:
    record.status = "failed"
    record.ended_at = time.time()
    record.error = {
        "error": exc.__class__.__name__,
        "detail": str(exc),
        "status_code": status_code,
    }
    record.append(
        {
            "type": "run_failed",
            "step": None,
            "message": str(exc),
            "status": "failed",
            "error": record.error,
        }
    )


def _format_sse_event(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_web_demo_app(
    *,
    default_config: str | Path = DEFAULT_WEB_DEMO_CONFIG,
    default_experiment: str = DEFAULT_WEB_DEMO_EXPERIMENT,
):
    """Create the FastAPI app used by the packaged web demo."""
    _load_web_dotenv()
    deps = _load_web_dependencies()
    FastAPI = deps["FastAPI"]
    Body = deps["Body"]
    HTTPException = deps["HTTPException"]
    HTMLResponse = deps["HTMLResponse"]
    JSONResponse = deps["JSONResponse"]
    FileResponse = deps["FileResponse"]
    PlainTextResponse = deps["PlainTextResponse"]
    StreamingResponse = deps["StreamingResponse"]

    app = FastAPI(title="ReDD Web Demo")
    run_registry = _WebRunRegistry()
    default_config_text = str(default_config)
    if default_config_text == DEFAULT_WEB_DEMO_CONFIG:
        default_config_text = os.getenv("REDD_WEB_DEMO_CONFIG", default_config_text)
    if default_experiment == DEFAULT_WEB_DEMO_EXPERIMENT:
        default_experiment = os.getenv("REDD_WEB_DEMO_EXPERIMENT", default_experiment)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_read_web_resource("index.html"))

    @app.exception_handler(404)
    def not_found(request: Any, exc: Any):
        path = str(request.url.path)
        if path.startswith("/api/") or path.startswith("/assets/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        escaped_path = html.escape(path)
        return HTMLResponse(
            f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Not Found - ReDD</title>
    <script>
      document.documentElement.dataset.theme = localStorage.getItem("redd-theme") || "light";
    </script>
    <link rel="stylesheet" href="/assets/styles.css" />
  </head>
  <body>
    <main class="not-found-page">
      <section class="not-found-panel">
        <div class="brand-mark">R</div>
        <p class="eyebrow">Not Found</p>
        <h1>Page not found</h1>
        <p>{escaped_path}</p>
        <a class="pill-link" href="/">Back to Workbench</a>
      </section>
    </main>
  </body>
</html>""",
            status_code=404,
        )

    @app.get("/assets/app.js")
    def app_js():
        return PlainTextResponse(
            _read_web_resource("app.js"),
            media_type="application/javascript; charset=utf-8",
        )

    @app.get("/assets/styles.css")
    def styles_css():
        return PlainTextResponse(
            _read_web_resource("styles.css"),
            media_type="text/css; charset=utf-8",
        )

    @app.api_route("/assets/papers/{paper_name}", methods=["GET", "HEAD"])
    def paper_pdf(paper_name: str):
        try:
            paper_path = WEB_DEMO_PAPERS[paper_name]
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "FileNotFoundError", "detail": f"Paper `{paper_name}` not found."},
            ) from exc
        resolved_path = resolve_repo_path(paper_path)
        if not resolved_path.exists():
            raise HTTPException(
                status_code=404,
                detail={"error": "FileNotFoundError", "detail": str(resolved_path)},
            )
        return FileResponse(
            resolved_path,
            headers={"Cache-Control": "no-store"},
            media_type="application/pdf",
        )

    @app.get("/api/defaults")
    def defaults():
        return {
            "config_path": default_config_text,
            "experiment": default_experiment,
            "stages": WEB_DEMO_STAGES,
            "default_stages": [DATA_EXTRACTION.value],
            "models": DEFAULT_WEB_DEMO_MODELS,
            "api_key_status": _api_key_status(),
            "model_catalog": get_model_catalog(),
        }

    @app.get("/api/config/inspect")
    def config_inspect(config_path: str | None = None, experiment: str | None = None):
        try:
            resolved_config = str(config_path or default_config_text).strip()
            resolved_experiment = str(experiment or default_experiment).strip()
            return JSONResponse(inspect_web_config(resolved_config, resolved_experiment))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.get("/api/datasets")
    def datasets_index():
        try:
            return JSONResponse(list_registry_datasets())
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.get("/api/datasets/{dataset_id}")
    def dataset_detail(dataset_id: str):
        try:
            return JSONResponse(get_registry_dataset(dataset_id))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.get("/api/datasets/{dataset_id}/documents")
    def dataset_documents(dataset_id: str, limit: int = 50):
        try:
            return JSONResponse(get_registry_dataset_documents(dataset_id, limit=limit))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.get("/api/datasets/{dataset_id}/schema")
    def dataset_schema(dataset_id: str):
        try:
            return JSONResponse(get_registry_dataset_schema(dataset_id))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.get("/api/datasets/{dataset_id}/queries")
    def dataset_queries(dataset_id: str):
        try:
            return JSONResponse(get_registry_dataset_queries(dataset_id))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.get("/api/results")
    def output_results(limit: int = 200):
        try:
            return JSONResponse(list_output_results(limit=limit))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.delete("/api/results")
    def delete_result(payload: dict[str, Any] = Body(default_factory=dict)):
        try:
            relative_path = str(payload.get("relative_path") or "").strip()
            return JSONResponse(delete_output_result(relative_path))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

    @app.post("/api/runs")
    def create_run(payload: dict[str, Any] = Body(default_factory=dict)):
        record = run_registry.create()
        worker = threading.Thread(
            target=_run_web_demo_async,
            kwargs={
                "record": record,
                "payload": payload,
                "default_config_text": default_config_text,
                "default_experiment": default_experiment,
            },
            daemon=True,
        )
        worker.start()
        return JSONResponse({"run_id": record.run_id, "status": record.status})

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        record = run_registry.get(run_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "RunNotFound", "detail": f"Run `{run_id}` not found."},
            )
        return JSONResponse(record.snapshot())

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str):
        record = run_registry.get(run_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "RunNotFound", "detail": f"Run `{run_id}` not found."},
            )

        def event_stream():
            index = 0
            while True:
                with record.condition:
                    while index >= len(record.events) and record.status not in {"completed", "failed"}:
                        record.condition.wait(timeout=0.5)
                    events = record.events[index:]
                    index = len(record.events)

                for event in events:
                    yield _format_sse_event(event)

                if record.status in {"completed", "failed"} and index >= len(record.events):
                    break

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/run")
    def run_demo(payload: dict[str, Any] = Body(default_factory=dict)):
        started = time.perf_counter()
        try:
            config_path, experiment, run_kwargs = _run_request_from_payload(
                payload,
                default_config_text=default_config_text,
                default_experiment=default_experiment,
            )
            result = run_web_demo(config_path, experiment, **run_kwargs)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": exc.__class__.__name__, "detail": str(exc)},
            ) from exc

        result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        return JSONResponse(result)

    return app


def serve_web_demo(
    *,
    config_path: str | Path = DEFAULT_WEB_DEMO_CONFIG,
    experiment: str = DEFAULT_WEB_DEMO_EXPERIMENT,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Serve the packaged FastAPI web demo with uvicorn."""
    deps = _load_web_dependencies()
    uvicorn = deps["uvicorn"]
    if reload:
        os.environ["REDD_WEB_DEMO_CONFIG"] = str(config_path)
        os.environ["REDD_WEB_DEMO_EXPERIMENT"] = experiment
        uvicorn.run(
            "redd.web_demo:create_web_demo_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return

    app = create_web_demo_app(
        default_config=config_path,
        default_experiment=experiment,
    )
    uvicorn.run(app, host=host, port=port, reload=reload)


def _stage_runtime_or_none(runtime: ExperimentRuntime, stage: StageName) -> dict[str, Any] | None:
    if stage not in runtime.stages or not runtime.stages[stage].enabled:
        return None
    return runtime.stage_runtime_dict(stage)


def _data_extraction_runtime(
    runtime: ExperimentRuntime,
    query_ids: Sequence[str],
    *,
    force_rerun: bool = False,
) -> dict[str, Any]:
    config = runtime.stage_runtime_dict("data_extraction")
    if query_ids:
        config["exp_query_id_list"] = [str(query_id) for query_id in query_ids]
    if force_rerun:
        config["force_rerun"] = True
    return config


def _runtime_with_selected_datasets(
    runtime: ExperimentRuntime,
    requested_datasets: Sequence[str],
    query_ids: Sequence[str] | None = None,
) -> ExperimentRuntime:
    selected: dict[str, DatasetRuntimeConfig] = {}
    registry: dict[str, Any] | None = None
    registry_path: Path | None = None

    for dataset_id in requested_datasets:
        if dataset_id in runtime.datasets:
            selected[dataset_id] = runtime.datasets[dataset_id]
            continue

        if registry is None or registry_path is None:
            registry, registry_path = _load_dataset_registry()
        try:
            entry = _registry_entry(registry, dataset_id)
        except KeyError as exc:
            raise ValueError(
                f"Dataset `{dataset_id}` is neither in the selected config experiment "
                "nor registered in dataset/manifest.yaml."
            ) from exc

        selected[dataset_id] = DatasetRuntimeConfig(
            id=dataset_id,
            root=_dataset_manifest_path(entry, registry_path).parent,
            loader="hf_manifest",
            query_ids=[str(query_id) for query_id in query_ids] if query_ids else None,
            split=DatasetSplitConfig(),
            loader_options={"manifest": "manifest.yaml"},
        )

    return runtime.model_copy(update={"datasets": selected})


def _load_dataset_registry(
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> tuple[dict[str, Any], Path]:
    resolved_path = resolve_repo_path(registry_path)
    registry = _read_yaml(resolved_path)
    datasets = registry.get("datasets") or {}
    if not isinstance(datasets, dict):
        raise TypeError("Dataset registry `datasets` must be a mapping.")
    return registry, resolved_path


def _dataset_search_sort_key(dataset_id: str) -> tuple[int, str]:
    group = dataset_id.split(".", 1)[0].lower()
    try:
        group_index = DATASET_SEARCH_GROUP_ORDER.index(group)
    except ValueError:
        group_index = len(DATASET_SEARCH_GROUP_ORDER)
    return group_index, dataset_id.lower()


def _registry_entry(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    datasets = registry.get("datasets") or {}
    entry = datasets.get(dataset_id)
    if not isinstance(entry, dict):
        raise KeyError(f"Dataset `{dataset_id}` not found in dataset registry.")
    return entry


def _registry_dataset_manifest_path(
    dataset_id: str,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> Path:
    registry, resolved_path = _load_dataset_registry(registry_path)
    return _dataset_manifest_path(_registry_entry(registry, dataset_id), resolved_path)


def _dataset_manifest_path(entry: dict[str, Any], registry_path: Path) -> Path:
    raw_path = entry.get("path")
    if not raw_path:
        raise ValueError("Dataset registry entry is missing `path`.")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (registry_path.parent / path).resolve()


def _dataset_resource_path(manifest_path: Path, resource: str) -> Path:
    manifest = _read_yaml(manifest_path)
    paths = manifest.get("paths") or {}
    raw_path = paths.get(resource)
    if not raw_path:
        raise KeyError(f"Dataset manifest is missing `{resource}` path.")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _dataset_summary(dataset_id: str, entry: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    manifest_path = _dataset_manifest_path(entry, registry_path)
    manifest = _read_yaml(manifest_path)
    paths = manifest.get("paths") or {}
    documents_path = _safe_resource_path(manifest_path, paths, "documents")
    schema_path = _safe_resource_path(manifest_path, paths, "schema")
    queries_path = _safe_resource_path(manifest_path, paths, "queries")
    ground_truth_path = _safe_resource_path(manifest_path, paths, "ground_truth")
    schema = _read_json(schema_path) if schema_path and schema_path.exists() else {}
    queries = _read_json(queries_path) if queries_path and queries_path.exists() else {}
    query_records = queries.get("queries", []) if isinstance(queries, dict) else []
    effective_query_records = (
        query_records
        or build_default_query_records(schema, dataset_id=dataset_id)
    )
    return {
        "id": dataset_id,
        "manifest_dataset_id": manifest.get("dataset_id"),
        "kind": entry.get("kind") or manifest.get("kind"),
        "path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "documents_path": str(documents_path) if documents_path else None,
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "schema_path": str(schema_path) if schema_path else None,
        "queries_path": str(queries_path) if queries_path else None,
        "documents_count": _parquet_row_count(documents_path) if documents_path else None,
        "queries_count": len(effective_query_records),
        "explicit_queries_count": len(query_records),
        "default_extraction": bool(effective_query_records) and not query_records,
        "tables_count": len(schema.get("tables") or []) if isinstance(schema, dict) else None,
        "exists": manifest_path.exists(),
    }


def _output_result_summary(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        content = _read_json(path)
        parse_error = None
    except Exception as exc:
        content = {}
        parse_error = f"{exc.__class__.__name__}: {exc}"

    relative_path = path.relative_to(root)
    parts = relative_path.parts
    project = parts[0] if parts else "outputs"
    stage = next((part for part in parts if part in WEB_DEMO_STAGE_SET), None)
    stage_index = parts.index(stage) if stage in parts else -1
    dataset_id = parts[stage_index - 1] if stage_index > 0 else None
    artifact = "/".join(parts[stage_index + 1 : -1]) if stage_index >= 0 else ""
    query_id = _query_id_from_result_name(path.name) or _query_id_from_eval_name(path.name)
    stats = _json_result_stats(content)
    return {
        "name": path.name,
        "project": project,
        "relative_path": str(relative_path),
        "path": str(path),
        "dataset_id": dataset_id,
        "stage": stage,
        "artifact": artifact,
        "query_id": query_id,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
        "parse_error": parse_error,
        **stats,
    }


def _query_id_from_result_name(name: str) -> str | None:
    match = re.match(r"res_tabular_data_(?P<query>.+?)_.+\.json$", name)
    return match.group("query") if match else None


def _query_id_from_eval_name(name: str) -> str | None:
    match = re.match(r"eval_(?P<query>.+?)_.+\.json$", name)
    return match.group("query") if match else None


def _json_result_stats(content: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(content, dict):
        return {"records_count": None, "tables": [], "columns": [], "preview": []}

    query_aware = content.get("query_aware")
    if isinstance(query_aware, dict):
        return _json_evaluation_stats(query_aware)

    tables: set[str] = set()
    columns: set[str] = set()
    preview: list[dict[str, Any]] = []
    for doc_id, record in list(content.items())[:5]:
        if not isinstance(record, dict):
            preview.append({"id": str(doc_id), "value": record})
            continue
        table = record.get("res")
        if table:
            tables.add(str(table))
        data = record.get("data")
        if isinstance(data, dict):
            columns.update(str(key) for key in data.keys())
        preview.append(
            {
                "id": str(doc_id),
                "table": table,
                "data": data if isinstance(data, dict) else None,
            }
        )
    return {
        "records_count": len(content),
        "tables": sorted(tables),
        "columns": sorted(columns),
        "preview": preview,
    }


def _json_evaluation_stats(query_aware: dict[str, Any]) -> dict[str, Any]:
    summary = query_aware.get("summary") if isinstance(query_aware.get("summary"), dict) else {}
    table = _recall_metric(query_aware.get("table_assignment") or query_aware.get("table_recall"))
    cell = _recall_metric(query_aware.get("cell_recall"))
    answer = _recall_metric(query_aware.get("answer_recall"))
    return {
        "records_count": None,
        "tables": [],
        "columns": [],
        "preview": [
            {
                "id": str(query_aware.get("query_id") or ""),
                "value": {
                    "can_answer_query": bool(summary.get("can_answer_query")),
                    "table_recall": table,
                    "cell_recall": cell,
                    "sql_answer_recall": answer,
                },
            }
        ],
        "evaluation": {
            "query_id": str(query_aware.get("query_id") or ""),
            "can_answer_query": bool(summary.get("can_answer_query")),
            "table_recall": table,
            "cell_recall": cell,
            "sql_answer_recall": answer,
        },
    }


def _recall_metric(value: Any) -> dict[str, Any]:
    metric = value if isinstance(value, dict) else {}
    covered = int(metric.get("covered") or 0)
    total = int(metric.get("total") or 0)
    recall = metric.get("recall")
    if not isinstance(recall, (int, float)):
        recall = _ratio(covered, total)
    return {
        "covered": covered,
        "total": total,
        "recall": float(recall),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 1.0


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def collect_web_evaluation_summary(run_payload: dict[str, Any]) -> dict[str, Any]:
    out_roots = _output_roots_from_run_payload(run_payload)
    query_ids_by_root = _query_ids_by_output_root(run_payload)
    files = _unique_existing_files(
        path
        for root in out_roots
        for path in root.glob("eval_*.json")
        if _artifact_query_allowed(
            root,
            _query_id_from_eval_name(path.name),
            query_ids_by_root,
        )
    )

    if not files:
        return {
            "id": "evaluation",
            "status": "no_metrics",
            "message": "No evaluation artifacts were found.",
            "summary": {},
            "queries": [],
        }

    queries: list[dict[str, Any]] = []
    totals = {
        "queries": 0,
        "can_answer": 0,
        "table_covered": 0,
        "table_total": 0,
        "cell_covered": 0,
        "cell_total": 0,
        "answer_covered": 0,
        "answer_total": 0,
    }

    for path in files:
        content = _read_json(path)
        query_aware = content.get("query_aware") if isinstance(content, dict) else {}
        if not isinstance(query_aware, dict):
            continue
        summary = query_aware.get("summary") if isinstance(query_aware.get("summary"), dict) else {}
        table = _recall_metric(query_aware.get("table_assignment") or query_aware.get("table_recall"))
        cell = _recall_metric(query_aware.get("cell_recall"))
        answer = _recall_metric(query_aware.get("answer_recall"))
        query_id = str(query_aware.get("query_id") or _query_id_from_eval_name(path.name) or "")
        dataset = _dataset_id_from_out_root(path.parent) or ""
        can_answer = bool(summary.get("can_answer_query"))
        queries.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "can_answer_query": can_answer,
                "table_recall": table,
                "cell_recall": cell,
                "sql_answer_recall": answer,
                "path": str(path),
            }
        )
        totals["queries"] += 1
        totals["can_answer"] += int(can_answer)
        totals["table_covered"] += int(table["covered"])
        totals["table_total"] += int(table["total"])
        totals["cell_covered"] += int(cell["covered"])
        totals["cell_total"] += int(cell["total"])
        totals["answer_covered"] += int(answer["covered"])
        totals["answer_total"] += int(answer["total"])

    queries.sort(key=lambda item: (item.get("dataset") or "", item.get("query_id") or ""))
    summary = {
        **totals,
        "table_recall": _ratio(totals["table_covered"], totals["table_total"]),
        "cell_recall": _ratio(totals["cell_covered"], totals["cell_total"]),
        "sql_answer_recall": _ratio(totals["answer_covered"], totals["answer_total"]),
    }
    return {
        "id": "evaluation",
        "status": "measured",
        "message": (
            "Evaluation measured: SQL answer recall "
            f"{totals['answer_covered']}/{totals['answer_total']} "
            f"({_format_percent(summary['sql_answer_recall'])})."
        ),
        "summary": summary,
        "queries": queries,
    }


def collect_web_optimization_metrics(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect demo-friendly optimization metrics from real emitted artifacts."""
    out_roots = _output_roots_from_run_payload(run_payload)
    query_ids_by_root = _query_ids_by_output_root(run_payload)
    metrics = [
        _collect_doc_filter_metric(out_roots, query_ids_by_root),
        _collect_schema_adaptive_metric(out_roots, query_ids_by_root),
        _collect_table_assignment_cache_metric(out_roots, query_ids_by_root),
        _collect_proxy_metric(out_roots, query_ids_by_root),
        _collect_join_proxy_metric(out_roots, query_ids_by_root),
        _collect_dataset_consistency_audit_metric(out_roots, run_payload),
        _collect_extraction_metric(run_payload, out_roots, query_ids_by_root),
    ]
    return [
        metric
        for metric in metrics
        if metric.get("id") == "extraction" or metric.get("status") != "not_enabled"
    ]


def _output_roots_from_run_payload(run_payload: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    result = run_payload.get("result") if isinstance(run_payload, dict) else {}
    if not isinstance(result, dict):
        return roots
    for stage_value in result.values():
        if not isinstance(stage_value, list):
            continue
        for item in stage_value:
            if not isinstance(item, dict) or not item.get("out_root"):
                continue
            path = Path(str(item["out_root"]))
            if not path.is_absolute():
                path = resolve_repo_path(path)
            try:
                path = path.resolve()
            except OSError:
                pass
            if path not in seen:
                roots.append(path)
                seen.add(path)
    return roots


def _query_ids_by_output_root(run_payload: dict[str, Any]) -> dict[Path, set[str]]:
    roots: dict[Path, set[str]] = {}
    result = run_payload.get("result") if isinstance(run_payload, dict) else {}
    if not isinstance(result, dict):
        return roots
    payload_query_ids = {
        str(query_id)
        for query_id in run_payload.get("query_ids", [])
        if str(query_id)
    }
    for stage_value in result.values():
        if not isinstance(stage_value, list):
            continue
        for item in stage_value:
            if not isinstance(item, dict) or not item.get("out_root"):
                continue
            raw_query_ids = item.get("query_ids")
            query_ids = raw_query_ids if isinstance(raw_query_ids, list) and raw_query_ids else list(payload_query_ids)
            if not query_ids:
                continue
            path = Path(str(item["out_root"]))
            if not path.is_absolute():
                path = resolve_repo_path(path)
            try:
                path = path.resolve()
            except OSError:
                pass
            roots[path] = {str(query_id) for query_id in query_ids}
    return roots


def _collect_doc_filter_metric(
    out_roots: Sequence[Path],
    query_ids_by_root: Mapping[Path, set[str]] | None = None,
) -> dict[str, Any]:
    doc_filter_files = _unique_existing_files(
        path
        for root in out_roots
        for path in root.glob("doc_filter/*.json")
        if _artifact_query_allowed(
            root,
            _query_id_from_filter_name(path.name),
            query_ids_by_root,
        )
    )
    chunk_filter_files = _unique_existing_files(
        path
        for root in out_roots
        for path in root.glob("chunk_filter/*.json")
        if _artifact_query_allowed(
            root,
            _query_id_from_filter_name(path.name),
            query_ids_by_root,
        )
    )
    files = doc_filter_files or chunk_filter_files
    totals = {
        "input_docs": 0,
        "kept_docs": 0,
        "excluded_docs": 0,
        "queries": 0,
    }
    target_recalls: list[float] = []
    details: list[dict[str, Any]] = []
    for path in files:
        content = _safe_read_result_json(path)
        metadata = content.get("metadata") if isinstance(content, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        input_docs = _metric_int(metadata.get("num_docs_input"))
        kept_docs = _metric_int(metadata.get("num_docs_kept"))
        excluded_docs = _metric_int(metadata.get("num_docs_excluded"))
        if kept_docs is None and isinstance(content, dict):
            kept_docs = len(content.get("kept_doc_ids") or [])
        if excluded_docs is None and isinstance(content, dict):
            excluded_docs = len(content.get("excluded_doc_ids") or [])
        if input_docs is None and kept_docs is not None and excluded_docs is not None:
            input_docs = kept_docs + excluded_docs
        totals["input_docs"] += input_docs or 0
        totals["kept_docs"] += kept_docs or 0
        totals["excluded_docs"] += excluded_docs or 0
        totals["queries"] += 1
        recall = _metric_float(metadata.get("target_recall"))
        if recall is not None:
            target_recalls.append(recall)
        query_id = str(content.get("query_id") or _query_id_from_filter_name(path.name) or "")
        excluded_ids = [str(doc_id) for doc_id in (content.get("excluded_doc_ids") or [])]
        details.append(
            {
                "kind": "doc_filter",
                "dataset": _dataset_id_from_out_root(path.parent.parent),
                "query_id": query_id,
                "input_docs": input_docs or 0,
                "kept_docs": kept_docs or 0,
                "excluded_docs": excluded_docs or 0,
                "llm_doc_calls_saved": excluded_docs or 0,
                "reduction": (excluded_docs / input_docs) if input_docs else None,
                "excluded_doc_ids_preview": sorted(excluded_ids)[:8],
                "excluded_doc_ids_total": len(excluded_ids),
            }
        )

    if not files:
        return {
            "id": "doc_filter",
            "title": "Chunk / Document Filter",
            "status": "not_enabled",
            "message": "No doc_filter or chunk_filter artifact was emitted for this run.",
            "metrics": {},
        }

    skip_rate = (
        totals["excluded_docs"] / totals["input_docs"]
        if totals["input_docs"]
        else None
    )
    llm_doc_calls_before = totals["input_docs"]
    llm_doc_calls_after = totals["kept_docs"]
    llm_doc_calls_saved = max(llm_doc_calls_before - llm_doc_calls_after, 0)
    return {
        "id": "doc_filter",
        "title": "Chunk / Document Filter",
        "status": "measured",
        "message": "Filtered schema-irrelevant documents before expensive LLM extraction.",
        "metrics": {
            "llm_doc_calls_before": llm_doc_calls_before,
            "llm_doc_calls_after": llm_doc_calls_after,
            "llm_doc_calls_saved": llm_doc_calls_saved,
            "llm_doc_call_reduction": (
                llm_doc_calls_saved / llm_doc_calls_before
                if llm_doc_calls_before
                else None
            ),
            "skip_rate": skip_rate,
            **totals,
            "target_recall": _average(target_recalls) if target_recalls else None,
            "artifact_count": len(files),
        },
        "details": details[-50:],
    }


def _collect_schema_adaptive_metric(
    out_roots: Sequence[Path],
    query_ids_by_root: Mapping[Path, set[str]] | None = None,
) -> dict[str, Any]:
    files = _unique_existing_files(
        path
        for root in out_roots
        for path in root.rglob("*_adaptive_stats.json")
        if _artifact_query_allowed(
            root,
            _query_id_from_adaptive_stats_name(path.name),
            query_ids_by_root,
        )
    )
    totals = {
        "queries": 0,
        "total_documents": 0,
        "filtered_documents": 0,
        "documents_processed": 0,
        "documents_saved": 0,
        "stopped_early": 0,
    }
    for path in files:
        content = _safe_read_result_json(path)
        if not isinstance(content, dict):
            continue
        totals["queries"] += 1
        totals["total_documents"] += _metric_int(content.get("total_documents")) or 0
        filtered = _metric_int(content.get("filtered_documents"))
        processed = _metric_int(content.get("documents_processed"))
        saved = _metric_int(content.get("documents_saved"))
        if filtered is None:
            filtered = _metric_int(content.get("total_documents"))
        if processed is None:
            processed = _metric_int(content.get("n_processed"))
        if saved is None and filtered is not None and processed is not None:
            saved = max(filtered - processed, 0)
        totals["filtered_documents"] += filtered or 0
        totals["documents_processed"] += processed or 0
        totals["documents_saved"] += saved or 0
        if content.get("stopped_early"):
            totals["stopped_early"] += 1

    if not files:
        return {
            "id": "schema_adaptive",
            "title": "Schema Adaptive Sampling",
            "status": "not_enabled",
            "message": "No schema adaptive sampling artifact was emitted for this run.",
            "metrics": {},
        }

    processed = totals["documents_processed"]
    filtered = totals["filtered_documents"]
    return {
        "id": "schema_adaptive",
        "title": "Schema Adaptive Sampling",
        "status": "measured",
        "message": "Stopped schema refinement once schema entropy stabilized.",
        "metrics": {
            **totals,
            "sampling_rate": processed / filtered if filtered else None,
            "artifact_count": len(files),
        },
    }


def _collect_table_assignment_cache_metric(
    out_roots: Sequence[Path],
    query_ids_by_root: Mapping[Path, set[str]] | None = None,
) -> dict[str, Any]:
    files = _unique_existing_files(root / "table_assignment_cache.json" for root in out_roots)
    totals = {
        "input_docs": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "source_table_metadata_hits": 0,
        "source_table_metadata_misses": 0,
        "excluded": 0,
        "table_assignment_calls_before": 0,
    }
    details: list[dict[str, Any]] = []
    for path in files:
        content = _safe_read_result_json(path)
        events = content.get("events") if isinstance(content, dict) else []
        if not isinstance(events, list):
            continue
        root = path.parent
        for event in events:
            if not isinstance(event, dict):
                continue
            query_id = str(event.get("query_id") or "")
            if not _artifact_query_allowed(root, query_id, query_ids_by_root):
                continue
            input_docs = _metric_int(event.get("input_docs")) or 0
            cache_hits = _metric_int(event.get("cache_hits")) or 0
            cache_misses = _metric_int(event.get("cache_misses")) or 0
            source_table_metadata_hits = (
                _metric_int(event.get("source_table_metadata_hits")) or 0
            )
            source_table_metadata_misses = (
                _metric_int(event.get("source_table_metadata_misses")) or 0
            )
            excluded = _metric_int(event.get("excluded")) or 0
            totals["input_docs"] += input_docs
            totals["cache_hits"] += cache_hits
            totals["cache_misses"] += cache_misses
            totals["source_table_metadata_hits"] += source_table_metadata_hits
            totals["source_table_metadata_misses"] += source_table_metadata_misses
            totals["excluded"] += excluded
            calls_before = max(input_docs - excluded, 0)
            calls_saved = cache_hits + source_table_metadata_hits
            totals["table_assignment_calls_before"] += calls_before
            details.append(
                {
                    "kind": "table_assignment_cache",
                    "dataset": event.get("dataset") or _dataset_id_from_out_root(root),
                    "query_id": query_id,
                    "input_docs": input_docs,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "source_table_metadata_hits": source_table_metadata_hits,
                    "source_table_metadata_misses": source_table_metadata_misses,
                    "excluded": excluded,
                    "table_assignment_calls_before": calls_before,
                    "table_assignment_calls_after": cache_misses,
                    "table_assignment_calls_saved": calls_saved,
                    "reduction": calls_saved / calls_before if calls_before else None,
                }
            )

    if not details:
        return {
            "id": "table_assignment_cache",
            "title": "Table Assignment Cache",
            "status": "not_enabled",
            "message": "No table-assignment cache artifact was emitted for this run.",
            "metrics": {},
        }

    calls_before = totals["table_assignment_calls_before"]
    calls_after = totals["cache_misses"]
    calls_saved = totals["cache_hits"] + totals["source_table_metadata_hits"]
    return {
        "id": "table_assignment_cache",
        "title": "Table Assignment Cache",
        "status": "measured",
        "message": "Reused table assignments across queries before extraction.",
        "metrics": {
            "table_assignment_calls_before": calls_before,
            "table_assignment_calls_after": calls_after,
            "table_assignment_calls_saved": calls_saved,
            "table_assignment_call_reduction": (
                calls_saved / calls_before if calls_before else None
            ),
            **totals,
            "artifact_count": len(files),
        },
        "details": details[-50:],
    }


def _collect_proxy_metric(
    out_roots: Sequence[Path],
    query_ids_by_root: Mapping[Path, set[str]] | None = None,
) -> dict[str, Any]:
    files = _unique_existing_files(
        path
        for root in out_roots
        for path in root.rglob("*_proxy_decisions.json")
        if _artifact_query_allowed(
            root,
            _query_id_from_proxy_decision_name(path.name),
            query_ids_by_root,
        )
    )
    totals = {
        "tables": 0,
        "evaluated": 0,
        "passed": 0,
        "rejected": 0,
        "llm_doc_calls_before": 0,
        "llm_doc_calls_after": 0,
        "gt_guard_rejected_doc_calls": 0,
    }
    recalls: list[float] = []
    precisions: list[float] = []
    details: list[dict[str, Any]] = []
    for path in files:
        content = _safe_read_result_json(path)
        if not isinstance(content, dict):
            continue
        query_id = _query_id_from_proxy_decision_name(path.name) or ""
        dataset = _dataset_id_from_out_root(path.parent)
        for table_decision in content.values():
            if not isinstance(table_decision, dict):
                continue
            totals["tables"] += 1
            proxy_stats = table_decision.get("proxy_stats")
            table_gt_guard_rejected = 0
            if isinstance(proxy_stats, dict):
                for proxy_name, stat in proxy_stats.items():
                    if not isinstance(stat, dict):
                        continue
                    totals["evaluated"] += _metric_int(stat.get("evaluated")) or 0
                    totals["passed"] += _metric_int(stat.get("passed")) or 0
                    rejected = _metric_int(stat.get("rejected")) or 0
                    totals["rejected"] += rejected
                    if _is_gt_text_consistency_proxy(proxy_name):
                        table_gt_guard_rejected += rejected
                        totals["gt_guard_rejected_doc_calls"] += rejected
            llm_before, llm_after = _proxy_llm_doc_call_counts(table_decision)
            totals["llm_doc_calls_before"] += llm_before
            totals["llm_doc_calls_after"] += llm_after
            table_name = next(
                (
                    str(name)
                    for name, value in content.items()
                    if value is table_decision
                ),
                "",
            )
            rejected_preview = _proxy_rejected_doc_ids(table_decision)[:8]
            details.append(
                {
                    "kind": "proxy_runtime",
                    "dataset": dataset,
                    "query_id": query_id,
                    "table": table_name,
                    "evaluated": llm_before,
                    "passed": llm_after,
                    "rejected": max(llm_before - llm_after, 0),
                    "llm_doc_calls_before": llm_before,
                    "llm_doc_calls_after": llm_after,
                    "llm_doc_calls_saved": max(llm_before - llm_after, 0),
                    "pass_rate": llm_after / llm_before if llm_before else None,
                    "offline_only_gt_guard": table_gt_guard_rejected > 0,
                    "gt_guard_rejected_doc_calls": table_gt_guard_rejected,
                    "rejected_doc_ids_preview": rejected_preview,
                    "rejected_doc_ids_total": len(_proxy_rejected_doc_ids(table_decision)),
                }
            )
            proxy_recalls = table_decision.get("proxy_recalls")
            if isinstance(proxy_recalls, dict):
                for stat in proxy_recalls.values():
                    if not isinstance(stat, dict):
                        continue
                    recall = _metric_float(stat.get("recall"))
                    precision = _metric_float(stat.get("precision"))
                    if recall is not None:
                        recalls.append(recall)
                    if precision is not None:
                        precisions.append(precision)

    if not files:
        return {
            "id": "proxy_runtime",
            "title": "Proxy Runtime",
            "status": "not_enabled",
            "message": "No proxy decision artifact was emitted for this run.",
            "metrics": {},
        }

    pass_rate = totals["passed"] / totals["evaluated"] if totals["evaluated"] else None
    llm_doc_calls_saved = max(
        totals["llm_doc_calls_before"] - totals["llm_doc_calls_after"],
        0,
    )
    gt_guard_rejected = totals["gt_guard_rejected_doc_calls"]
    proxy_metrics = {
        "llm_doc_calls_before": totals["llm_doc_calls_before"],
        "llm_doc_calls_after": totals["llm_doc_calls_after"],
        "llm_doc_calls_saved": llm_doc_calls_saved,
        "llm_doc_call_reduction": (
            llm_doc_calls_saved / totals["llm_doc_calls_before"]
            if totals["llm_doc_calls_before"]
            else None
        ),
        **totals,
        "offline_only_gt_guard": "enabled"
        if gt_guard_rejected
        else None,
        "pass_rate": pass_rate,
        "avg_recall": _average(recalls) if recalls else None,
        "avg_precision": _average(precisions) if precisions else None,
        "artifact_count": len(files),
    }
    if not gt_guard_rejected:
        proxy_metrics.pop("gt_guard_rejected_doc_calls", None)
    return {
        "id": "proxy_runtime",
        "title": (
            "Proxy Runtime (Offline GT Guard)"
            if gt_guard_rejected
            else "Proxy Runtime"
        ),
        "status": "measured",
        "message": (
            "Offline-only GT/text consistency guard is active; this result is for benchmark ablation, not deployment."
            if gt_guard_rejected
            else "Used lightweight proxy decisions to reduce oracle-bound extraction work."
        ),
        "metrics": proxy_metrics,
        "details": details[-50:],
    }


def _collect_dataset_consistency_audit_metric(
    out_roots: Sequence[Path],
    run_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = _unique_existing_files(
        candidate
        for root in out_roots
        for candidate in _dataset_consistency_audit_candidates(root)
    )
    if not files:
        audit_inputs = _dataset_consistency_inputs_from_payload(run_payload or {})
        if audit_inputs:
            try:
                audit = build_dataset_consistency_audit(audit_inputs)
                output_dir = _common_output_dir(out_roots)
                if output_dir is not None:
                    write_dataset_consistency_audit(output_dir, audit)
                return _dataset_consistency_audit_metric_from_reports(
                    [audit],
                    report_count=1,
                )
            except Exception as exc:
                return {
                    "id": "dataset_consistency_audit",
                    "title": "Dataset Consistency Audit",
                    "status": "no_metrics",
                    "message": f"Dataset consistency audit could not be generated: {exc}",
                    "metrics": {},
                }
    if not files:
        return {
            "id": "dataset_consistency_audit",
            "title": "Dataset Consistency Audit",
            "status": "not_enabled",
            "message": "No dataset consistency audit was emitted for this run.",
            "metrics": {},
        }

    reports = [_safe_read_result_json(path) for path in files]
    return _dataset_consistency_audit_metric_from_reports(
        reports,
        report_count=len(files),
    )


def _dataset_consistency_audit_metric_from_reports(
    reports: Sequence[dict[str, Any]],
    *,
    report_count: int,
) -> dict[str, Any]:
    conflicts: list[dict[str, Any]] = []
    checked_doc_predicates = 0
    datasets_seen: set[str] = set()
    for content in reports:
        for dataset in content.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            dataset_id = str(dataset.get("dataset") or "")
            if dataset_id:
                datasets_seen.add(dataset_id)
            checked_doc_predicates += _metric_int(dataset.get("checked_doc_predicates")) or 0
            for conflict in dataset.get("conflicts") or []:
                if isinstance(conflict, dict):
                    conflicts.append(conflict)

    by_type: dict[str, int] = {}
    affected_docs: set[tuple[str, str]] = set()
    affected_queries: set[tuple[str, str]] = set()
    for conflict in conflicts:
        conflict_type = str(conflict.get("conflict_type") or "unknown")
        by_type[conflict_type] = by_type.get(conflict_type, 0) + 1
        dataset = str(conflict.get("dataset") or "")
        doc_id = str(conflict.get("doc_id") or "")
        query_id = str(conflict.get("query_id") or "")
        if doc_id:
            affected_docs.add((dataset, doc_id))
        if query_id:
            affected_queries.add((dataset, query_id))

    details = [
        {
            "kind": "dataset_consistency_audit",
            "dataset": conflict.get("dataset"),
            "query_id": conflict.get("query_id"),
            "doc_id": conflict.get("doc_id"),
            "attribute": conflict.get("attribute"),
            "conflict_type": conflict.get("conflict_type"),
            "text_values": conflict.get("text_values") or [],
            "ground_truth_values": conflict.get("ground_truth_values") or [],
            "conflict_ref": "::".join(
                str(part or "")
                for part in (
                    conflict.get("dataset"),
                    conflict.get("query_id"),
                    conflict.get("doc_id"),
                    conflict.get("attribute"),
                )
            ),
        }
        for conflict in sorted(
            conflicts,
            key=lambda item: (
                str(item.get("dataset") or ""),
                str(item.get("query_id") or ""),
                str(item.get("doc_id") or ""),
                str(item.get("attribute") or ""),
            ),
        )[:20]
    ]
    return {
        "id": "dataset_consistency_audit",
        "title": "Dataset Consistency Audit",
        "status": "measured",
        "message": "Reported explicit text evidence that disagrees with GT predicate outcomes.",
        "metrics": {
            "total_conflicts": len(conflicts),
            "text_pass_gt_fail": by_type.get("text_pass_gt_fail", 0),
            "gt_pass_text_fail": by_type.get("gt_pass_text_fail", 0),
            "affected_docs": len(affected_docs),
            "affected_queries": len(affected_queries),
            "checked_doc_predicates": checked_doc_predicates,
            "datasets": len(datasets_seen),
            "audit_reports": report_count,
        },
        "details": details,
    }


def _dataset_consistency_inputs_from_payload(
    run_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = run_payload.get("dataset_roots")
    if not isinstance(raw, dict):
        return {}
    inputs: dict[str, dict[str, Any]] = {}
    for dataset_id, value in raw.items():
        if isinstance(value, dict):
            root = value.get("root")
            query_ids = value.get("query_ids")
        else:
            root = value
            query_ids = None
        if not root:
            continue
        root_path = Path(str(root))
        if not root_path.is_absolute():
            root_path = resolve_repo_path(root_path)
        inputs[str(dataset_id)] = {
            "root": str(root_path),
            "query_ids": [str(query_id) for query_id in query_ids or []],
        }
    return inputs


def _common_output_dir(out_roots: Sequence[Path]) -> Path | None:
    if not out_roots:
        return None
    root = out_roots[0]
    return root.parents[2] if len(root.parents) > 2 else root.parent


def _dataset_consistency_audit_candidates(out_root: Path) -> list[Path]:
    candidates = [out_root / "dataset_consistency_audit.json"]
    try:
        candidates.extend(
            ancestor / "dataset_consistency_audit.json"
            for ancestor in out_root.parents[:4]
        )
    except IndexError:
        pass
    return candidates


def _proxy_llm_doc_call_counts(table_decision: dict[str, Any]) -> tuple[int, int]:
    """Return table-level docs before/after proxy filtering.

    Proxy stats are per predicate and can double count cascade work. The
    decision artifact's document id lists are the accurate count of documents
    that would still require the expensive extraction call.
    """
    all_doc_ids = table_decision.get("all_doc_ids")
    passed_doc_ids = table_decision.get("passed_doc_ids")
    extracted_doc_ids = table_decision.get("extracted_doc_ids")
    if isinstance(all_doc_ids, list) and isinstance(passed_doc_ids, list):
        after_ids = extracted_doc_ids if isinstance(extracted_doc_ids, list) else passed_doc_ids
        return len(set(map(str, all_doc_ids))), len(set(map(str, after_ids)))

    proxy_stats = table_decision.get("proxy_stats")
    if not isinstance(proxy_stats, dict):
        return 0, 0
    evaluated_counts: list[int] = []
    passed_counts: list[int] = []
    for stat in proxy_stats.values():
        if not isinstance(stat, dict):
            continue
        evaluated = _metric_int(stat.get("evaluated"))
        passed = _metric_int(stat.get("passed"))
        if evaluated is not None:
            evaluated_counts.append(evaluated)
        if passed is not None:
            passed_counts.append(passed)
    before = max(evaluated_counts) if evaluated_counts else 0
    after = min(passed_counts) if passed_counts else before
    return before, after


def _proxy_rejected_doc_ids(table_decision: dict[str, Any]) -> list[str]:
    all_doc_ids = table_decision.get("all_doc_ids")
    passed_doc_ids = table_decision.get("passed_doc_ids")
    if isinstance(all_doc_ids, list) and isinstance(passed_doc_ids, list):
        rejected = {str(doc_id) for doc_id in all_doc_ids} - {
            str(doc_id) for doc_id in passed_doc_ids
        }
        return sorted(rejected)
    rejected: set[str] = set()
    proxy_rejected = table_decision.get("proxy_rejected_doc_ids")
    if isinstance(proxy_rejected, dict):
        for ids in proxy_rejected.values():
            if isinstance(ids, list):
                rejected.update(str(doc_id) for doc_id in ids)
    return sorted(rejected)


def _is_gt_text_consistency_proxy(proxy_name: Any) -> bool:
    return str(proxy_name or "").startswith("gt_text_consistency_")


def _query_id_from_filter_name(name: str) -> str | None:
    match = re.match(r"(?:doc|chunk)_filter_(?P<query>.+?)(?:_.+)?\.json$", name)
    return match.group("query") if match else None


def _query_id_from_proxy_decision_name(name: str) -> str | None:
    match = re.match(r"res_tabular_data_(?P<query>.+?)_.+_proxy_decisions\.json$", name)
    return match.group("query") if match else None


def _query_id_from_adaptive_stats_name(name: str) -> str | None:
    match = re.match(r".*?(?P<query>q\d+|Q\d+).*_adaptive_stats\.json$", name)
    return match.group("query") if match else None


def _artifact_query_allowed(
    root: Path,
    query_id: str | None,
    query_ids_by_root: Mapping[Path, set[str]] | None,
) -> bool:
    if not query_ids_by_root:
        return True
    try:
        resolved_root = root.resolve()
    except OSError:
        resolved_root = root
    allowed = query_ids_by_root.get(resolved_root)
    if not allowed:
        return True
    return query_id in allowed


def _dataset_id_from_out_root(path: Path) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part in WEB_DEMO_STAGE_SET and index > 0:
            return parts[index - 1]
    return None


def _collect_join_proxy_metric(
    out_roots: Sequence[Path],
    query_ids_by_root: Mapping[Path, set[str]] | None = None,
) -> dict[str, Any]:
    files = _unique_existing_files(
        path
        for root in out_roots
        for path in root.rglob("*_proxy_decisions.json")
        if _artifact_query_allowed(
            root,
            _query_id_from_proxy_decision_name(path.name),
            query_ids_by_root,
        )
    )
    evaluated = passed = rejected = 0
    join_proxies = 0
    for path in files:
        content = _safe_read_result_json(path)
        if not isinstance(content, dict):
            continue
        for table_decision in content.values():
            if not isinstance(table_decision, dict):
                continue
            proxy_stats = table_decision.get("proxy_stats")
            if not isinstance(proxy_stats, dict):
                continue
            for proxy_name, stat in proxy_stats.items():
                if "join" not in str(proxy_name).lower():
                    continue
                if not isinstance(stat, dict):
                    continue
                join_proxies += 1
                evaluated += _metric_int(stat.get("evaluated")) or 0
                passed += _metric_int(stat.get("passed")) or 0
                rejected += _metric_int(stat.get("rejected")) or 0

    if not join_proxies:
        return {
            "id": "join_proxy",
            "title": "Join-aware Filtering",
            "status": "not_enabled",
            "message": "No join-aware proxy decisions were emitted for this run.",
            "metrics": {},
        }
    return {
        "id": "join_proxy",
        "title": "Join-aware Filtering",
        "status": "measured",
        "message": "Used extracted join keys to avoid child-table work that cannot join.",
        "metrics": {
            "join_proxies": join_proxies,
            "evaluated": evaluated,
            "passed": passed,
            "rejected": rejected,
            "pass_rate": passed / evaluated if evaluated else None,
        },
    }


def _collect_extraction_metric(
    run_payload: dict[str, Any],
    out_roots: Sequence[Path],
    query_ids_by_root: Mapping[Path, set[str]] | None = None,
) -> dict[str, Any]:
    files = _unique_existing_files(
        path
        for root in out_roots
        for path in root.rglob("res_tabular_data_*.json")
        if not path.name.endswith("_proxy_decisions.json")
        if _artifact_query_allowed(
            root,
            _query_id_from_result_name(path.name),
            query_ids_by_root,
        )
    )
    records = 0
    tables: set[str] = set()
    for path in files:
        stats = _json_result_stats(_safe_read_result_json(path))
        records += stats.get("records_count") or 0
        tables.update(stats.get("tables") or [])

    return {
        "id": "extraction",
        "title": "Extraction",
        "status": "measured" if files else "no_metrics",
        "message": (
            "Materialized extraction result artifacts."
            if files
            else "The run completed, but no extraction result artifact was found yet."
        ),
        "metrics": {
            "datasets": len(run_payload.get("datasets") or []),
            "queries": len(run_payload.get("query_ids") or []) or len(files),
            "result_files": len(files),
            "records": records,
            "tables": len(tables),
        },
    }


def _safe_read_result_json(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except Exception:
        return {}


def _unique_existing_files(paths: Any) -> list[Path]:
    seen: set[Path] = set()
    results: list[Path] = []
    for path in paths:
        if not isinstance(path, Path) or not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        results.append(resolved)
    return results


def _metric_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average(values: Sequence[float]) -> float | None:
    numeric = [float(value) for value in values]
    return sum(numeric) / len(numeric) if numeric else None


def _safe_resource_path(manifest_path: Path, paths: Any, key: str) -> Path | None:
    if not isinstance(paths, dict) or not paths.get(key):
        return None
    path = Path(str(paths[key]))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {resolved_path}")
    return data


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {resolved_path}")
    return data


def _parquet_row_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            import pandas as pd

            return int(len(pd.read_parquet(path)))
        except Exception:
            return None


def _read_parquet_records(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("`limit` must be greater than 0.")
    import pandas as pd

    frame = pd.read_parquet(path).head(limit)
    frame = frame.astype(object).where(pd.notnull(frame), None)
    return frame.to_dict(orient="records")


def _load_web_dependencies() -> dict[str, Any]:
    try:
        fastapi = importlib.import_module("fastapi")
        responses = importlib.import_module("fastapi.responses")
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The ReDD web demo requires the optional `web` dependencies. "
            'Install with `pip install -e ".[web]"` or run with `uv run --extra web redd web`.'
        ) from exc

    return {
        "Body": fastapi.Body,
        "FastAPI": fastapi.FastAPI,
        "HTTPException": fastapi.HTTPException,
        "HTMLResponse": responses.HTMLResponse,
        "JSONResponse": responses.JSONResponse,
        "FileResponse": responses.FileResponse,
        "PlainTextResponse": responses.PlainTextResponse,
        "StreamingResponse": responses.StreamingResponse,
        "uvicorn": uvicorn,
    }


def _read_web_resource(name: str) -> str:
    return (
        resources.files("redd.resources.web_demo")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


def _clean_optional_list(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        values = [str(item).strip() for item in value]
    else:
        raise TypeError("Expected a comma-separated string or list.")

    cleaned = [item for item in values if item]
    return cleaned or None
