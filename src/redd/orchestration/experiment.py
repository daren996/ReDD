"""Experiment runtime selection helpers shared by CLI, API, and web flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import yaml

from redd.config import (
    DatasetRuntimeConfig,
    DatasetSplitConfig,
    ExperimentRuntime,
    StageName,
    resolve_repo_path,
)

DEFAULT_DATASET_REGISTRY = "dataset/manifest.yaml"


def normalize_selection(values: Sequence[str] | str | None) -> list[str] | None:
    """Normalize comma-separated or repeated CLI/API selection values."""

    if values is None:
        return None
    raw_values = [values] if isinstance(values, str) else list(values)
    normalized = [
        item.strip()
        for value in raw_values
        for item in str(value).split(",")
        if item.strip()
    ]
    return normalized or None


def select_runtime(
    runtime: ExperimentRuntime,
    *,
    datasets: Sequence[str] | str | None = None,
    query_ids: Sequence[str] | str | None = None,
    stages: Sequence[StageName | str] | StageName | str | None = None,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> ExperimentRuntime:
    """Return a runtime narrowed to requested datasets, query IDs, and stages."""

    selected = select_runtime_datasets(
        runtime,
        datasets=datasets,
        query_ids=query_ids,
        registry_path=registry_path,
    )
    return select_runtime_stages(selected, stages=stages)


def select_runtime_datasets(
    runtime: ExperimentRuntime,
    *,
    datasets: Sequence[str] | str | None = None,
    query_ids: Sequence[str] | str | None = None,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> ExperimentRuntime:
    """Return a runtime narrowed to requested datasets and optional query IDs.

    Dataset IDs may come from the experiment config or from the canonical
    dataset registry, which makes the same selection behavior available to both
    the web demo and command-line/API runners.
    """

    requested_datasets = normalize_selection(datasets) or runtime.dataset_ids()
    requested_query_ids = normalize_selection(query_ids)
    selected: dict[str, DatasetRuntimeConfig] = {}
    registry: dict[str, Any] | None = None
    resolved_registry_path: Path | None = None

    for dataset_id in requested_datasets:
        if dataset_id in runtime.datasets:
            dataset_config = runtime.datasets[dataset_id]
            if requested_query_ids:
                dataset_config = dataset_config.model_copy(
                    update={"query_ids": list(requested_query_ids)}
                )
            selected[dataset_id] = dataset_config
            continue

        if registry is None or resolved_registry_path is None:
            registry, resolved_registry_path = load_dataset_registry(registry_path)
        try:
            entry = dataset_registry_entry(registry, dataset_id)
        except KeyError as exc:
            raise ValueError(
                f"Dataset `{dataset_id}` is neither in experiment `{runtime.id}` "
                f"nor registered in {resolved_registry_path}."
            ) from exc

        selected[dataset_id] = DatasetRuntimeConfig(
            id=dataset_id,
            root=dataset_manifest_path(entry, resolved_registry_path).parent,
            loader="hf_manifest",
            query_ids=list(requested_query_ids) if requested_query_ids else None,
            split=DatasetSplitConfig(train_count=0),
            loader_options={"manifest": "manifest.yaml"},
        )

    if not selected:
        raise ValueError("Runtime selection contains no datasets.")
    return runtime.model_copy(update={"datasets": selected})


def select_runtime_stages(
    runtime: ExperimentRuntime,
    *,
    stages: Sequence[StageName | str] | StageName | str | None = None,
) -> ExperimentRuntime:
    """Return a runtime narrowed to requested enabled stages."""

    requested = normalize_selection(cast(Sequence[str] | str | None, stages))
    if requested is None:
        return runtime

    stage_order: list[StageName] = []
    for stage in requested:
        normalized = stage.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == "refine":
            normalized = "schema_refinement"
        elif normalized in {"extract", "data"}:
            normalized = "data_extraction"
        elif normalized == "preprocess":
            normalized = "preprocessing"
        if normalized not in runtime.stages:
            raise ValueError(f"Stage `{stage}` is not configured for experiment `{runtime.id}`.")
        stage_name = cast(StageName, normalized)
        if not runtime.stages[stage_name].enabled:
            raise ValueError(f"Stage `{stage_name}` is disabled for experiment `{runtime.id}`.")
        stage_order.append(stage_name)

    if not stage_order:
        raise ValueError("Runtime selection contains no stages.")
    return runtime.model_copy(
        update={
            "stages": {stage: runtime.stages[stage] for stage in stage_order},
            "stage_order": stage_order,
        }
    )


def load_dataset_registry(
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
) -> tuple[dict[str, Any], Path]:
    resolved_path = resolve_repo_path(registry_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    datasets = registry.get("datasets") or {}
    if not isinstance(datasets, dict):
        raise TypeError("Dataset registry `datasets` must be a mapping.")
    return registry, resolved_path


def dataset_registry_entry(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    datasets = registry.get("datasets") or {}
    entry = datasets.get(dataset_id)
    if not isinstance(entry, dict):
        raise KeyError(dataset_id)
    return entry


def dataset_manifest_path(entry: dict[str, Any], registry_path: Path) -> Path:
    raw_path = entry.get("path")
    if not raw_path:
        raise ValueError("Dataset registry entry is missing `path`.")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (registry_path.parent / path).resolve()


__all__ = [
    "DEFAULT_DATASET_REGISTRY",
    "dataset_manifest_path",
    "dataset_registry_entry",
    "load_dataset_registry",
    "normalize_selection",
    "select_runtime",
    "select_runtime_datasets",
    "select_runtime_stages",
]
