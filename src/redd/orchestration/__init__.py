"""Runtime orchestration helpers for ReDD stages, experiments, and runners."""

from __future__ import annotations

from .experiment import (
    DEFAULT_DATASET_REGISTRY,
    dataset_manifest_path,
    dataset_registry_entry,
    load_dataset_registry,
    normalize_selection,
    select_runtime,
    select_runtime_datasets,
    select_runtime_stages,
)
from .runtime import (
    DatasetRuntimeContext,
    SchemaArtifactSource,
    build_data_loader_config,
    configure_stage_logging,
    ensure_shared_output_root,
    normalize_stage_config,
    resolve_data_root,
    resolve_dataset_contexts,
    resolve_dataset_roots,
    resolve_dataset_targets,
    resolve_log_dir,
    resolve_output_root,
    resolve_schema_artifact_source,
    resolve_stage_output_root,
    setup_runtime_logging,
    should_place_module_under_task,
)

__all__ = [
    "DEFAULT_DATASET_REGISTRY",
    "DatasetRuntimeContext",
    "SchemaArtifactSource",
    "build_data_loader_config",
    "configure_stage_logging",
    "dataset_manifest_path",
    "dataset_registry_entry",
    "ensure_shared_output_root",
    "load_dataset_registry",
    "normalize_selection",
    "normalize_stage_config",
    "resolve_data_root",
    "resolve_dataset_contexts",
    "resolve_dataset_roots",
    "resolve_dataset_targets",
    "resolve_log_dir",
    "resolve_output_root",
    "resolve_schema_artifact_source",
    "resolve_stage_output_root",
    "select_runtime",
    "select_runtime_datasets",
    "select_runtime_stages",
    "setup_runtime_logging",
    "should_place_module_under_task",
]
