from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ExperimentRuntime, StageName, load_experiment_runtime
from .core.data_loader import DataLoaderBase
from .loader import create_data_loader
from .runtime import (
    configure_stage_logging,
    ensure_shared_output_root,
    normalize_stage_config,
)
from .stages import data_extraction as data_extraction_stage
from .stages import schema as schema_stage


class PipelineStage(str, Enum):
    PREPROCESSING = "preprocessing"
    SCHEMA_REFINEMENT = "schema_refinement"
    DATA_EXTRACTION = "data_extraction"


PREPROCESSING = PipelineStage.PREPROCESSING
SCHEMA_REFINEMENT = PipelineStage.SCHEMA_REFINEMENT
DATA_EXTRACTION = PipelineStage.DATA_EXTRACTION


def _coerce_stage(stage: PipelineStage | str) -> PipelineStage:
    if isinstance(stage, PipelineStage):
        return stage

    normalized = str(stage).strip().lower().replace(" ", "_")
    return PipelineStage(normalized)


def _has_query_input(config: Mapping[str, Any]) -> bool:
    return schema_stage.has_query_input(config)


def _stage_config_from_runtime(runtime: ExperimentRuntime, stage: StageName) -> dict[str, Any] | None:
    if stage not in runtime.stages or not runtime.stages[stage].enabled:
        return None
    return runtime.stage_runtime_dict(stage)


def _build_schema_generator_impl(config: Mapping[str, Any], api_key: str | None = None):
    return schema_stage.build_schema_generator_impl(config, api_key=api_key)


def _build_data_populator_impl(config: Mapping[str, Any], api_key: str | None = None):
    return data_extraction_stage.build_data_populator_impl(config, api_key=api_key)


def _build_doc_dict(loader: Any) -> dict[str, list[str]]:
    return schema_stage.build_doc_dict(loader)


def _create_loader_for_impl(impl: Any, data_root: str | Path):
    return schema_stage.create_loader_for_impl(impl, data_root)


class SchemaGenerator:
    """Stable public entry point for schema-generation stages."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        preprocessing_config: Mapping[str, Any] | None = None,
        refinement_config: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        configure_logging: bool = True,
    ) -> None:
        if config is not None:
            copied = normalize_stage_config(config, module="schemagen") or {}
            if _has_query_input(copied):
                refinement_config = copied
            else:
                preprocessing_config = copied

        self.preprocessing_config = normalize_stage_config(preprocessing_config, module="schemagen")
        self.refinement_config = normalize_stage_config(refinement_config, module="schemagen")
        self.api_key = api_key

        if self.preprocessing_config is None and self.refinement_config is None:
            raise ValueError("SchemaGenerator requires at least one schema config.")

        if self.preprocessing_config and self.refinement_config:
            try:
                ensure_shared_output_root(self.preprocessing_config, self.refinement_config)
            except ValueError as exc:
                raise ValueError(
                    "PREPROCESSING and SCHEMA REFINEMENT currently need the same output root "
                    "so refinement can reuse preprocessing artifacts."
                ) from exc

        if configure_logging:
            base_config = self.refinement_config or self.preprocessing_config or {}
            configure_stage_logging(base_config)

    @classmethod
    def from_experiment(
        cls,
        config_path: str | Path,
        exp: str,
        *,
        api_key: str | None = None,
        configure_logging: bool = True,
    ) -> "SchemaGenerator":
        runtime, _ = load_experiment_runtime(config_path, exp)
        return cls(
            preprocessing_config=_stage_config_from_runtime(runtime, "preprocessing"),
            refinement_config=_stage_config_from_runtime(runtime, "schema_refinement"),
            api_key=api_key,
            configure_logging=configure_logging,
        )

    def run(self, datasets: Sequence[str] | None = None) -> dict[str, Any]:
        return schema_stage.run_schema(self, datasets=datasets)

    def schema_global(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Alias for query-independent schema extraction."""
        return self.preprocessing(datasets=datasets)

    def preprocessing(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        return schema_stage.run_schema_preprocessing(self, datasets=datasets)

    def schema_refine(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        return schema_stage.run_schema_refinement(self, datasets=datasets)

    def schema_refinement(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible alias for `schema_refine`."""
        return self.schema_refine(datasets=datasets)

    def _schema_refine_with_schema_tailor(
        self,
        config: Mapping[str, Any],
        *,
        datasets: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        return schema_stage.run_schema_refinement_with_schema_tailor(
            self,
            config,
            datasets=datasets,
        )


class DataPopulator:
    """Stable public entry point for the DATA EXTRACTION stage."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        api_key: str | None = None,
        configure_logging: bool = True,
    ) -> None:
        self.config = normalize_stage_config(config, module="data_extraction") or {}
        self.api_key = api_key

        if configure_logging:
            configure_stage_logging(self.config)

    @classmethod
    def from_experiment(
        cls,
        config_path: str | Path,
        exp: str,
        *,
        api_key: str | None = None,
        configure_logging: bool = True,
    ) -> "DataPopulator":
        runtime, _ = load_experiment_runtime(config_path, exp)
        config = _stage_config_from_runtime(runtime, "data_extraction")
        if config is None:
            raise ValueError(f"Experiment `{exp}` does not enable data_extraction.")
        return cls(config, api_key=api_key, configure_logging=configure_logging)

    def run(
        self,
        datasets: Sequence[str] | None = None,
        *,
        schema_generator: SchemaGenerator | None = None,
        schema_config: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.data_extraction(
            datasets=datasets,
            schema_generator=schema_generator,
            schema_config=schema_config,
        )

    def data_extraction(
        self,
        datasets: Sequence[str] | None = None,
        *,
        schema_generator: SchemaGenerator | None = None,
        schema_config: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return data_extraction_stage.run_data_extraction(
            self,
            datasets=datasets,
            schema_generator=schema_generator,
            schema_config=schema_config,
        )

    @staticmethod
    def _resolve_schema_source_mode(config: Mapping[str, Any]) -> str:
        return data_extraction_stage.resolve_schema_source_mode(config)

    def _resolve_general_schema_source(
        self,
        *,
        schema_generator: SchemaGenerator | None,
        schema_config: Mapping[str, Any] | None,
    ) -> Any | None:
        return data_extraction_stage.resolve_general_schema_source(
            schema_generator=schema_generator,
            schema_config=schema_config,
        )

    def _resolve_query_schema_source(
        self,
        *,
        schema_generator: SchemaGenerator | None,
        schema_config: Mapping[str, Any] | None,
    ) -> Any | None:
        return data_extraction_stage.resolve_query_schema_source(
            schema_generator=schema_generator,
            schema_config=schema_config,
        )

    def _build_loader_config(
        self,
        *,
        schema_source: str,
        base_loader_config: Mapping[str, Any] | None,
        dataset: str,
        general_schema_source: Any | None,
        query_schema_source: Any | None,
    ) -> dict[str, Any]:
        return data_extraction_stage.build_loader_config(
            schema_source=schema_source,
            base_loader_config=base_loader_config,
            dataset=dataset,
            general_schema_source=general_schema_source,
            query_schema_source=query_schema_source,
        )


def run_pipeline(
    *,
    schema_generator: SchemaGenerator | None = None,
    data_populator: DataPopulator | None = None,
    stages: Sequence[PipelineStage | str] | None = None,
    datasets: Sequence[str] | None = None,
) -> dict[str, Any]:
    from .stages.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(
        schema_generator=schema_generator,
        data_populator=data_populator,
        stages=stages,
        datasets=datasets,
        stage_type=PipelineStage,
    )


__all__ = [
    "DataLoaderBase",
    "DataPopulator",
    "PipelineStage",
    "PREPROCESSING",
    "SCHEMA_REFINEMENT",
    "DATA_EXTRACTION",
    "SchemaGenerator",
    "create_data_loader",
    "run_pipeline",
]
