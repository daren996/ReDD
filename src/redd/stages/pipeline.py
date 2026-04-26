"""Pipeline orchestration across public ReDD stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from redd.api import DataPopulator, PipelineStage, SchemaGenerator

__all__ = ["run_pipeline"]


def run_pipeline(
    *,
    schema_generator: SchemaGenerator | None = None,
    data_populator: DataPopulator | None = None,
    stages: Sequence[PipelineStage | str] | None = None,
    datasets: Sequence[str] | None = None,
    stage_type: type[PipelineStage],
) -> dict[str, Any]:
    preprocessing = stage_type.PREPROCESSING
    schema_refinement = stage_type.SCHEMA_REFINEMENT
    data_extraction = stage_type.DATA_EXTRACTION

    if stages is None:
        resolved_stages = []
        if schema_generator and schema_generator.preprocessing_config is not None:
            resolved_stages.append(preprocessing)
        if schema_generator and schema_generator.refinement_config is not None:
            resolved_stages.append(schema_refinement)
        if data_populator:
            resolved_stages.append(data_extraction)

        if not resolved_stages:
            raise ValueError("run_pipeline requires at least one pipeline component.")
    else:
        resolved_stages = [_coerce_stage(stage, stage_type=stage_type) for stage in stages]

    results: dict[str, Any] = {}

    for stage in resolved_stages:
        if stage is preprocessing:
            if schema_generator is None:
                raise ValueError("PREPROCESSING requires `schema_generator=`.")
            results[stage.value] = schema_generator.preprocessing(datasets=datasets)
        elif stage is schema_refinement:
            if schema_generator is None:
                raise ValueError("SCHEMA REFINEMENT requires `schema_generator=`.")
            results[stage.value] = schema_generator.schema_refinement(datasets=datasets)
        elif stage is data_extraction:
            if data_populator is None:
                raise ValueError("DATA EXTRACTION requires `data_populator=`.")
            results[stage.value] = data_populator.data_extraction(
                datasets=datasets,
                schema_generator=schema_generator,
            )

    return results


def _coerce_stage(stage: PipelineStage | str, *, stage_type: type[PipelineStage]) -> PipelineStage:
    if isinstance(stage, stage_type):
        return stage

    normalized = str(stage).strip().lower().replace(" ", "_")
    return stage_type(normalized)
