"""Strict v2 config orchestration helpers for the package CLI."""

from __future__ import annotations

from typing import Any, Sequence

from redd.api import DataExtractor, SchemaGenerator, run_pipeline
from redd.config import ExperimentRuntime, StageName, load_experiment_runtime
from redd.orchestration.experiment import select_runtime, select_runtime_datasets
from redd.orchestration.runtime import setup_runtime_logging

__all__ = [
    "load_runtime",
    "run_evaluation",
    "run_experiment",
    "run_extract",
    "run_preprocessing",
    "run_schema_refinement",
]


def load_runtime(config_path: str, experiment: str) -> ExperimentRuntime:
    runtime, _ = load_experiment_runtime(config_path, experiment)
    return runtime


def _dataset_ids(runtime: ExperimentRuntime) -> list[str]:
    return runtime.dataset_ids()


def _query_ids(runtime: ExperimentRuntime) -> list[str]:
    query_ids: list[str] = []
    seen: set[str] = set()
    for dataset in runtime.datasets.values():
        for query_id in dataset.query_ids or []:
            value = str(query_id)
            if value not in seen:
                seen.add(value)
                query_ids.append(value)
    return query_ids


def _stage_config(runtime: ExperimentRuntime, stage: StageName) -> dict[str, Any]:
    config = runtime.stage_runtime_dict(stage)
    query_ids = _query_ids(runtime)
    if query_ids:
        config["exp_query_id_list"] = query_ids
    return config


def _schema_generator(runtime: ExperimentRuntime, api_key: str | None = None) -> SchemaGenerator | None:
    preprocessing_config = (
        _stage_config(runtime, "preprocessing")
        if "preprocessing" in runtime.stages and runtime.stages["preprocessing"].enabled
        else None
    )
    refinement_config = (
        _stage_config(runtime, "schema_refinement")
        if "schema_refinement" in runtime.stages and runtime.stages["schema_refinement"].enabled
        else None
    )
    if preprocessing_config is None and refinement_config is None:
        return None
    return SchemaGenerator(
        preprocessing_config=preprocessing_config,
        refinement_config=refinement_config,
        api_key=api_key,
        configure_logging=False,
    )


def _data_extractor(runtime: ExperimentRuntime, api_key: str | None = None) -> DataExtractor | None:
    if "data_extraction" not in runtime.stages or not runtime.stages["data_extraction"].enabled:
        return None
    return DataExtractor(
        _stage_config(runtime, "data_extraction"),
        api_key=api_key,
        configure_logging=False,
    )


def _setup_logging(runtime: ExperimentRuntime) -> None:
    first_stage: StageName = runtime.stage_order[0]
    log_name = f"{runtime.project.name}.{runtime.id}"
    setup_runtime_logging(_stage_config(runtime, first_stage), log_name)


def run_preprocessing(
    config_path: str,
    experiment: str,
    api_key: str | None = None,
    *,
    datasets: Sequence[str] | str | None = None,
    query_ids: Sequence[str] | str | None = None,
) -> dict:
    runtime = load_runtime(config_path, experiment)
    runtime = select_runtime_datasets(runtime, datasets=datasets, query_ids=query_ids)
    _setup_logging(runtime)
    schema_generator = _schema_generator(runtime, api_key=api_key)
    if schema_generator is None or schema_generator.preprocessing_config is None:
        raise ValueError(f"Experiment `{experiment}` does not enable preprocessing.")
    result = schema_generator.preprocessing(datasets=_dataset_ids(runtime))
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "preprocessing",
        "datasets": _dataset_ids(runtime),
        "query_ids": _query_ids(runtime),
        "result": result,
    }


def run_schema_refinement(
    config_path: str,
    experiment: str,
    api_key: str | None = None,
    *,
    datasets: Sequence[str] | str | None = None,
    query_ids: Sequence[str] | str | None = None,
) -> dict:
    runtime = load_runtime(config_path, experiment)
    runtime = select_runtime_datasets(runtime, datasets=datasets, query_ids=query_ids)
    _setup_logging(runtime)
    schema_generator = _schema_generator(runtime, api_key=api_key)
    if schema_generator is None or schema_generator.refinement_config is None:
        raise ValueError(f"Experiment `{experiment}` does not enable schema_refinement.")
    result = schema_generator.schema_refine(datasets=_dataset_ids(runtime))
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "schema_refinement",
        "datasets": _dataset_ids(runtime),
        "query_ids": _query_ids(runtime),
        "result": result,
    }


def run_extract(
    config_path: str,
    experiment: str,
    api_key: str | None = None,
    *,
    datasets: Sequence[str] | str | None = None,
    query_ids: Sequence[str] | str | None = None,
) -> dict:
    runtime = load_runtime(config_path, experiment)
    runtime = select_runtime_datasets(runtime, datasets=datasets, query_ids=query_ids)
    _setup_logging(runtime)
    data_extractor = _data_extractor(runtime, api_key=api_key)
    if data_extractor is None:
        raise ValueError(f"Experiment `{experiment}` does not enable data_extraction.")
    schema_generator = _schema_generator(runtime, api_key=api_key)
    result = data_extractor.data_extraction(
        datasets=_dataset_ids(runtime),
        schema_generator=schema_generator,
    )
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "data_extraction",
        "datasets": _dataset_ids(runtime),
        "query_ids": _query_ids(runtime),
        "result": result,
    }


def run_experiment(
    config_path: str,
    experiment: str,
    api_key: str | None = None,
    *,
    datasets: Sequence[str] | str | None = None,
    query_ids: Sequence[str] | str | None = None,
    stages: Sequence[StageName | str] | StageName | str | None = None,
) -> dict:
    runtime = load_runtime(config_path, experiment)
    runtime = select_runtime(runtime, datasets=datasets, query_ids=query_ids, stages=stages)
    _setup_logging(runtime)
    schema_generator = _schema_generator(runtime, api_key=api_key)
    data_extractor = _data_extractor(runtime, api_key=api_key)
    result = run_pipeline(
        schema_generator=schema_generator,
        data_extractor=data_extractor,
        stages=runtime.stage_order,
        datasets=_dataset_ids(runtime),
    )
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "datasets": _dataset_ids(runtime),
        "query_ids": _query_ids(runtime),
        "stages": list(runtime.stage_order),
        "result": result,
    }


def run_evaluation(
    config_path: str,
    experiment: str,
    api_key: str | None = None,
    *,
    datasets: Sequence[str] | str | None = None,
    query_ids: Sequence[str] | str | None = None,
) -> dict:
    from redd.exp.evaluation import EvalDataExtraction

    runtime = load_runtime(config_path, experiment)
    runtime = select_runtime_datasets(runtime, datasets=datasets, query_ids=query_ids)
    _setup_logging(runtime)
    if "data_extraction" not in runtime.stages or not runtime.stages["data_extraction"].enabled:
        raise ValueError(f"Experiment `{experiment}` does not enable data_extraction.")

    evaluator = EvalDataExtraction(_stage_config(runtime, "data_extraction"), api_key=api_key)
    evaluator(datasets := _dataset_ids(runtime))
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "evaluation",
        "datasets": datasets,
        "query_ids": _query_ids(runtime),
    }
