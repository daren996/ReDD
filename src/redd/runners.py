"""Strict v2 config orchestration helpers for the package CLI."""

from __future__ import annotations

from redd.api import DataPopulator, SchemaGenerator, run_pipeline
from redd.config import ExperimentRuntime, StageName, load_experiment_runtime
from redd.runtime import setup_runtime_logging


def load_runtime(config_path: str, experiment: str) -> ExperimentRuntime:
    runtime, _ = load_experiment_runtime(config_path, experiment)
    return runtime


def _dataset_ids(runtime: ExperimentRuntime) -> list[str]:
    return runtime.dataset_ids()


def _schema_generator(runtime: ExperimentRuntime, api_key: str | None = None) -> SchemaGenerator | None:
    preprocessing_config = (
        runtime.stage_runtime_dict("preprocessing")
        if "preprocessing" in runtime.stages and runtime.stages["preprocessing"].enabled
        else None
    )
    refinement_config = (
        runtime.stage_runtime_dict("schema_refinement")
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


def _data_populator(runtime: ExperimentRuntime, api_key: str | None = None) -> DataPopulator | None:
    if "data_extraction" not in runtime.stages or not runtime.stages["data_extraction"].enabled:
        return None
    return DataPopulator(
        runtime.stage_runtime_dict("data_extraction"),
        api_key=api_key,
        configure_logging=False,
    )


def _setup_logging(runtime: ExperimentRuntime) -> None:
    first_stage: StageName = runtime.stage_order[0]
    log_name = f"{runtime.project.name}.{runtime.id}"
    setup_runtime_logging(runtime.stage_runtime_dict(first_stage), log_name)


def run_preprocessing(config_path: str, experiment: str, api_key: str | None = None) -> dict:
    runtime = load_runtime(config_path, experiment)
    _setup_logging(runtime)
    schema_generator = _schema_generator(runtime, api_key=api_key)
    if schema_generator is None or schema_generator.preprocessing_config is None:
        raise ValueError(f"Experiment `{experiment}` does not enable preprocessing.")
    result = schema_generator.preprocessing(datasets=_dataset_ids(runtime))
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "preprocessing",
        "result": result,
    }


def run_schema_refinement(config_path: str, experiment: str, api_key: str | None = None) -> dict:
    runtime = load_runtime(config_path, experiment)
    _setup_logging(runtime)
    schema_generator = _schema_generator(runtime, api_key=api_key)
    if schema_generator is None or schema_generator.refinement_config is None:
        raise ValueError(f"Experiment `{experiment}` does not enable schema_refinement.")
    result = schema_generator.schema_refine(datasets=_dataset_ids(runtime))
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "schema_refinement",
        "result": result,
    }


def run_extract(config_path: str, experiment: str, api_key: str | None = None) -> dict:
    runtime = load_runtime(config_path, experiment)
    _setup_logging(runtime)
    data_populator = _data_populator(runtime, api_key=api_key)
    if data_populator is None:
        raise ValueError(f"Experiment `{experiment}` does not enable data_extraction.")
    schema_generator = _schema_generator(runtime, api_key=api_key)
    result = data_populator.data_extraction(
        datasets=_dataset_ids(runtime),
        schema_generator=schema_generator,
    )
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "data_extraction",
        "result": result,
    }


def run_experiment(config_path: str, experiment: str, api_key: str | None = None) -> dict:
    runtime = load_runtime(config_path, experiment)
    _setup_logging(runtime)
    schema_generator = _schema_generator(runtime, api_key=api_key)
    data_populator = _data_populator(runtime, api_key=api_key)
    result = run_pipeline(
        schema_generator=schema_generator,
        data_populator=data_populator,
        stages=runtime.stage_order,
        datasets=_dataset_ids(runtime),
    )
    return {"project": runtime.project.name, "experiment": runtime.id, "result": result}


def run_evaluation(config_path: str, experiment: str, api_key: str | None = None) -> dict:
    del api_key
    from redd.exp.evaluation import EvalDataExtraction

    runtime = load_runtime(config_path, experiment)
    _setup_logging(runtime)
    if "data_extraction" not in runtime.stages or not runtime.stages["data_extraction"].enabled:
        raise ValueError(f"Experiment `{experiment}` does not enable data_extraction.")

    evaluator = EvalDataExtraction(runtime.stage_runtime_dict("data_extraction"))
    evaluator(datasets := _dataset_ids(runtime))
    return {
        "project": runtime.project.name,
        "experiment": runtime.id,
        "stage": "evaluation",
        "datasets": datasets,
    }


def run_ensemble_classifiers(config_path: str, experiment: str) -> dict:
    del config_path, experiment
    raise NotImplementedError("Correction workflows are outside the strict v2 runtime path.")
