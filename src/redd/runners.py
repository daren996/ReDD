"""CLI orchestration helpers.

These functions are retained for backwards compatibility with repository
scripts. New external integrations should use `redd.SchemaGenerator`,
`redd.DataPopulator`, and `redd.run_pipeline`.
"""

from __future__ import annotations

from redd.api import DataPopulator, SchemaGenerator
from redd.config import load_experiment_config
from redd.runtime import resolve_dataset_targets, setup_runtime_logging
from redd.core.data_population import create_data_populator
from redd.core.schema_gen import create_schema_generator


def _build_schema_generator(config: dict, api_key: str | None = None):
    return create_schema_generator(config, api_key=api_key)


def _build_datapopulator(config: dict, api_key: str | None = None):
    return create_data_populator(config, api_key=api_key)


def run_preprocessing(config_path: str, exp: str, api_key: str | None = None) -> dict:
    config, _ = load_experiment_config(config_path, exp, module="schemagen")
    setup_runtime_logging(config, exp)
    schema_generator = SchemaGenerator(
        preprocessing_config=config,
        api_key=api_key,
        configure_logging=False,
    )
    schema_generator.preprocessing(datasets=resolve_dataset_targets(config))
    return config


def run_schema_refinement(config_path: str, exp: str, api_key: str | None = None) -> dict:
    config, _ = load_experiment_config(config_path, exp, module="schemagen")
    setup_runtime_logging(config, exp)
    schema_generator = SchemaGenerator(
        refinement_config=config,
        api_key=api_key,
        configure_logging=False,
    )
    schema_generator.schema_refine(datasets=resolve_dataset_targets(config))
    return config


def run_schemagen(config_path: str, exp: str, api_key: str | None = None) -> dict:
    config, _ = load_experiment_config(config_path, exp, module="schemagen")
    setup_runtime_logging(config, exp)
    schema_gen = _build_schema_generator(config, api_key=api_key)
    schema_gen(resolve_dataset_targets(config))
    return config


def run_datapop(config_path: str, exp: str, api_key: str | None = None) -> dict:
    config, _ = load_experiment_config(config_path, exp, module="datapop")
    setup_runtime_logging(config, exp)
    data_populator = DataPopulator(
        config,
        api_key=api_key,
        configure_logging=False,
    )
    data_populator.data_extraction(datasets=resolve_dataset_targets(config))
    return config


def run_evaluation(config_path: str, exp: str, api_key: str | None = None) -> dict:
    return run_datapop_evaluation(config_path, exp, api_key=api_key)


def run_datapop_evaluation(config_path: str, exp: str, api_key: str | None = None) -> dict:
    from redd.exp.evaluation import EvalDataPop
    from redd.llm import is_local_provider, normalize_provider_name

    config, _ = load_experiment_config(config_path, exp, module="datapop")
    setup_runtime_logging(config, exp)

    eval_mode = normalize_provider_name(config["eval"]["mode"])
    if "committee" not in config["eval"] and is_local_provider(eval_mode):
        raise ValueError(f"Invalid eval mode `{eval_mode}`")

    evaluator = EvalDataPop(config, api_key=api_key)
    evaluator(resolve_dataset_targets(config))
    return config


def run_ensemble_classifiers(config_path: str, exp: str) -> dict:
    from redd.correction import ClassifierVal

    config, _ = load_experiment_config(config_path, exp, module="correction")
    setup_runtime_logging(config, exp)

    validator = ClassifierVal(config)
    validator(config["model_dn_fn_list"], config["test_dn_fn"], test_mode="ensemble")
    return config
