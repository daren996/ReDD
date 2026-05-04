from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_VERSION = "2.1.1"
DEFAULT_DOC_FILTER_THRESHOLD = 0.585
DEFAULT_DOC_FILTER_ENABLE_CALIBRATE = False
DEFAULT_PROXY_RUNTIME_MODE = "pretrained"
DEFAULT_PROXY_THRESHOLD = 0.51

StageName = Literal["preprocessing", "schema_refinement", "data_extraction"]

API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together": "TOGETHER_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    import yaml

    resolved_path = resolve_repo_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping, got {type(config).__name__}")
    return config, resolved_path


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    seed: int = 42


class RuntimeConfig(StrictModel):
    output_dir: str | Path
    artifact_id: str
    log_dir: str | Path = DEFAULT_LOG_DIR
    output_layout: Literal["dataset_stage"] = "dataset_stage"
    console_log_level: str = "WARNING"
    force_rerun: bool = False


class LLMConfigModel(StrictModel):
    provider: Literal["openai", "deepseek", "together", "siliconflow", "gemini", "local", "none"]
    model: str
    api_key_env: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    structured_backend: Literal["auto", "instructor", "json"] = "auto"
    max_retries: int = 5
    wait_time: float = 10.0
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    local_model_path: str | None = None


class EmbeddingConfigModel(StrictModel):
    provider: Literal["openai", "deepseek", "together", "siliconflow", "gemini", "local", "none"]
    model: str
    enabled: bool = True
    api_key_env: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 100
    storage_file: str = "embeddings.sqlite3"


class ModelsConfig(StrictModel):
    llm: LLMConfigModel | None = None
    embedding: EmbeddingConfigModel | None = None


class DatasetSplitConfig(StrictModel):
    train_count: int | None = None


class DatasetConfig(StrictModel):
    loader: str = "hf_manifest"
    root: str | Path
    query_ids: list[str] | None = None
    split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
    loader_options: dict[str, Any] = Field(default_factory=dict)


class StrategyConfig(StrictModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    options: dict[str, Any] = Field(default_factory=dict)

    def as_internal_dict(self) -> dict[str, Any]:
        extras = deepcopy(self.model_extra or {})
        return {"enabled": self.enabled, **deepcopy(self.options), **extras}


class StageConfig(StrictModel):
    enabled: bool = True
    artifact_id: str | None = None
    prompt: str | dict[str, Any] | None = None
    prompts: dict[str, str] | None = None
    input_fields: dict[str, str] | None = None
    output_fields: dict[str, str] | None = None
    source_stage: StageName | None = None
    schema_source: str | None = None
    oracle: Literal["llm", "ground_truth"] = "llm"
    data_loader: dict[str, Any] = Field(default_factory=dict)
    embedding: StrategyConfig | None = None
    retrieval: StrategyConfig | None = None
    adaptive_sampling: StrategyConfig | None = None
    document_filtering: StrategyConfig | None = None
    table_assignment_cache: StrategyConfig | None = None
    proxy_runtime: StrategyConfig | None = None
    alpha_allocation: StrategyConfig | None = None
    schema_tailoring: StrategyConfig | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(StrictModel):
    datasets: list[str]
    stages: list[StageName]
    artifact_id: str | None = None

    @field_validator("datasets", "stages")
    @classmethod
    def _non_empty_list(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must not be empty")
        return value


class ReDDConfig(StrictModel):
    config_version: Literal["2.1.1"]
    project: ProjectConfig
    runtime: RuntimeConfig
    models: ModelsConfig
    datasets: dict[str, DatasetConfig]
    stages: dict[StageName, StageConfig]
    experiments: dict[str, ExperimentConfig]

    @model_validator(mode="after")
    def _validate_references(self) -> "ReDDConfig":
        for experiment_id, experiment in self.experiments.items():
            missing_datasets = [dataset for dataset in experiment.datasets if dataset not in self.datasets]
            if missing_datasets:
                raise ValueError(
                    f"Experiment `{experiment_id}` references unknown datasets: {missing_datasets}"
                )
            missing_stages = [stage for stage in experiment.stages if stage not in self.stages]
            if missing_stages:
                raise ValueError(
                    f"Experiment `{experiment_id}` references unknown stages: {missing_stages}"
                )
        return self


class DatasetRuntimeConfig(StrictModel):
    id: str
    root: Path
    loader: str
    query_ids: list[str] | None = None
    split: DatasetSplitConfig
    loader_options: dict[str, Any] = Field(default_factory=dict)


class ExperimentRuntime(StrictModel):
    id: str
    project: ProjectConfig
    runtime: RuntimeConfig
    models: ModelsConfig
    datasets: dict[str, DatasetRuntimeConfig]
    stages: dict[StageName, StageConfig]
    stage_order: list[StageName]

    @property
    def artifact_id(self) -> str:
        return self.runtime.artifact_id

    def dataset_ids(self) -> list[str]:
        return list(self.datasets)

    def stage_artifact_id(self, stage: StageName) -> str:
        return self.stages[stage].artifact_id or self.runtime.artifact_id

    def stage_output_root(self, dataset_id: str, stage: StageName) -> Path:
        if self.runtime.output_layout != "dataset_stage":
            raise ValueError(f"Unsupported output layout `{self.runtime.output_layout}`")
        return (
            resolve_repo_path(self.runtime.output_dir)
            / dataset_id
            / stage
            / self.stage_artifact_id(stage)
        )

    def schema_output_root(self, dataset_id: str, source_stage: StageName | None = None) -> Path:
        stage = source_stage or "schema_refinement"
        return self.stage_output_root(dataset_id, stage)

    def stage_runtime_dict(self, stage: StageName) -> dict[str, Any]:
        if stage not in self.stages:
            raise KeyError(f"Stage `{stage}` is not configured for experiment `{self.id}`")
        stage_config = self.stages[stage]
        if not stage_config.enabled:
            raise ValueError(f"Stage `{stage}` is disabled for experiment `{self.id}`")

        llm = self.models.llm
        first_dataset = next(iter(self.datasets.values()))
        config: dict[str, Any] = {
            "project_name": self.project.name,
            "project_seed": self.project.seed,
            "exp_name": self.id,
            "stage": stage,
            "artifact_id": self.stage_artifact_id(stage),
            "res_param_str": self.stage_artifact_id(stage),
            "log_dir": str(resolve_repo_path(self.runtime.log_dir)),
            "console_log_level": self.runtime.console_log_level,
            "out_main": str(resolve_repo_path(self.runtime.output_dir)),
            "data_main": str(first_dataset.root.parent),
            "exp_dn_fn_list": self.dataset_ids(),
            "datasets": self.dataset_ids(),
            "_runtime_contexts": [
                {
                    "dataset": dataset_id,
                    "data_root": str(dataset.root),
                    "out_root": str(self.stage_output_root(dataset_id, stage)),
                    "query_ids": dataset.query_ids,
                }
                for dataset_id, dataset in self.datasets.items()
            ],
            "data_loader_type": first_dataset.loader,
            "data_loader_config": deepcopy(first_dataset.loader_options),
            **deepcopy(stage_config.options),
        }
        if first_dataset.split.train_count is not None:
            config["training_data_count"] = first_dataset.split.train_count

        if llm is not None and llm.provider != "none":
            config.update(
                {
                    "mode": llm.provider,
                    "llm_model": llm.model,
                    "structured_backend": llm.structured_backend,
                    "max_retries": llm.max_retries,
                    "wait_time": llm.wait_time,
                }
            )
            if llm.api_key:
                config["api_key"] = llm.api_key
            if llm.api_key_env and os.getenv(llm.api_key_env):
                config["api_key"] = os.getenv(llm.api_key_env)
            if llm.base_url:
                config["base_url"] = llm.base_url
            if llm.local_model_path:
                config["llm_model_path"] = llm.local_model_path
            for key in ("temperature", "top_p", "max_tokens"):
                value = getattr(llm, key)
                if value is not None:
                    config[key] = value
        else:
            config["mode"] = "ground_truth"
            config["disable_llm"] = True

        if stage in {"preprocessing", "schema_refinement"}:
            config.update(_schema_stage_runtime(stage, stage_config, self))
        elif stage == "data_extraction":
            config.update(_data_extraction_stage_runtime(stage_config, self))
        else:
            raise ValueError(f"Unsupported stage `{stage}`")

        return config


def _prompt_reference(prompt: str | dict[str, Any] | None, default_id: str) -> dict[str, Any]:
    if isinstance(prompt, dict):
        if "prompt_path" not in prompt:
            raise ValueError("Prompt mappings must include `prompt_path`.")
        return deepcopy(prompt)
    prompt_id = prompt or default_id
    prompt_path = prompt_id if str(prompt_id).endswith(".txt") else f"{prompt_id}.txt"
    return {"prompt_path": prompt_path}


def _schema_stage_runtime(
    stage: StageName,
    stage_config: StageConfig,
    experiment: ExperimentRuntime,
) -> dict[str, Any]:
    is_refinement = stage == "schema_refinement"
    config: dict[str, Any] = {
        "prompt": _prompt_reference(stage_config.prompt, "schemagen_5_0"),
        "in_fields": stage_config.input_fields
        or (
            {"document": "Document", "query": "Query"}
            if is_refinement
            else {"document": "Document"}
        ),
        "out_fields": stage_config.output_fields
        or {
            "table": "Table Assignment",
            "schema": "Updated Record of Schema",
            "reason": "Reasoning",
        },
    }
    if is_refinement:
        source_stage = stage_config.source_stage or "preprocessing"
        config["general_param_str"] = experiment.stage_artifact_id(source_stage)
    if stage_config.adaptive_sampling:
        config["adaptive_sampling"] = stage_config.adaptive_sampling.as_internal_dict()
    if stage_config.document_filtering:
        doc_filter = {
            "enable_calibrate": DEFAULT_DOC_FILTER_ENABLE_CALIBRATE,
            "threshold": DEFAULT_DOC_FILTER_THRESHOLD,
            **stage_config.document_filtering.as_internal_dict(),
        }
        config["document_filtering"] = doc_filter
        config["doc_filter"] = doc_filter
    if stage_config.schema_tailoring:
        config["schema_tailor"] = stage_config.schema_tailoring.as_internal_dict()
    if experiment.models.embedding and experiment.models.embedding.enabled:
        embedding = experiment.models.embedding
        config["embedding"] = {
            "enabled": True,
            "provider": embedding.provider,
            "model": embedding.model,
            "api_key": embedding.api_key
            or (os.getenv(embedding.api_key_env) if embedding.api_key_env else None),
            "base_url": embedding.base_url,
            "batch_size": embedding.batch_size,
            "storage_file": embedding.storage_file,
        }
    if stage_config.embedding:
        config["embedding"] = {
            **config.get("embedding", {}),
            **stage_config.embedding.as_internal_dict(),
        }
    if stage_config.retrieval:
        config["retrieval"] = stage_config.retrieval.as_internal_dict()
    return config


def _data_extraction_stage_runtime(
    stage_config: StageConfig,
    experiment: ExperimentRuntime,
) -> dict[str, Any]:
    schema_source = str(stage_config.schema_source or "schema_refinement")
    generated_schema_source = schema_source not in {"ground_truth", "gt"}
    source_stage: StageName = (
        "schema_refinement" if schema_source == "schema_refinement" else "preprocessing"
    )
    config: dict[str, Any] = {
        "schema_source": "generated" if generated_schema_source else "ground_truth",
        "schema_source_stage": source_stage,
        "prompts": stage_config.prompts
        or {
            "prompt_table": "datapop_table_json.txt",
            "prompt_attr": "datapop_attr_json.txt",
        },
        "in_fields": stage_config.input_fields
        or {
            "document": "Document",
            "query": "Query",
            "schema": "Schema",
        },
        "out_fields": stage_config.output_fields
        or {
            "table": "Table Assignment",
            "attribute": "Attribute Value",
            "reason": "Reasoning",
        },
    }
    if stage_config.oracle == "ground_truth":
        config["disable_llm"] = True
        config["use_ground_truth"] = True
        config["mode"] = "ground_truth"
    if stage_config.document_filtering:
        doc_filter = {
            "enable_calibrate": DEFAULT_DOC_FILTER_ENABLE_CALIBRATE,
            "threshold": DEFAULT_DOC_FILTER_THRESHOLD,
            **stage_config.document_filtering.as_internal_dict(),
        }
        config["document_filtering"] = doc_filter
        config["doc_filter"] = doc_filter
    if stage_config.table_assignment_cache:
        config["table_assignment_cache"] = stage_config.table_assignment_cache.as_internal_dict()
    if stage_config.proxy_runtime:
        config["proxy_runtime"] = {
            "predicate_proxy_mode": DEFAULT_PROXY_RUNTIME_MODE,
            "proxy_threshold": DEFAULT_PROXY_THRESHOLD,
            **stage_config.proxy_runtime.as_internal_dict(),
        }
    if stage_config.alpha_allocation:
        config["alpha_allocation"] = stage_config.alpha_allocation.as_internal_dict()
    return config


def load_redd_config(config_path: str | Path) -> tuple[ReDDConfig, Path]:
    config, resolved_path = load_yaml(config_path)
    return ReDDConfig.model_validate(config), resolved_path


def select_experiment(config: ReDDConfig, experiment_id: str) -> ExperimentRuntime:
    try:
        experiment = config.experiments[experiment_id]
    except KeyError as exc:
        raise KeyError(f"Experiment `{experiment_id}` not found in config") from exc

    datasets = {
        dataset_id: DatasetRuntimeConfig(
            id=dataset_id,
            root=resolve_repo_path(config.datasets[dataset_id].root),
            loader=config.datasets[dataset_id].loader,
            query_ids=config.datasets[dataset_id].query_ids,
            split=config.datasets[dataset_id].split,
            loader_options=deepcopy(config.datasets[dataset_id].loader_options),
        )
        for dataset_id in experiment.datasets
    }
    runtime = config.runtime.model_copy(
        update={
            "artifact_id": experiment.artifact_id or config.runtime.artifact_id,
        }
    )
    return ExperimentRuntime(
        id=experiment_id,
        project=config.project,
        runtime=runtime,
        models=config.models,
        datasets=datasets,
        stages={stage: config.stages[stage] for stage in experiment.stages},
        stage_order=list(experiment.stages),
    )


def load_experiment_runtime(
    config_path: str | Path,
    experiment_id: str,
) -> tuple[ExperimentRuntime, Path]:
    config, resolved_path = load_redd_config(config_path)
    return select_experiment(config, experiment_id), resolved_path


def resolve_api_key(config: Mapping[str, Any], provider: str, api_key: str | None = None) -> str:
    if api_key:
        return api_key
    if config.get("api_key"):
        return str(config["api_key"])
    if config.get("api_key_env") and os.getenv(str(config["api_key_env"])):
        return str(os.getenv(str(config["api_key_env"])))

    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var and os.getenv(env_var):
        return str(os.getenv(env_var))

    raise ValueError(
        f"API key is required for provider `{provider}`. "
        f"Provide api_key, api_key_env, or environment variable {env_var}."
    )
