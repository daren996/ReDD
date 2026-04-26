from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from redd.config import DEFAULT_LOG_DIR, resolve_repo_path
from redd.core.utils import logging_utils
from redd.core.utils.constants import PATH_TEMPLATES

MODULE_UNDER_TASK_LAYOUTS = {
    "module_under_task",
    "task_module",
    "dataset_task_module",
}


@dataclass(frozen=True)
class DatasetRuntimeContext:
    dataset: str
    data_root: Path
    out_root: Path


@dataclass(frozen=True)
class SchemaArtifactSource:
    out_root: Path
    param_str: str
    dataset_roots: dict[str, Path] | None = None

    def general_schema_path(self, dataset: str) -> Path:
        if self.dataset_roots and dataset in self.dataset_roots:
            return self.dataset_roots[dataset] / PATH_TEMPLATES.schema_general(self.param_str)
        return self.out_root / dataset / PATH_TEMPLATES.schema_general(self.param_str)

    def query_schema_pattern(self, dataset: str) -> Path:
        if self.dataset_roots and dataset in self.dataset_roots:
            return self.dataset_roots[dataset] / PATH_TEMPLATES.SCHEMA_QUERY_TAILORED.format(
                qid="{qid}",
                param_str=self.param_str,
            )
        return self.out_root / dataset / PATH_TEMPLATES.SCHEMA_QUERY_TAILORED.format(
            qid="{qid}",
            param_str=self.param_str,
        )


def normalize_stage_config(
    config: Mapping[str, Any] | None,
    *,
    module: str | None = None,
) -> dict[str, Any] | None:
    del module
    if config is None:
        return None
    return deepcopy(dict(config))


def resolve_log_dir(config: dict[str, Any]) -> Path:
    return resolve_repo_path(config.get("log_dir", DEFAULT_LOG_DIR))


def resolve_output_root(config: dict[str, Any]) -> Path:
    if "out_main" in config:
        return resolve_repo_path(config["out_main"])
    return resolve_repo_path(config["runtime"]["output_dir"])


def resolve_data_root(config: Mapping[str, Any]) -> Path:
    if "data_main" in config:
        return resolve_repo_path(config["data_main"])
    if "out_main" in config:
        return resolve_repo_path(config["out_main"])
    return resolve_repo_path(config["runtime"]["output_dir"])


def should_place_module_under_task(config: Mapping[str, Any]) -> bool:
    if bool(config.get("module_subdir_in_task", False)):
        return True

    layout = str(config.get("output_layout", "")).strip().lower()
    return layout in MODULE_UNDER_TASK_LAYOUTS


def resolve_stage_output_root(
    config: Mapping[str, Any],
    dataset_task: str,
    *,
    module_name: str | None = None,
) -> Path:
    out_main = resolve_output_root(dict(config))
    task_path = Path(dataset_task)

    if module_name and should_place_module_under_task(config):
        return out_main / task_path / module_name

    return out_main / task_path


def resolve_dataset_targets(config: dict[str, Any], key: str = "exp_dn_fn_list") -> list[str]:
    if "_runtime_contexts" in config:
        return [str(context["dataset"]) for context in config["_runtime_contexts"]]
    values = config.get(key)
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return values
    raise TypeError(f"`{key}` must be a string or list, got {type(values).__name__}")


def ensure_shared_output_root(*configs: Mapping[str, Any] | None) -> Path | None:
    roots = {
        resolve_output_root(dict(config))
        for config in configs
        if config is not None
    }
    if len(roots) > 1:
        raise ValueError("Expected all stage configs to share the same output root.")
    return next(iter(roots), None)


def resolve_dataset_contexts(
    config: Mapping[str, Any],
    datasets: Sequence[str] | None = None,
    *,
    key: str = "exp_dn_fn_list",
    module_name: str | None = None,
) -> list[DatasetRuntimeContext]:
    runtime_contexts = config.get("_runtime_contexts")
    if isinstance(runtime_contexts, list):
        requested = set(str(dataset) for dataset in datasets) if datasets is not None else None
        contexts = []
        for item in runtime_contexts:
            if not isinstance(item, Mapping):
                raise TypeError("Runtime dataset contexts must be mappings.")
            dataset = str(item["dataset"])
            if requested is not None and dataset not in requested:
                continue
            contexts.append(
                DatasetRuntimeContext(
                    dataset=dataset,
                    data_root=resolve_repo_path(item["data_root"]),
                    out_root=resolve_repo_path(item["out_root"]),
                )
            )
        if not contexts:
            raise ValueError("No datasets configured for this runtime selection.")
        return contexts

    resolved_datasets = (
        [str(dataset) for dataset in datasets]
        if datasets is not None
        else resolve_dataset_targets(dict(config), key=key)
    )
    if not resolved_datasets:
        raise ValueError("No datasets configured. Set datasets in the selected v2 experiment.")

    data_root = resolve_data_root(config)
    return [
        DatasetRuntimeContext(
            dataset=dataset,
            data_root=data_root / dataset,
            out_root=resolve_stage_output_root(config, dataset, module_name=module_name),
        )
        for dataset in resolved_datasets
    ]


def resolve_dataset_roots(
    config: Mapping[str, Any],
    datasets: Sequence[str] | None = None,
    *,
    key: str = "exp_dn_fn_list",
    module_name: str | None = None,
) -> list[tuple[str, Path, Path]]:
    return [
        (context.dataset, context.data_root, context.out_root)
        for context in resolve_dataset_contexts(
            config,
            datasets,
            key=key,
            module_name=module_name,
        )
    ]


def setup_runtime_logging(config: dict[str, Any], exp: str) -> Path:
    log_dir = resolve_log_dir(config)
    console_log_level = logging_utils.get_log_level(config.get("console_log_level", "WARNING"))
    logging_utils.setup_logging(
        exp=exp,
        log_dir=str(log_dir),
        console_log_level=console_log_level,
    )
    return log_dir


def configure_stage_logging(config: Mapping[str, Any], exp: str | None = None) -> Path:
    runtime_config = dict(config)
    resolved_exp = exp or str(runtime_config.get("exp_name", "redd"))
    return setup_runtime_logging(runtime_config, resolved_exp)


def resolve_schema_artifact_source(
    config: Mapping[str, Any] | None,
    *,
    prefer_general_param: bool = False,
    module: str = "schemagen",
) -> SchemaArtifactSource | None:
    normalized = normalize_stage_config(config, module=module)
    if normalized is None:
        return None

    param_str = None
    if prefer_general_param:
        param_str = normalized.get("general_param_str") or normalized.get("res_param_str")
    else:
        param_str = normalized.get("res_param_str")

    if not param_str:
        return None

    runtime_contexts = normalized.get("_runtime_contexts")
    if isinstance(runtime_contexts, list):
        dataset_roots = {
            str(context["dataset"]): resolve_repo_path(context["out_root"])
            for context in runtime_contexts
            if isinstance(context, Mapping)
        }
        return SchemaArtifactSource(
            out_root=Path(),
            param_str=str(param_str),
            dataset_roots=dataset_roots,
        )

    return SchemaArtifactSource(
        out_root=resolve_output_root(normalized),
        param_str=str(param_str),
    )


def build_data_loader_config(
    *,
    base_loader_config: Mapping[str, Any] | None,
    dataset: str,
    general_schema_source: SchemaArtifactSource | None,
    query_schema_source: SchemaArtifactSource | None,
) -> dict[str, Any]:
    loader_config = deepcopy(dict(base_loader_config or {}))
    filemap = deepcopy(dict(loader_config.get("filemap") or {}))

    if general_schema_source is not None:
        filemap.setdefault(
            "schema_general",
            str(general_schema_source.general_schema_path(dataset)),
        )

    if query_schema_source is not None:
        filemap.setdefault(
            "schema_query",
            str(query_schema_source.query_schema_pattern(dataset)),
        )

    if filemap:
        loader_config["filemap"] = filemap

    return loader_config
