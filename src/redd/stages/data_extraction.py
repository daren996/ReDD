"""Stage orchestration for data extraction."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from redd.core.data_population import create_data_populator
from redd.runtime import (
    build_data_loader_config,
    resolve_dataset_roots,
    resolve_schema_artifact_source,
)

if TYPE_CHECKING:
    from redd.api import DataPopulator, SchemaGenerator

__all__ = [
    "build_data_populator_impl",
    "build_loader_config",
    "resolve_general_schema_source",
    "resolve_query_schema_source",
    "resolve_schema_source_mode",
    "run_data_extraction",
]


def build_data_populator_impl(config: Mapping[str, Any], api_key: str | None = None):
    return create_data_populator(dict(config), api_key=api_key)


def run_data_extraction(
    populator: DataPopulator,
    datasets: Sequence[str] | None = None,
    *,
    schema_generator: SchemaGenerator | None = None,
    schema_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = deepcopy(populator.config)
    impl = build_data_populator_impl(config, api_key=populator.api_key)
    base_impl_config = deepcopy(impl.config)
    summaries = []
    schema_source = resolve_schema_source_mode(config)

    general_schema_source = resolve_general_schema_source(
        schema_generator=schema_generator,
        schema_config=schema_config,
    )
    query_schema_source = resolve_query_schema_source(
        schema_generator=schema_generator,
        schema_config=schema_config,
    )

    for dataset, data_root, out_root in resolve_dataset_roots(
        config,
        datasets,
        module_name="data_extraction",
    ):
        out_root.mkdir(parents=True, exist_ok=True)
        loader_config = build_loader_config(
            schema_source=schema_source,
            base_loader_config=config.get("data_loader_config"),
            dataset=dataset,
            general_schema_source=general_schema_source,
            query_schema_source=query_schema_source,
        )
        impl.loader_config = loader_config
        impl.config = deepcopy(base_impl_config)
        upstream_doc_filter_root = _upstream_doc_filter_root(
            schema_generator=schema_generator,
            schema_config=schema_config,
            dataset=dataset,
        )
        if upstream_doc_filter_root is not None:
            impl.config["upstream_doc_filter_root"] = str(upstream_doc_filter_root)
        query_ids = _runtime_query_ids_for_dataset(config, dataset)
        if query_ids is not None:
            impl.config["exp_query_id_list"] = query_ids
        impl._process_dataset(data_root, out_root)

        summaries.append(
            {
                "dataset": dataset,
                "data_root": str(data_root),
                "out_root": str(out_root),
                "schema_source": schema_source,
                "loader_config": deepcopy(loader_config),
                "query_ids": query_ids,
            }
        )

    return summaries


def _upstream_doc_filter_root(
    *,
    schema_generator: SchemaGenerator | None,
    schema_config: Mapping[str, Any] | None,
    dataset: str,
) -> Path | None:
    candidates: list[Mapping[str, Any]] = []
    if schema_generator and schema_generator.refinement_config:
        candidates.append(schema_generator.refinement_config)
    if schema_config:
        candidates.append(schema_config)

    for candidate in candidates:
        doc_filter = candidate.get("doc_filter") or candidate.get("document_filtering") or {}
        if not isinstance(doc_filter, Mapping) or not doc_filter.get("enabled"):
            continue
        runtime_contexts = candidate.get("_runtime_contexts")
        if not isinstance(runtime_contexts, list):
            continue
        for item in runtime_contexts:
            if not isinstance(item, Mapping) or str(item.get("dataset")) != dataset:
                continue
            out_root = item.get("out_root")
            if out_root:
                return Path(str(out_root))
    return None


def _runtime_query_ids_for_dataset(
    config: Mapping[str, Any],
    dataset: str,
) -> list[str] | None:
    runtime_contexts = config.get("_runtime_contexts")
    if not isinstance(runtime_contexts, list):
        return None
    for item in runtime_contexts:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("dataset")) != dataset:
            continue
        query_ids = item.get("query_ids")
        if query_ids is None:
            return None
        if isinstance(query_ids, str):
            return [query_ids]
        return [str(query_id) for query_id in query_ids]
    return None


def resolve_schema_source_mode(config: Mapping[str, Any]) -> str:
    raw_value = str(config.get("schema_source", "generated")).strip().lower().replace("-", "_")
    if raw_value in {"generated", "gen"}:
        return "generated"
    if raw_value in {"ground_truth", "gt"}:
        return "ground_truth"
    raise ValueError("Unsupported `schema_source`. Use `generated` or `ground_truth`.")


def resolve_general_schema_source(
    *,
    schema_generator: SchemaGenerator | None,
    schema_config: Mapping[str, Any] | None,
) -> Any | None:
    if schema_generator and schema_generator.preprocessing_config:
        source = resolve_schema_artifact_source(schema_generator.preprocessing_config)
        if source is not None:
            return source

    if schema_config:
        source = resolve_schema_artifact_source(schema_config, prefer_general_param=True)
        if source is not None:
            return source

    return None


def resolve_query_schema_source(
    *,
    schema_generator: SchemaGenerator | None,
    schema_config: Mapping[str, Any] | None,
) -> Any | None:
    if schema_generator and schema_generator.refinement_config:
        source = resolve_schema_artifact_source(schema_generator.refinement_config)
        if source is not None:
            return source

    if schema_config:
        source = resolve_schema_artifact_source(schema_config)
        if source is not None:
            return source

    return None


def build_loader_config(
    *,
    schema_source: str,
    base_loader_config: Mapping[str, Any] | None,
    dataset: str,
    general_schema_source: Any | None,
    query_schema_source: Any | None,
) -> dict[str, Any]:
    if schema_source == "ground_truth":
        return deepcopy(dict(base_loader_config or {}))

    return build_data_loader_config(
        base_loader_config=base_loader_config,
        dataset=dataset,
        general_schema_source=general_schema_source,
        query_schema_source=query_schema_source,
    )
