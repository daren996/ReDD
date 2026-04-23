from __future__ import annotations

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_experiment_config
from .core.data_loader import DataLoaderBase
from .core.data_population import create_data_populator
from .core.schema_gen import create_schema_generator
from .core.utils.constants import PATH_TEMPLATES
from .loader import create_data_loader
from .runtime import (
    build_data_loader_config,
    configure_stage_logging,
    ensure_shared_output_root,
    normalize_stage_config,
    resolve_dataset_roots,
    resolve_schema_artifact_source,
)


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
    in_fields = config.get("in_fields") or {}
    return "query" in in_fields


def _build_schema_generator_impl(config: Mapping[str, Any], api_key: str | None = None):
    return create_schema_generator(dict(config), api_key=api_key)


def _build_data_populator_impl(config: Mapping[str, Any], api_key: str | None = None):
    return create_data_populator(dict(config), api_key=api_key)


def _build_doc_dict(loader: Any) -> dict[str, list[str]]:
    doc_dict: dict[str, list[str]] = {}
    for doc_text, doc_id, metadata in loader.iter_docs():
        source_info = metadata.get("source_file") or metadata.get("table_name") or ""
        doc_dict[str(doc_id)] = [doc_text, source_info]
    return doc_dict


def _create_loader_for_impl(impl: Any, data_root: str | Path):
    loader_type = str(getattr(impl, "loader_type", "sqlite")).lower()
    raw_loader_config = getattr(impl, "loader_config", {})
    loader_config = deepcopy(dict(raw_loader_config or {}))
    return create_data_loader(
        data_root=data_root,
        loader_type=loader_type,
        loader_config=loader_config,
    )


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
                    "PREPROCESSING and SCHEMA REFINEMENT currently need the same `out_main` "
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
        config, _ = load_experiment_config(config_path, exp, module="schemagen")
        return cls(config, api_key=api_key, configure_logging=configure_logging)

    def run(self, datasets: Sequence[str] | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {}

        if self.preprocessing_config is not None:
            results[PREPROCESSING.value] = self.preprocessing(datasets=datasets)
        if self.refinement_config is not None:
            results[SCHEMA_REFINEMENT.value] = self.schema_refine(datasets=datasets)

        return results

    def schema_global(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Alias for query-independent schema extraction."""
        return self.preprocessing(datasets=datasets)

    def preprocessing(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        from .embedding import EmbeddingManager
        from .schema_global import discover_global_schema
        from .retrieval import build_retrieval_index

        config = self.preprocessing_config
        if config is None:
            raise ValueError("No preprocessing config was provided.")
        if _has_query_input(config):
            raise ValueError("PREPROCESSING requires a query-independent schema config.")
        if (config.get("retrieval") or {}).get("enabled") and not (config.get("embedding") or {}).get("enabled"):
            raise ValueError("Retrieval index preparation currently requires `embedding.enabled = true`.")

        impl = _build_schema_generator_impl(config, api_key=self.api_key)
        summaries = []

        for dataset, data_root, out_root in resolve_dataset_roots(config, datasets):
            out_root.mkdir(parents=True, exist_ok=True)
            impl.data_root = data_root
            impl.out_root = out_root
            impl.loader = _create_loader_for_impl(impl, data_root)

            doc_dict = impl._build_doc_dict()
            result_path = out_root / PATH_TEMPLATES.schema_gen_result_general(impl.param_str)
            res_dict = impl.load_processed_res(result_path)
            progress_name = data_root.name

            if getattr(impl, "adaptive_enabled", False):
                impl.process_documents_adaptive(doc_dict, "", res_dict, {}, None, result_path, progress_name)
            else:
                impl.process_documents(doc_dict, "", res_dict, {}, None, result_path, progress_name)

            embedding_path = None
            retrieval_index_path = None

            embedding_config = config.get("embedding") or {}
            if embedding_config.get("enabled"):
                embedding_path = out_root / str(embedding_config.get("storage_file", "embeddings.sqlite3"))
                manager = EmbeddingManager(
                    embedding_path,
                    model=str(embedding_config.get("model", "text-embedding-3-small")),
                    api_key=embedding_config.get("api_key") or self.api_key,
                    provider=embedding_config.get("provider"),
                    base_url=embedding_config.get("base_url"),
                )
                doc_embeddings = manager.get_doc_embeddings(
                    impl.loader,
                    batch_size=int(embedding_config.get("batch_size", 100)),
                )

                retrieval_config = config.get("retrieval") or {}
                if retrieval_config.get("enabled"):
                    retrieval_index_path = out_root / str(
                        retrieval_config.get("index_file", "retrieval_index.npz")
                    )
                    index = build_retrieval_index(
                        doc_embeddings,
                        model=manager.model,
                        metadata={"dataset": dataset},
                    )
                    index.save(retrieval_index_path)

            artifacts = discover_global_schema(
                config=config,
                res_dict=impl.load_json(result_path),
                doc_dict=doc_dict,
                out_root=out_root,
                param_str=impl.param_str,
            )
            impl.doc_clustering(out_root, doc_dict)

            doc_cluster_path = None
            if getattr(impl, "doc_cluster_file", None):
                doc_cluster_path = str(out_root / impl.doc_cluster_file)

            summaries.append(
                {
                    "dataset": dataset,
                    "data_root": str(data_root),
                    "out_root": str(out_root),
                    "result_path": str(result_path),
                    "global_schema_path": str(artifacts.schema_path),
                    "embedding_path": str(embedding_path) if embedding_path else None,
                    "retrieval_index_path": str(retrieval_index_path) if retrieval_index_path else None,
                    "doc_cluster_path": doc_cluster_path,
                }
            )

        return summaries

    def schema_refine(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        config = self.refinement_config
        if config is None:
            raise ValueError("No schema refinement config was provided.")
        if not _has_query_input(config):
            raise ValueError("SCHEMA REFINEMENT requires a query-aware schema config.")
        schema_tailor_config = config.get("schema_tailor") or {}
        if isinstance(schema_tailor_config, Mapping) and schema_tailor_config.get("enabled"):
            return self._schema_refine_with_schema_tailor(config, datasets=datasets)

        impl = _build_schema_generator_impl(config, api_key=self.api_key)
        summaries = []

        for dataset, data_root, out_root in resolve_dataset_roots(config, datasets):
            out_root.mkdir(parents=True, exist_ok=True)
            impl.data_root = data_root
            impl.out_root = out_root
            impl.loader = _create_loader_for_impl(impl, data_root)

            doc_dict = impl._build_doc_dict()
            query_dict = impl.loader.load_query_dict()

            if impl.general_param_str:
                preprocessing_result_path = out_root / PATH_TEMPLATES.schema_gen_result_general(impl.general_param_str)
                if not preprocessing_result_path.exists():
                    raise FileNotFoundError(
                        f"Missing preprocessing artifact for dataset `{dataset}`: {preprocessing_result_path}. "
                        "Run PREPROCESSING first."
                    )

            impl.doc_clustering(out_root, doc_dict)

            query_results = {}
            for qid, query_info in query_dict.items():
                query = query_info["query"]
                result_path = out_root / PATH_TEMPLATES.schema_gen_result_query(qid, impl.param_str)
                res_dict = impl.load_processed_res(result_path)
                log_init = impl.load_log_init(out_root, qid)
                general_schema = impl.get_general_schema(out_root, doc_dict, qid, query)
                progress_name = f"{data_root.name}-{qid}"

                if getattr(impl, "adaptive_enabled", False):
                    impl.process_documents_adaptive(
                        doc_dict,
                        query,
                        res_dict,
                        log_init,
                        general_schema,
                        result_path,
                        progress_name,
                    )
                else:
                    impl.process_documents(
                        doc_dict,
                        query,
                        res_dict,
                        log_init,
                        general_schema,
                        result_path,
                        progress_name,
                    )
                impl.tailor_schema(out_root, doc_dict, qid, query)

                query_results[str(qid)] = {
                    "result_path": str(result_path),
                    "general_schema_path": str(
                        out_root / PATH_TEMPLATES.schema_general(impl.general_param_str, qid)
                    )
                    if impl.general_param_str
                    else None,
                    "tailored_schema_path": str(
                        out_root / PATH_TEMPLATES.schema_query_tailored(qid, impl.param_str)
                    ),
                }

            summaries.append(
                {
                    "dataset": dataset,
                    "data_root": str(data_root),
                    "out_root": str(out_root),
                    "queries": query_results,
                }
            )

        return summaries

    def schema_refinement(self, datasets: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Backward-compatible alias for `schema_refine`."""
        return self.schema_refine(datasets=datasets)

    def _schema_refine_with_schema_tailor(
        self,
        config: Mapping[str, Any],
        *,
        datasets: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        from .schema_refine import SchemaTailor

        impl = _build_schema_generator_impl(config, api_key=self.api_key)
        tailor_config = deepcopy(dict(config))
        nested_config = config.get("schema_tailor") or {}
        if isinstance(nested_config, Mapping):
            tailor_config.update(dict(nested_config))

        tailor = SchemaTailor(tailor_config, api_key=self.api_key)
        summaries = []

        for dataset, data_root, out_root in resolve_dataset_roots(config, datasets):
            out_root.mkdir(parents=True, exist_ok=True)
            impl.data_root = data_root
            impl.out_root = out_root
            impl.loader = _create_loader_for_impl(impl, data_root)

            doc_dict = impl._build_doc_dict() if hasattr(impl, "_build_doc_dict") else _build_doc_dict(impl.loader)
            general_schema = impl.get_general_schema(out_root, doc_dict)
            query_dict = impl.loader.load_query_dict()

            query_results = {}
            for qid, query_info in query_dict.items():
                result = tailor.process_query_with_adaptive_sampling(
                    doc_dict,
                    query_info["query"],
                    general_schema,
                    out_root,
                    str(qid),
                )
                tailored_schema_path = out_root / PATH_TEMPLATES.schema_query_tailored(str(qid), tailor.param_str)
                stats_path = out_root / f"{tailored_schema_path.stem}_stats.json"
                query_results[str(qid)] = {
                    "tailored_schema_path": str(tailored_schema_path),
                    "stats_path": str(stats_path),
                    "documents_processed": result["documents_processed"],
                    "stopped_early": result["stopped_early"],
                }

            summaries.append(
                {
                    "dataset": dataset,
                    "data_root": str(data_root),
                    "out_root": str(out_root),
                    "queries": query_results,
                    "engine": "schema_tailor",
                }
            )

        return summaries


class DataPopulator:
    """Stable public entry point for the DATA EXTRACTION stage."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        api_key: str | None = None,
        configure_logging: bool = True,
    ) -> None:
        self.config = normalize_stage_config(config, module="datapop") or {}
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
        config, _ = load_experiment_config(config_path, exp, module="datapop")
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
        config = deepcopy(self.config)
        impl = _build_data_populator_impl(config, api_key=self.api_key)
        summaries = []

        general_schema_root, general_schema_param = self._resolve_general_schema_source(
            schema_generator=schema_generator,
            schema_config=schema_config,
        )
        query_schema_root, query_schema_param = self._resolve_query_schema_source(
            schema_generator=schema_generator,
            schema_config=schema_config,
        )

        for dataset, data_root, out_root in resolve_dataset_roots(
            config,
            datasets,
            module_name="data_pop",
        ):
            out_root.mkdir(parents=True, exist_ok=True)
            loader_config = self._build_loader_config(
                base_loader_config=config.get("data_loader_config"),
                dataset=dataset,
                general_schema_root=general_schema_root,
                general_schema_param=general_schema_param,
                query_schema_root=query_schema_root,
                query_schema_param=query_schema_param,
            )
            impl.loader_config = loader_config
            impl._process_dataset(data_root, out_root)

            summaries.append(
                {
                    "dataset": dataset,
                    "data_root": str(data_root),
                    "out_root": str(out_root),
                    "loader_config": deepcopy(loader_config),
                }
            )

        return summaries

    def _resolve_general_schema_source(
        self,
        *,
        schema_generator: SchemaGenerator | None,
        schema_config: Mapping[str, Any] | None,
    ) -> tuple[Path | None, str | None]:
        if schema_generator and schema_generator.preprocessing_config:
            source = resolve_schema_artifact_source(schema_generator.preprocessing_config)
            if source is not None:
                return source.out_root, source.param_str

        if schema_config:
            source = resolve_schema_artifact_source(schema_config, prefer_general_param=True)
            if source is not None:
                return source.out_root, source.param_str

        return None, None

    def _resolve_query_schema_source(
        self,
        *,
        schema_generator: SchemaGenerator | None,
        schema_config: Mapping[str, Any] | None,
    ) -> tuple[Path | None, str | None]:
        if schema_generator and schema_generator.refinement_config:
            source = resolve_schema_artifact_source(schema_generator.refinement_config)
            if source is not None:
                return source.out_root, source.param_str

        if schema_config:
            source = resolve_schema_artifact_source(schema_config)
            if source is not None:
                return source.out_root, source.param_str

        return None, None

    def _build_loader_config(
        self,
        *,
        base_loader_config: Mapping[str, Any] | None,
        dataset: str,
        general_schema_root: Path | None,
        general_schema_param: str | None,
        query_schema_root: Path | None,
        query_schema_param: str | None,
    ) -> dict[str, Any]:
        general_schema_source = None
        if general_schema_root and general_schema_param:
            general_schema_source = resolve_schema_artifact_source(
                {
                    "out_main": str(general_schema_root),
                    "general_param_str": general_schema_param,
                    "res_param_str": general_schema_param,
                },
                prefer_general_param=True,
            )

        query_schema_source = None
        if query_schema_root and query_schema_param:
            query_schema_source = resolve_schema_artifact_source(
                {
                    "out_main": str(query_schema_root),
                    "res_param_str": query_schema_param,
                },
            )

        return build_data_loader_config(
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
    if stages is None:
        resolved_stages = []
        if schema_generator and schema_generator.preprocessing_config is not None:
            resolved_stages.append(PREPROCESSING)
        if schema_generator and schema_generator.refinement_config is not None:
            resolved_stages.append(SCHEMA_REFINEMENT)
        if data_populator:
            resolved_stages.append(DATA_EXTRACTION)

        if not resolved_stages:
            raise ValueError("run_pipeline requires at least one pipeline component.")
    else:
        resolved_stages = [_coerce_stage(stage) for stage in stages]

    results: dict[str, Any] = {}

    for stage in resolved_stages:
        if stage is PREPROCESSING:
            if schema_generator is None:
                raise ValueError("PREPROCESSING requires `schema_generator=`.")
            results[stage.value] = schema_generator.preprocessing(datasets=datasets)
        elif stage is SCHEMA_REFINEMENT:
            if schema_generator is None:
                raise ValueError("SCHEMA REFINEMENT requires `schema_generator=`.")
            results[stage.value] = schema_generator.schema_refinement(datasets=datasets)
        elif stage is DATA_EXTRACTION:
            if data_populator is None:
                raise ValueError("DATA EXTRACTION requires `data_populator=`.")
            results[stage.value] = data_populator.data_extraction(
                datasets=datasets,
                schema_generator=schema_generator,
            )

    return results


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
