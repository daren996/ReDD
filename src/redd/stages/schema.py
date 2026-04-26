"""Stage orchestration for schema preprocessing and refinement."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from redd.core.data_population.strategies.doc_filtering import DocFilteringStrategy
from redd.core.schema_gen import create_schema_generator
from redd.core.utils.constants import PATH_TEMPLATES
from redd.core.utils.data_split import resolve_training_data_count, split_doc_ids
from redd.loader import create_data_loader
from redd.runtime import resolve_dataset_roots

if TYPE_CHECKING:
    from redd.api import SchemaGenerator

__all__ = [
    "build_doc_dict",
    "create_loader_for_impl",
    "has_query_input",
    "run_schema",
    "run_schema_preprocessing",
    "run_schema_refinement",
]


def has_query_input(config: Mapping[str, Any]) -> bool:
    in_fields = config.get("in_fields") or {}
    return "query" in in_fields


def build_schema_generator_impl(config: Mapping[str, Any], api_key: str | None = None):
    return create_schema_generator(dict(config), api_key=api_key)


def build_doc_dict(loader: Any) -> dict[str, list[str]]:
    doc_dict: dict[str, list[str]] = {}
    for doc_text, doc_id, metadata in loader.iter_docs():
        source_info = metadata.get("source_file") or metadata.get("table_name") or ""
        doc_dict[str(doc_id)] = [doc_text, source_info]
    return doc_dict


def create_loader_for_impl(impl: Any, data_root: str | Path):
    loader_type = str(getattr(impl, "loader_type", "hf_manifest")).lower()
    raw_loader_config = getattr(impl, "loader_config", {})
    loader_config = deepcopy(dict(raw_loader_config or {}))
    return create_data_loader(
        data_root=data_root,
        loader_type=loader_type,
        loader_config=loader_config,
    )


def run_schema(generator: SchemaGenerator, datasets: Sequence[str] | None = None) -> dict[str, Any]:
    from redd.api import PREPROCESSING, SCHEMA_REFINEMENT

    results: dict[str, Any] = {}
    if generator.preprocessing_config is not None:
        results[PREPROCESSING.value] = run_schema_preprocessing(generator, datasets=datasets)
    if generator.refinement_config is not None:
        results[SCHEMA_REFINEMENT.value] = run_schema_refinement(generator, datasets=datasets)
    return results


def run_schema_preprocessing(
    generator: SchemaGenerator,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    from redd.embedding import EmbeddingManager
    from redd.retrieval import build_retrieval_index
    from redd.schema_global import discover_global_schema

    config = generator.preprocessing_config
    if config is None:
        raise ValueError("No preprocessing config was provided.")
    if has_query_input(config):
        raise ValueError("PREPROCESSING requires a query-independent schema config.")
    if (config.get("retrieval") or {}).get("enabled") and not (config.get("embedding") or {}).get("enabled"):
        raise ValueError("Retrieval index preparation currently requires `embedding.enabled = true`.")

    impl = build_schema_generator_impl(config, api_key=generator.api_key)
    summaries = []

    for dataset, data_root, out_root in resolve_dataset_roots(config, datasets):
        out_root.mkdir(parents=True, exist_ok=True)
        impl.data_root = data_root
        impl.out_root = out_root
        impl.loader = create_loader_for_impl(impl, data_root)

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
                api_key=embedding_config.get("api_key") or generator.api_key,
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


def run_schema_refinement(
    generator: SchemaGenerator,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    config = generator.refinement_config
    if config is None:
        raise ValueError("No schema refinement config was provided.")
    if not has_query_input(config):
        raise ValueError("SCHEMA REFINEMENT requires a query-aware schema config.")
    schema_tailor_config = config.get("schema_tailor") or {}
    if isinstance(schema_tailor_config, Mapping) and schema_tailor_config.get("enabled"):
        return run_schema_refinement_with_schema_tailor(generator, config, datasets=datasets)

    impl = build_schema_generator_impl(config, api_key=generator.api_key)
    summaries = []

    for dataset, data_root, out_root in resolve_dataset_roots(config, datasets):
        out_root.mkdir(parents=True, exist_ok=True)
        impl.data_root = data_root
        impl.out_root = out_root
        impl.loader = create_loader_for_impl(impl, data_root)

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
            docs_for_query, filter_stats = _filter_schema_refinement_docs(
                config=config,
                impl=impl,
                qid=str(qid),
                doc_dict=doc_dict,
                schema_query=general_schema,
                out_root=out_root,
            )
            progress_name = f"{data_root.name}-{qid}"

            if getattr(impl, "adaptive_enabled", False):
                impl.process_documents_adaptive(
                    docs_for_query,
                    query,
                    res_dict,
                    log_init,
                    general_schema,
                    result_path,
                    progress_name,
                    total_documents_override=filter_stats["total_documents"],
                    filtered_documents=filter_stats["filtered_documents"],
                )
            else:
                impl.process_documents(
                    docs_for_query,
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


def _filter_schema_refinement_docs(
    *,
    config: Mapping[str, Any],
    impl: Any,
    qid: str,
    doc_dict: dict[str, list[str]],
    schema_query: Any,
    out_root: Path,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    total_documents = len(doc_dict)
    stats = {
        "total_documents": total_documents,
        "filtered_documents": total_documents,
        "excluded_documents": 0,
    }
    strategy = DocFilteringStrategy(dict(config))
    if not strategy.enabled:
        return doc_dict, stats

    all_doc_ids = list(doc_dict.keys())
    training_data_count = resolve_training_data_count(dict(config))
    train_doc_ids, test_doc_ids = split_doc_ids(all_doc_ids, training_data_count)
    excluded_doc_ids = strategy.excluded_doc_ids_for_query(
        query_id=qid,
        schema_query=schema_query or [],
        loader=impl.loader,
        test_doc_ids=test_doc_ids,
        train_doc_ids=train_doc_ids,
        api_key=getattr(impl, "api_key", None) or impl.config.get("api_key"),
        out_root=out_root,
        param_str=impl.param_str,
        save_results_fn=lambda p, d: impl.save_results(str(p), d),
    )
    kept_doc_ids = [
        doc_id
        for doc_id in test_doc_ids
        if doc_id not in excluded_doc_ids and doc_id in doc_dict
    ]
    filtered = {
        str(index): doc_dict[doc_id]
        for index, doc_id in enumerate(kept_doc_ids)
    }
    stats = {
        "total_documents": total_documents,
        "filtered_documents": len(filtered),
        "excluded_documents": len(excluded_doc_ids),
    }
    logging.info(
        "[schema_refinement] Doc filter query=%s input=%d train=%d test=%d kept=%d excluded=%d",
        qid,
        total_documents,
        len(train_doc_ids),
        len(test_doc_ids),
        len(filtered),
        len(excluded_doc_ids),
    )
    return filtered, stats


def run_schema_refinement_with_schema_tailor(
    generator: SchemaGenerator,
    config: Mapping[str, Any],
    *,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    from redd.schema_refine import SchemaTailor

    impl = build_schema_generator_impl(config, api_key=generator.api_key)
    tailor_config = deepcopy(dict(config))
    nested_config = config.get("schema_tailor") or {}
    if isinstance(nested_config, Mapping):
        tailor_config.update(dict(nested_config))

    tailor = SchemaTailor(tailor_config, api_key=generator.api_key)
    summaries = []

    for dataset, data_root, out_root in resolve_dataset_roots(config, datasets):
        out_root.mkdir(parents=True, exist_ok=True)
        impl.data_root = data_root
        impl.out_root = out_root
        impl.loader = create_loader_for_impl(impl, data_root)

        doc_dict = impl._build_doc_dict() if hasattr(impl, "_build_doc_dict") else build_doc_dict(impl.loader)
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
