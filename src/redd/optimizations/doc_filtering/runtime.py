"""
Doc Filtering Runtime Utilities.

This module contains orchestration helpers for running doc filtering
inside higher-level pipelines (e.g., data population), while keeping
filter implementations in this package and reducing caller complexity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Set

from redd.embedding import EmbeddingManager
from redd.core.utils.constants import PATH_TEMPLATES

from .base import DocFilterBase, NoOpFilter


def normalize_doc_filter_config(raw_config: Any) -> Dict[str, Any]:
    """
    Normalize doc filter config for backward compatibility.

    Supports both:
    - Legacy explicit style: {"filter_type": "schema_relevance", ...}
    - New style in configs: {"enabled": true, "only": false, ...}
    """
    if not isinstance(raw_config, dict) or not raw_config:
        return {}

    config = dict(raw_config)
    enabled = config.get("enabled")
    if enabled is None:
        enabled = True
    config["enabled"] = bool(enabled)
    config["only"] = bool(config.get("only", False))

    if config["enabled"] and not config.get("filter_type"):
        config["filter_type"] = "schema_relevance"

    if "train_ratio" in config:
        raise ValueError(
            "doc_filter.train_ratio is deprecated. "
            "Use top-level training_data_count instead."
        )

    return config


def save_doc_filter_result(
    query_id: str,
    excluded_doc_ids: Set[str],
    all_doc_ids: Sequence[str],
    out_root: Path,
    param_str: str,
    doc_filter_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    save_results_fn: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Path:
    """
    Persist doc-filter output for downstream evaluation scripts.
    """
    cfg = doc_filter_config or {}
    cf_dir = out_root / "doc_filter"
    cf_dir.mkdir(parents=True, exist_ok=True)

    target_recall = cfg.get("target_recall", 0.95)
    try:
        target_recall = float(target_recall)
    except (TypeError, ValueError):
        target_recall = 0.95

    excluded_sorted = sorted(excluded_doc_ids)
    kept_sorted = sorted(set(all_doc_ids) - set(excluded_doc_ids))
    payload = {
        "query_id": query_id,
        "excluded_doc_ids": excluded_sorted,
        "kept_doc_ids": kept_sorted,
        "metadata": {
            **(metadata or {}),
            "target_recall": target_recall,
            "num_docs_input": len(all_doc_ids),
            "num_docs_excluded": len(excluded_sorted),
            "num_docs_kept": len(kept_sorted),
        },
    }

    cf_path = cf_dir / PATH_TEMPLATES.doc_filter_result(
        query_id,
        param_str,
        target_recall,
    )

    if save_results_fn is None:
        import json

        with open(cf_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        save_results_fn(str(cf_path), payload)

    legacy_dir = out_root / "chunk_filter"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_cf_path = legacy_dir / PATH_TEMPLATES.chunk_filter_result(
        query_id,
        param_str,
        target_recall,
    )
    if legacy_cf_path != cf_path:
        if save_results_fn is None:
            import json

            with open(legacy_cf_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            save_results_fn(str(legacy_cf_path), payload)

    logging.info(
        "[doc_filtering.runtime:save_doc_filter_result] Saved doc filter result: %s",
        cf_path,
    )
    return cf_path


def run_query_doc_filter(
    query_id: str,
    schema_query: Any,
    loader: Any,
    test_doc_ids: Sequence[str],
    train_doc_ids: Sequence[str],
    doc_filter: DocFilterBase,
    doc_filter_config: Dict[str, Any],
    api_key: Optional[str],
    out_root: Path,
    param_str: str,
    save_results_fn: Callable[[str, Dict[str, Any]], None],
) -> Set[str]:
    """
    Run doc filtering for one query and save standard result JSON.

    Returns:
        Set of excluded doc IDs.
    """
    filter_kwargs = {
        "data_loader": loader,
        "schema_context": schema_query,
    }

    if not isinstance(doc_filter, NoOpFilter):
        emb_model = doc_filter_config.get(
            "embedding_model",
            "text-embedding-3-small",
        )
        emb_api_key = doc_filter_config.get("embedding_api_key") or api_key
        emb_manager = EmbeddingManager(
            loader=loader,
            model=emb_model,
            api_key=emb_api_key,
        )
        filter_kwargs["embedding_manager"] = emb_manager

        enable_calibrate = bool(doc_filter_config.get("enable_calibrate", False))
        filter_kwargs["enable_calibrate"] = enable_calibrate
        if enable_calibrate:
            if train_doc_ids:
                filter_kwargs["train_doc_ids"] = list(train_doc_ids)
            else:
                logging.warning(
                    "[doc_filtering.runtime:run_query_doc_filter] "
                    "enable_calibrate=True but no calibration docs selected."
                )

    result = doc_filter.filter(
        query_id=query_id,
        doc_ids=list(test_doc_ids),
        **filter_kwargs,
    )
    excluded_doc_ids = set(result.excluded_doc_ids)

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    metadata.update(
        {
            "num_train_docs": len(train_doc_ids),
            "num_test_docs": len(test_doc_ids),
        }
    )

    save_doc_filter_result(
        query_id=query_id,
        excluded_doc_ids=excluded_doc_ids,
        all_doc_ids=list(test_doc_ids),
        out_root=out_root,
        param_str=param_str,
        doc_filter_config=doc_filter_config,
        metadata=metadata,
        save_results_fn=save_results_fn,
    )
    return excluded_doc_ids
