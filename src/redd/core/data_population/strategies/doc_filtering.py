"""Document-filter orchestration for the unified data-population stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from redd.optimizations.doc_filtering import create_doc_filter
from redd.optimizations.doc_filtering.runtime import (
    load_doc_filter_result,
    normalize_doc_filter_config,
    run_query_doc_filter,
    save_doc_filter_result,
)

SaveResultsFn = Callable[[Path, Dict[str, Any]], None]

__all__ = ["DocFilteringStrategy"]


class DocFilteringStrategy:
    """Stage-facing wrapper around optional document filtering."""

    def __init__(self, config: Dict[str, Any]):
        self.config = normalize_doc_filter_config(config.get("doc_filter"))
        self.enabled = bool(self.config.get("enabled", False))
        self.only = bool(self.config.get("only", False))
        self._base_filter = create_doc_filter(self.config) if self.enabled else None

        if self.enabled:
            logging.info(
                "[DocFilteringStrategy] Enabled: type=%s only=%s",
                self.config.get("filter_type"),
                self.only,
            )
        elif self.only:
            logging.warning(
                "[DocFilteringStrategy] doc_filter.only=True but doc_filter.enabled=False. "
                "Ignoring only mode."
            )
            self.only = False

    def excluded_doc_ids_for_query(
        self,
        *,
        query_id: str,
        schema_query: List[Dict[str, Any]],
        loader: Any,
        test_doc_ids: List[str],
        train_doc_ids: List[str],
        api_key: Optional[str],
        out_root: Path,
        param_str: str,
        save_results_fn: SaveResultsFn,
        target_recall_override: Optional[float] = None,
    ) -> Set[str]:
        if self._base_filter is None:
            return set()

        config_for_query = dict(self.config)
        doc_filter = self._base_filter
        if target_recall_override is not None:
            config_for_query["target_recall"] = float(target_recall_override)
            doc_filter = create_doc_filter(config_for_query)

        cached_payload, cached_path = load_doc_filter_result(out_root, query_id=query_id)
        if cached_payload is not None and cached_path is not None:
            excluded_doc_ids = {
                str(doc_id)
                for doc_id in cached_payload.get("excluded_doc_ids", [])
            }
            logging.info(
                "[DocFilteringStrategy] Reused existing doc filter for query=%s: "
                "excluded=%d source=%s",
                query_id,
                len(excluded_doc_ids),
                cached_path,
            )
            return excluded_doc_ids

        return set(
            run_query_doc_filter(
                query_id=query_id,
                schema_query=schema_query,
                loader=loader,
                test_doc_ids=test_doc_ids,
                train_doc_ids=train_doc_ids,
                doc_filter=doc_filter,
                doc_filter_config=config_for_query,
                api_key=api_key,
                out_root=out_root,
                param_str=param_str,
                save_results_fn=save_results_fn,
            )
        )

    def reused_excluded_doc_ids_for_query(
        self,
        *,
        query_id: str,
        upstream_root: Path,
        test_doc_ids: List[str],
        out_root: Path,
        param_str: str,
        save_results_fn: SaveResultsFn,
    ) -> Set[str] | None:
        payload, source_path = load_doc_filter_result(upstream_root, query_id=query_id)
        if payload is None or source_path is None:
            return None

        excluded_doc_ids = {
            str(doc_id)
            for doc_id in payload.get("excluded_doc_ids", [])
        }
        metadata = payload.get("metadata") if isinstance(payload, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = {
            **metadata,
            "reused_from_stage": "schema_refinement",
            "source_artifact": str(source_path),
            "num_docs_input": len(test_doc_ids),
            "num_docs_excluded": len(excluded_doc_ids),
            "num_docs_kept": len(set(test_doc_ids) - excluded_doc_ids),
        }
        save_doc_filter_result(
            query_id=query_id,
            excluded_doc_ids=excluded_doc_ids,
            all_doc_ids=list(test_doc_ids),
            out_root=out_root,
            param_str=param_str,
            doc_filter_config=self.config,
            metadata=metadata,
            save_results_fn=save_results_fn,
        )
        logging.info(
            "[DocFilteringStrategy] Reused schema_refinement doc filter for query=%s: "
            "excluded=%d input=%d source=%s",
            query_id,
            len(excluded_doc_ids),
            len(test_doc_ids),
            source_path,
        )
        return excluded_doc_ids
