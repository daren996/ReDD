"""Document-filter orchestration for the unified data-population stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from redd.optimizations.doc_filtering import create_doc_filter
from redd.optimizations.doc_filtering.runtime import (
    normalize_doc_filter_config,
    run_query_doc_filter,
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
