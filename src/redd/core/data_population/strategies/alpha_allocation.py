"""Alpha-allocation orchestration for unified data extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from redd.optimizations.alpha_allocation.data_extraction_adapter import (
    DataExtractionAlphaAllocator,
)
from redd.optimizations.alpha_allocation.types import AlphaAllocationResult

__all__ = ["AlphaAllocationStrategy"]


class AlphaAllocationStrategy:
    """Thin wrapper that keeps alpha-allocation setup out of `data_extraction.py`."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        data_path: Path,
        loader: Any,
        api_key: Optional[str],
        train_doc_ids: List[str],
        proxy_runtime_enabled: bool,
    ) -> None:
        raw_config = config.get("alpha_allocation", {})
        self.enabled = bool(isinstance(raw_config, dict) and raw_config.get("enabled", False))
        self._allocator: Optional[DataExtractionAlphaAllocator] = None

        if not self.enabled:
            return

        if not proxy_runtime_enabled:
            logging.warning(
                "[AlphaAllocationStrategy] alpha_allocation enabled but proxy_runtime "
                "is disabled; skipping alpha allocation."
            )
            return

        logging.info(
            "[AlphaAllocationStrategy] Enabled (target_recall=%s)",
            raw_config.get("target_recall", 0.95),
        )
        self._allocator = DataExtractionAlphaAllocator(
            extraction_config=config,
            data_path=data_path,
            loader=loader,
            api_key=api_key,
            train_doc_ids=train_doc_ids,
        )

    def allocate_for_query(
        self,
        *,
        query_id: str,
        schema_query: List[Dict[str, Any]],
    ) -> Optional[AlphaAllocationResult]:
        if self._allocator is None:
            return None
        return self._allocator.allocate_for_query(
            query_id=query_id,
            schema_query=schema_query,
        )
