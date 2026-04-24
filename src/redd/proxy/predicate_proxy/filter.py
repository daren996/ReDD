from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import PredicateProxyBase, FilterResult

__all__ = ["PredicateProxyFilter"]


class PredicateProxyFilter(PredicateProxyBase):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def filter(self, query_id: str, doc_ids: List[str], **kwargs) -> FilterResult:
        # TODO: wire in actual predicate-proxy runtime logic
        return FilterResult(
            excluded_doc_ids=set(),
            metadata={
                "filter_name": self._name,
                "query_id": query_id,
                "num_docs_input": len(doc_ids),
                "num_docs_excluded": 0,
                "num_docs_kept": len(doc_ids),
                "stub": True,
            },
        )
