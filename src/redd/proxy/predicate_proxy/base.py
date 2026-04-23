from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from redd.optimizations.doc_filtering.base import FilterResult

__all__ = ["PredicateProxyBase", "FilterResult"]


class PredicateProxyBase(ABC):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._name = self.__class__.__name__

    @abstractmethod
    def filter(self, query_id: str, doc_ids: List[str], **kwargs) -> FilterResult:
        ...

    def __call__(self, query_id: str, doc_ids: List[str], **kwargs) -> FilterResult:
        return self.filter(query_id=query_id, doc_ids=doc_ids, **kwargs)
