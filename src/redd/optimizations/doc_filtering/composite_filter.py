"""
Composite Filter.

This module provides CompositeFilter that combines multiple filters
with configurable merge strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import DocFilterBase, FilterResult

__all__ = [
    "CompositeFilter",
]


class CompositeFilter(DocFilterBase):
    """
    Composite filter that combines multiple filters.
    
    This filter allows combining multiple filtering strategies with
    configurable merge strategies (union or intersection).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize composite filter.
        
        Args:
            config: Configuration dictionary. Expected keys:
                - filters: List of filter instances to combine
                - merge_strategy: How to combine results ("union" or "intersection")
                    - "union": Exclude docs flagged by ANY filter
                    - "intersection": Exclude docs flagged by ALL filters
        """
        super().__init__(config)
        self.filters: List[DocFilterBase] = self.config.get("filters", [])
        self.merge_strategy = self.config.get("merge_strategy", "union")
        
        logging.info(
            f"[{self._name}:__init__] Initialized with {len(self.filters)} filters, "
            f"strategy={self.merge_strategy}"
        )
    
    def add_filter(self, filter_instance: DocFilterBase) -> None:
        """
        Add a filter to the composite.
        
        Args:
            filter_instance: Filter instance to add.
        """
        self.filters.append(filter_instance)
        logging.info(f"[{self._name}:add_filter] Added filter: {filter_instance.name}")
    
    def remove_filter(self, filter_name: str) -> bool:
        """
        Remove a filter by name.
        
        Args:
            filter_name: Name of the filter to remove.
            
        Returns:
            True if filter was removed, False if not found.
        """
        for i, f in enumerate(self.filters):
            if f.name == filter_name:
                self.filters.pop(i)
                logging.info(f"[{self._name}:remove_filter] Removed filter: {filter_name}")
                return True
        return False
    
    def filter(
        self,
        query_id: str,
        doc_ids: List[str],
        **kwargs,
    ) -> FilterResult:
        """
        Filter documents using all configured filters.
        
        Args:
            query_id: The query identifier.
            doc_ids: List of document IDs to filter.
            **kwargs: Additional arguments passed to all sub-filters.
                
        Returns:
            FilterResult combined from all sub-filters.
        """
        if not self.filters:
            logging.warning(
                f"[{self._name}:filter] No filters configured. "
                f"Returning empty filter result."
            )
            return FilterResult(
                excluded_doc_ids=set(),
                metadata={"filter_name": self._name, "query_id": query_id},
            )
        
        # Get results from all filters
        results = []
        for f in self.filters:
            try:
                result = f.filter(query_id=query_id, doc_ids=doc_ids, **kwargs)
                results.append(result)
                logging.debug(
                    f"[{self._name}:filter] Filter {f.name} excluded "
                    f"{len(result.excluded_doc_ids)} documents"
                )
            except Exception as e:
                logging.error(
                    f"[{self._name}:filter] Error in filter {f.name}: {e}"
                )
                continue
        
        if not results:
            return FilterResult(
                excluded_doc_ids=set(),
                metadata={"filter_name": self._name, "query_id": query_id, "error": "all_filters_failed"},
            )
        
        # Merge results
        merged = results[0]
        for result in results[1:]:
            merged = merged.merge(result, strategy=self.merge_strategy)
        
        # Update metadata
        merged.metadata["filter_name"] = self._name
        merged.metadata["sub_filters"] = [f.name for f in self.filters]
        
        logging.info(
            f"[{self._name}:filter] Combined {len(self.filters)} filters, "
            f"excluded {len(merged.excluded_doc_ids)} documents"
        )
        
        return merged
