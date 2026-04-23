"""
Base class for doc filtering.

This module provides the abstract base class for implementing various doc filtering
strategies. Doc filtering is used to identify which document chunks are irrelevant
to a given query and can be skipped during data population, improving efficiency.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

__all__ = ["DocFilterBase", "FilterResult", "NoOpFilter"]


class FilterResult:
    """
    Result of a doc filtering operation.
    
    Attributes:
        excluded_doc_ids: Set of document IDs that should be excluded (skipped).
        metadata: Optional dict for additional filtering metadata.
    """
    
    def __init__(
        self,
        excluded_doc_ids: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize FilterResult.
        
        Args:
            excluded_doc_ids: Set of document IDs to exclude (skip).
            metadata: Optional metadata about the filtering process.
        """
        self.excluded_doc_ids: Set[str] = excluded_doc_ids or set()
        self.metadata: Dict[str, Any] = metadata or {}
    
    def __len__(self) -> int:
        """Return number of excluded documents."""
        return len(self.excluded_doc_ids)
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document ID is in the excluded set."""
        return doc_id in self.excluded_doc_ids
    
    def should_skip(self, doc_id: str) -> bool:
        """
        Check if a document should be skipped.
        
        Args:
            doc_id: Document ID to check.
            
        Returns:
            True if the document should be skipped, False otherwise.
        """
        return doc_id in self.excluded_doc_ids
    
    def merge(self, other: "FilterResult", strategy: str = "union") -> "FilterResult":
        """
        Merge with another FilterResult.
        
        Args:
            other: Another FilterResult to merge with.
            strategy: Merge strategy for excluded_doc_ids.
                - "union": Exclude docs that appear in either result
                - "intersection": Exclude docs that appear in both results
                
        Returns:
            New merged FilterResult.
        """
        if strategy == "union":
            merged_ids = self.excluded_doc_ids | other.excluded_doc_ids
        elif strategy == "intersection":
            merged_ids = self.excluded_doc_ids & other.excluded_doc_ids
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        # Merge metadata: preserve original metadata from both results
        merged_metadata = {}
        # Add self's metadata with prefix
        for k, v in self.metadata.items():
            merged_metadata[f"self_{k}"] = v
        # Add other's metadata with prefix
        for k, v in other.metadata.items():
            merged_metadata[f"other_{k}"] = v
        # Add merge info
        merged_metadata["merge_strategy"] = strategy
        merged_metadata["merged_from"] = [
            self.metadata.get("filter_name", "unknown"),
            other.metadata.get("filter_name", "unknown"),
        ]
        
        return FilterResult(
            excluded_doc_ids=merged_ids,
            metadata=merged_metadata,
        )
    
    def __repr__(self) -> str:
        return (
            f"FilterResult(excluded={len(self.excluded_doc_ids)}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )


class DocFilterBase(ABC):
    """
    Abstract base class for doc filtering strategies.
    
    Subclasses should implement the `filter` method to determine which
    document chunks are irrelevant to a query and can be skipped.
    
    Example usage:
        ```python
        class EmbeddingFilter(DocFilterBase):
            def filter(self, query_id, doc_ids, **kwargs):
                # Implement embedding-based filtering
                ...
        
        filter = EmbeddingFilter(config)
        result = filter.filter(query_id="q1", doc_ids=["doc1", "doc2", ...])
        
        for doc_id in doc_ids:
            if result.should_skip(doc_id):
                continue
            # Process document
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the doc filter.
        
        Args:
            config: Configuration dictionary for the filter.
        """
        self.config = config or {}
        self._name = self.__class__.__name__
        logging.info(f"[{self._name}:__init__] Initialized doc filter")
    
    @property
    def name(self) -> str:
        """Return the filter name."""
        return self._name
    
    @abstractmethod
    def filter(
        self,
        query_id: str,
        doc_ids: List[str],
        **kwargs,
    ) -> FilterResult:
        """
        Filter documents based on relevance to the query.
        
        Args:
            query_id: The query identifier.
            doc_ids: List of document IDs to filter.
            **kwargs: Additional arguments for specific filter implementations.
                Common kwargs may include:
                - query_text: The actual query text
                - schema: Schema information for attribute-based filtering
                - data_loader: Data loader for accessing document content
              `  - threshold: Similarity/relevance threshold for filtering
                
        Returns:
            FilterResult containing the set of document IDs to skip.
        """
        raise NotImplementedError(f"[{self._name}:filter] Filter method not implemented")
    
    def filter_for_table_assignment(
        self,
        query_id: str,
        doc_ids: List[str],
        **kwargs,
    ) -> FilterResult:
        """Filter documents specifically for the table assignment phase."""
        return self.filter(query_id=query_id, doc_ids=doc_ids, phase="table_assignment", **kwargs)
    
    def filter_for_attr_extraction(
        self,
        query_id: str,
        doc_ids: List[str],
        **kwargs,
    ) -> FilterResult:
        """Filter documents specifically for the attribute extraction phase."""
        return self.filter(query_id=query_id, doc_ids=doc_ids, phase="attr_extraction", **kwargs)
    
    def __call__(
        self,
        query_id: str,
        doc_ids: List[str],
        **kwargs,
    ) -> FilterResult:
        """Callable interface for filtering."""
        return self.filter(query_id=query_id, doc_ids=doc_ids, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self._name}(config={self.config})"


class NoOpFilter(DocFilterBase):
    """
    Pass-through filter that excludes no documents.
    
    Use when doc filtering is disabled or for baseline comparisons.
    """

    def filter(
        self,
        query_id: str,
        doc_ids: List[str],
        **kwargs,
    ) -> FilterResult:
        """Return empty excluded set (no documents filtered)."""
        return FilterResult(
            excluded_doc_ids=set(),
            metadata={
                "filter_name": self._name,
                "query_id": query_id,
                "num_docs_input": len(doc_ids),
                "num_docs_excluded": 0,
            },
        )
