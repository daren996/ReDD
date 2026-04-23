"""
Doc Filtering Module for ReDD.

This module provides interfaces and implementations for filtering documents
based on their relevance to queries. Doc filtering improves data population
efficiency by skipping irrelevant documents.

Main components:
- DocFilterBase: Abstract base class for implementing filter strategies
- FilterResult: Container for filtering results
- NoOpFilter: Pass-through filter (default, no filtering)
- EmbeddingFilter: Embedding-based semantic similarity filtering
- CompositeFilter: Combines multiple filters

Factory function:
- create_doc_filter: Create filter instances from configuration

Usage:
    ```python
    from core.doc_filtering import create_doc_filter, FilterResult
    
    # Create from config
    config = {"filter_type": "embedding", "threshold": 0.6}
    filter = create_doc_filter(config)
    
    # Use the filter
    result = filter.filter(query_id="q1", doc_ids=["doc1", "doc2", ...])
    
    # Check which docs to skip
    for doc_id in doc_ids:
        if result.should_skip(doc_id):
            continue
        # Process document
    ```
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import DocFilterBase, FilterResult, NoOpFilter
from .composite_filter import CompositeFilter
from .schema_relevance_filter import SchemaRelevanceFilter

__all__ = [
    # Base classes
    "DocFilterBase",
    "FilterResult",
    # Concrete filters
    "NoOpFilter",
    "CompositeFilter",
    "SchemaRelevanceFilter",
    # Factory function
    "create_doc_filter",
]

# Registry of available filter types
FILTER_REGISTRY: Dict[str, type] = {
    "noop": NoOpFilter,
    "none": NoOpFilter,  # Alias
    "composite": CompositeFilter,
    "schema_relevance": SchemaRelevanceFilter,
    "schema": SchemaRelevanceFilter,  # Alias
}


def create_doc_filter(
    config: Optional[Dict[str, Any]] = None,
    filter_type: Optional[str] = None,
) -> DocFilterBase:
    """
    Factory function to create doc filter instances.
    
    Args:
        config: Configuration dictionary for the filter.
            Expected keys:
            - filter_type: Type of filter to create (if not provided as argument)
            - Other keys depend on the specific filter type
        filter_type: Type of filter to create. Overrides config["filter_type"].
            Available types:
            - "noop" / "none": NoOpFilter (no filtering)
            - "schema_relevance" / "schema": SchemaRelevanceFilter
            - "composite": CompositeFilter
            
    Returns:
        Configured DocFilterBase instance.
        
    Raises:
        ValueError: If filter_type is not recognized.
        
    Example:
        ```python
        # Create embedding filter with custom threshold
        config = {"filter_type": "embedding", "threshold": 0.7}
        filter = create_doc_filter(config)
        
        # Or specify type directly
        filter = create_doc_filter({"threshold": 0.7}, filter_type="embedding")
        
        # Default is NoOpFilter
        filter = create_doc_filter()  # Returns NoOpFilter
        ```
    """
    config = config or {}
    
    # Determine filter type
    ftype = filter_type or config.get("filter_type", "noop")
    ftype = ftype.lower()
    
    if ftype not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown filter type: {ftype}. "
            f"Available types: {list(FILTER_REGISTRY.keys())}"
        )
    
    filter_class = FILTER_REGISTRY[ftype]
    filter_instance = filter_class(config)
    
    logging.info(f"[create_doc_filter] Created {filter_class.__name__} filter")
    
    return filter_instance


def create_composite_filter(
    filter_configs: list,
    merge_strategy: str = "union",
) -> CompositeFilter:
    """
    Create a composite filter from multiple filter configurations.
    
    Args:
        filter_configs: List of filter configurations, each a dict with
            at least "filter_type" key.
        merge_strategy: How to combine filter results.
            - "union": Filter docs flagged by ANY filter (more aggressive)
            - "intersection": Filter docs flagged by ALL filters (more conservative)
            
    Returns:
        Configured CompositeFilter instance.
        
    Example:
        ```python
        configs = [
            {"filter_type": "schema_relevance", "threshold": 0.6},
        ]
        composite = create_composite_filter(configs, merge_strategy="intersection")
        ```
    """
    filters = [create_doc_filter(cfg) for cfg in filter_configs]
    
    composite_config = {
        "filters": filters,
        "merge_strategy": merge_strategy,
    }
    
    return CompositeFilter(composite_config)
