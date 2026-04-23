"""
Document Chunking Module

This module provides various document chunking algorithms for splitting
documents into smaller segments. The chunked documents are written back
to SQLite database files for use in downstream tasks.

Available chunkers:
- BaseChunker: Abstract base class for all chunking algorithms
- FixedSizeChunker: Fixed-length chunking with optional overlap
- FixedSizeTokenChunker: Token-based chunking with optional overlap
- ParagraphChunker: Paragraph-based chunking (TODO)
- MapReduceChunker: Map-Reduce style chunking (TODO)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base_chunker import BaseChunker
from .fixed_size_chunker import FixedSizeChunker, FixedSizeTokenChunker

if TYPE_CHECKING:
    from ..data_loader import DataLoaderBase

__all__ = [
    "create_chunker",
    "BaseChunker",
    "FixedSizeChunker",
    "FixedSizeTokenChunker",
]


# Chunker registry mapping chunker names to classes
CHUNKER_REGISTRY = {
    "fixed_size": FixedSizeChunker,
    "fixed_token": FixedSizeTokenChunker,
}


def create_chunker(
    config: Dict[str, Any],
    loader: Optional["DataLoaderBase"] = None,
    chunker_type: Optional[str] = None,
) -> BaseChunker:
    """Factory function to create a chunker.
    
    Args:
        config: Configuration dictionary containing:
            - data_main: Base data directory
            - exp_dataset_task_list: List of data paths to process
            - data_loader_type: Type of data loader (required if loader not provided)
            - chunker_type: Type of chunker (can also be specified as argument)
            - chunk_size: Target chunk size (default varies by chunker type)
            - overlap: Overlap between chunks (default varies by chunker type)
            - tokenizer_name: Tokenizer name for token-based chunking
        loader: Optional pre-created DataLoader instance
        chunker_type: Type of chunker to create. Options:
            - "fixed_size": FixedSizeChunker (character-based)
            - "fixed_token": FixedSizeTokenChunker (token-based)
            If not provided, will try to get from config["chunker_type"]
    
    Returns:
        BaseChunker: Instance of the specified chunker
        
    Raises:
        ValueError: If chunker_type is not recognized or not provided
        
    Examples:
        >>> # Using with config only (loader created internally)
        >>> config = {
        ...     "data_main": "dataset/fda_sqlite/",
        ...     "exp_dataset_task_list": ["no_chunk"],
        ...     "data_loader_type": "sqlite",
        ...     "chunker_type": "fixed_size",
        ...     "chunk_size": 4000,
        ...     "overlap": 200,
        ... }
        >>> chunker = create_chunker(config)
        >>> chunker()  # Run chunking
        
        >>> # Using with pre-created loader
        >>> from redd.core.data_loader import create_data_loader
        >>> loader = create_data_loader("dataset/fda_sqlite/no_chunk/default_task")
        >>> chunker = create_chunker(config, loader=loader, chunker_type="fixed_size")
        >>> chunker()
    """
    # Determine chunker type
    if chunker_type is None:
        chunker_type = config.get("chunker_type")
    
    if chunker_type is None:
        available = ", ".join(CHUNKER_REGISTRY.keys())
        logging.error(f"[create_chunker] chunker_type not specified. "
                     f"Available chunkers: {available}")
        raise ValueError(
            f"chunker_type not specified. Provide via argument or config['chunker_type']. "
            f"Available chunkers: {available}"
        )
    
    chunker_type = chunker_type.lower()
    
    if chunker_type not in CHUNKER_REGISTRY:
        available = ", ".join(CHUNKER_REGISTRY.keys())
        logging.error(f"[create_chunker] Unknown chunker type: '{chunker_type}'. "
                     f"Available chunkers: {available}")
        raise ValueError(
            f"Unknown chunker type: '{chunker_type}'. "
            f"Available chunkers: {available}"
        )
    
    chunker_class = CHUNKER_REGISTRY[chunker_type]
    
    logging.info(f"[create_chunker] Creating {chunker_class.__name__}")
    
    return chunker_class(config=config, loader=loader)
