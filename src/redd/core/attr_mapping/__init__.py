"""
Attribute Mapping Module

This module provides functionality for mapping partial document chunks to their
relevant schema attributes. When documents are chunked, partial chunks may only
contain information for a subset of the original table's attributes.

Available classes:
- ChunkAttributeMapper: Maps partial chunks to their relevant attributes using LLM
- ChunkAttributeMapperAPI: API-based implementation using remote LLM services

Usage:
    >>> from core.attr_mapping import create_chunk_attr_mapper
    >>> config = {
    ...     "data_main": "dataset/fda_sqlite/",
    ...     "exp_dataset_task_list": ["fixed_size_50k"],
    ...     "data_loader_type": "hf_manifest",
    ...     "mode": "api",  # or "local"
    ...     "llm_model": "deepseek-chat",
    ...     "api_key": "your_api_key",
    ...     "base_url": "https://api.deepseek.com/v1",
    ... }
    >>> mapper = create_chunk_attr_mapper(config)
    >>> mapper()  # Run attribute mapping for partial chunks
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import ChunkAttributeMapperBase
from .chunk_attr_mapper import ChunkAttributeMapper
from .chunk_attr_mapper_api import ChunkAttributeMapperAPI

if TYPE_CHECKING:
    from ..data_loader import DataLoaderBase

__all__ = [
    "create_chunk_attr_mapper",
    "ChunkAttributeMapperBase",
    "ChunkAttributeMapper",
    "ChunkAttributeMapperAPI",
]


# Mapper registry
MAPPER_REGISTRY = {
    "local": ChunkAttributeMapper,
    "api": ChunkAttributeMapperAPI,
    "deepseek": ChunkAttributeMapperAPI,
    "siliconflow": ChunkAttributeMapperAPI,
    "together": ChunkAttributeMapperAPI,
}


def create_chunk_attr_mapper(
    config: Dict[str, Any],
    loader: Optional["DataLoaderBase"] = None,
    mapper_type: Optional[str] = None,
) -> ChunkAttributeMapperBase:
    """
    Factory function to create a chunk attribute mapper.
    
    Args:
        config: Configuration dictionary containing:
            - data_main: Base data directory
            - exp_dataset_task_list: List of data paths to process
            - data_loader_type: Type of data loader
            - mode: Type of mapper ("local" or "api")
            - llm_model: LLM model name
            - For API mode: api_key, base_url
            - For local mode: llm_model_path
        loader: Optional pre-created DataLoader instance
        mapper_type: Type of mapper. Options:
            - "local": Local GPU inference
            - "api": API-based inference (DeepSeek, SiliconFlow, Together, etc.)
            If not provided, will try to get from config["mode"]
    
    Returns:
        ChunkAttributeMapperBase: Instance of the specified mapper
        
    Raises:
        ValueError: If mapper_type is not recognized
        
    Examples:
        >>> # API-based mapper
        >>> config = {
        ...     "data_main": "dataset/fda_sqlite/",
        ...     "exp_dataset_task_list": ["fixed_size_50k"],
        ...     "data_loader_type": "hf_manifest",
        ...     "mode": "api",
        ...     "llm_model": "deepseek-chat",
        ...     "api_key": "your_api_key",
        ...     "base_url": "https://api.deepseek.com/v1",
        ... }
        >>> mapper = create_chunk_attr_mapper(config)
    """
    # Determine mapper type
    if mapper_type is None:
        mapper_type = config.get("mode", "api")
    
    mapper_type = mapper_type.lower()
    
    if mapper_type not in MAPPER_REGISTRY:
        available = ", ".join(MAPPER_REGISTRY.keys())
        logging.error(f"[create_chunk_attr_mapper] Unknown mapper type: '{mapper_type}'. "
                     f"Available mappers: {available}")
        raise ValueError(
            f"Unknown mapper type: '{mapper_type}'. "
            f"Available mappers: {available}"
        )
    
    mapper_class = MAPPER_REGISTRY[mapper_type]
    
    logging.info(f"[create_chunk_attr_mapper] Creating {mapper_class.__name__}")
    
    return mapper_class(config=config, loader=loader)
