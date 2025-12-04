"""
Data Loader Module

This module contains classes and utilities for loading and processing
various types of datasets including Spider, Bird, and Galois datasets.
"""

from pathlib import Path
from typing import Dict, Any

from .data_loader_basic import DataLoaderBase
from .data_loader_spider import DataLoaderSpider
from .data_loader_perfile import DataLoaderPerFile
from .data_loader_sqlite import DataLoaderSQLite
from .data_loader_cuad import DataLoaderCUAD
# from .data_loader_swde import DataLoaderSwde

__all__ = [
    "create_data_loader",
    "DataLoaderBase",
    "DataLoaderSpider",
    "DataLoaderPerFile",
    "DataLoaderSQLite",
    "DataLoaderCUAD",
    # "DataLoaderSwde",
]


# Loader registry mapping loader names to classes
LOADER_REGISTRY = {
    "spider": DataLoaderSpider,
    "perfile": DataLoaderPerFile,
    "sqlite": DataLoaderSQLite,
    "cuad": DataLoaderCUAD,
    # Add more loaders here as they are implemented
}


def create_data_loader(
    data_root: str | Path,
    loader_type: str = "sqlite",
    loader_config: Dict[str, Any] | None = None,
) -> DataLoaderBase:
    """Factory function to create a data loader based on type.
    
    Args:
        data_root: Root directory of the dataset
        loader_type: Type of loader to create. Options:
            - "spider": DataLoaderSpider (default) - for JSON-based datasets
            - "perfile": DataLoaderPerFile - for per-file document storage
            - "sqlite": DataLoaderSQLite - for SQLite database storage
        loader_config: Additional configuration dict for the loader (optional)
            May include keys like:
            - "filemap": custom file mapping
            - "file_extension": for perfile loader (default: ".txt")
            - "encoding": for perfile loader (default: "utf-8")
    
    Returns:
        DataLoaderBase: Instance of the specified loader
        
    Raises:
        ValueError: If loader_type is not recognized
        
    Examples:
        >>> # Create a Spider loader (default)
        >>> loader = create_data_loader("dataset/spider/college_2")
        
        >>> # Create a per-file loader with custom extension
        >>> loader = create_data_loader(
        ...     "dataset/fda/documents",
        ...     loader_type="perfile",
        ...     loader_config={"file_extension": ".html"}
        ... )
        
        >>> # Create a SQLite loader
        >>> loader = create_data_loader(
        ...     "dataset/spider_chunked/college_2",
        ...     loader_type="sqlite"
        ... )
    """
    loader_type = loader_type.lower()
    
    if loader_type not in LOADER_REGISTRY:
        available = ", ".join(LOADER_REGISTRY.keys())
        raise ValueError(
            f"Unknown loader type: '{loader_type}'. "
            f"Available loaders: {available}"
        )
    
    loader_class = LOADER_REGISTRY[loader_type]
    loader_config = loader_config or {}
    
    return loader_class(data_root, **loader_config)
