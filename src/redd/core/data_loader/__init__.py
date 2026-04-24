"""Internal data-loader implementations for ReDD.

Public callers should prefer `redd.create_data_loader` and `redd.DataLoaderBase`.
`create_data_loader()` is the stable factory boundary; the concrete classes in
this module remain internal implementation details while the loader family keeps
converging around genuinely distinct storage/layout models.
"""

from pathlib import Path
from typing import Any, Dict

from .data_loader_basic import DataLoaderBase
from .data_loader_cuad import DataLoaderCUAD
from .data_loader_perfile import DataLoaderPerFile
from .data_loader_spider import DataLoaderSpider
from .data_loader_sqlite import DataLoaderSQLite

__all__ = [
    "create_data_loader",
    "get_loader_profile_notes",
    "get_loader_registry",
    "DataLoaderBase",
    "DataLoaderSpider",
    "DataLoaderPerFile",
    "DataLoaderSQLite",
    "DataLoaderCUAD",
]


LOADER_REGISTRY = {
    "spider": DataLoaderSpider,
    "perfile": DataLoaderPerFile,
    "sqlite": DataLoaderSQLite,
    "cuad": DataLoaderCUAD,
}


LOADER_PROFILE_NOTES = {
    "spider": "JSON-style dataset layout with query/schema artifacts stored as files.",
    "perfile": "Per-document file storage where document contents live outside a DB container.",
    "sqlite": "SQLite-backed runtime storage for document/content retrieval and packaged execution.",
    "cuad": "Dataset-specific specialization kept explicit until its quirks are absorbed into shared loader profiles.",
}


def get_loader_registry() -> Dict[str, type[DataLoaderBase]]:
    """Return the currently supported loader family."""
    return dict(LOADER_REGISTRY)


def get_loader_profile_notes() -> Dict[str, str]:
    """Return human-readable notes about why each loader exists."""
    return dict(LOADER_PROFILE_NOTES)


def create_data_loader(
    data_root: str | Path,
    loader_type: str = "sqlite",
    loader_config: Dict[str, Any] | None = None,
) -> DataLoaderBase:
    """Create a data loader from the registry-driven loader family."""
    normalized_loader_type = loader_type.lower()

    if normalized_loader_type not in LOADER_REGISTRY:
        available = ", ".join(LOADER_REGISTRY.keys())
        raise ValueError(
            f"Unknown loader type: '{normalized_loader_type}'. "
            f"Available loaders: {available}"
        )

    loader_class = LOADER_REGISTRY[normalized_loader_type]
    resolved_loader_config = loader_config or {}
    return loader_class(data_root, **resolved_loader_config)
