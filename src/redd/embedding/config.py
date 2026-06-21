from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

DEFAULT_EMBEDDING_STORAGE_FILE = "embeddings.sqlite3"
EMBEDDING_DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def embedding_config_value(
    config: Mapping[str, Any] | None,
    *keys: str,
    default: Any = None,
) -> Any:
    """Return the first non-empty embedding config value for any of the keys."""
    if not isinstance(config, Mapping):
        return default
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return default


def resolve_embedding_storage_path(
    *,
    config: Mapping[str, Any] | None = None,
    storage_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cache_file: str | Path | None = None,
    storage_file: str | Path | None = None,
    out_root: str | Path | None = None,
    loader: Any | None = None,
    dataset_name: str | None = None,
) -> Path | None:
    """Resolve a SQLite embedding cache path from shared embedding config keys.

    Precedence:
    1. Explicit storage path: storage_path / embedding_storage_path
    2. Cache directory/file: cache_dir / embedding_cache_dir / embeddings_cache_dir
    3. storage_file under out_root, or under loader.data_root when no out_root exists
    4. None, allowing EmbeddingManager to fall back to dataset anchors
    """
    explicit_storage_path = embedding_config_value(
        config,
        "storage_path",
        "embedding_storage_path",
        default=storage_path,
    )
    if explicit_storage_path:
        return Path(explicit_storage_path).expanduser().resolve()

    resolved_cache_dir = embedding_config_value(
        config,
        "cache_dir",
        "embedding_cache_dir",
        "embeddings_cache_dir",
        default=cache_dir,
    )
    if resolved_cache_dir:
        cache_path = Path(resolved_cache_dir).expanduser()
        if cache_path.suffix.lower() in EMBEDDING_DB_SUFFIXES:
            return cache_path.resolve()
        resolved_cache_file = embedding_config_value(
            config,
            "cache_file",
            "embedding_cache_file",
            default=cache_file,
        )
        file_name = (
            Path(str(resolved_cache_file))
            if resolved_cache_file
            else Path(_dataset_cache_file_name(loader=loader, dataset_name=dataset_name))
        )
        return (cache_path / file_name).resolve()

    resolved_storage_file = embedding_config_value(
        config,
        "storage_file",
        "embedding_storage_file",
        default=storage_file or DEFAULT_EMBEDDING_STORAGE_FILE,
    )
    if out_root is not None:
        storage_file_path = Path(str(resolved_storage_file)).expanduser()
        if storage_file_path.is_absolute():
            return storage_file_path.resolve()
        return (Path(out_root).expanduser() / storage_file_path).resolve()
    if loader is not None and getattr(loader, "data_root", None) is not None:
        storage_file_path = Path(str(resolved_storage_file)).expanduser()
        if storage_file_path.is_absolute():
            return storage_file_path.resolve()
        return (Path(loader.data_root).expanduser() / storage_file_path).resolve()

    return None


def embedding_manager_kwargs(
    config: Mapping[str, Any] | None,
    *,
    default_model: str,
    fallback_api_key: str | None = None,
) -> dict[str, Any]:
    """Build common EmbeddingManager kwargs from global or stage config keys."""
    api_key = embedding_config_value(
        config,
        "api_key",
        "embedding_api_key",
        default=fallback_api_key,
    )
    kwargs: dict[str, Any] = {
        "model": str(
            embedding_config_value(
                config,
                "model",
                "embedding_model",
                default=default_model,
            )
        ),
        "api_key": api_key,
    }
    provider = embedding_config_value(config, "provider", "embedding_provider")
    if provider:
        kwargs["provider"] = provider
    base_url = embedding_config_value(config, "base_url", "embedding_base_url")
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _dataset_cache_file_name(*, loader: Any | None, dataset_name: str | None) -> str:
    resolved_dataset_name = str(dataset_name or "").strip()
    if not resolved_dataset_name and loader is not None:
        data_root = getattr(loader, "data_root", None)
        if data_root is not None:
            resolved_dataset_name = Path(data_root).name
    if not resolved_dataset_name:
        return DEFAULT_EMBEDDING_STORAGE_FILE
    return f"{resolved_dataset_name}.embeddings.sqlite3"
