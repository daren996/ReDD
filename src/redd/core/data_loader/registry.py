"""Registry and profile metadata for internal dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .data_loader_basic import DataLoaderBase
from .data_loader_hf_manifest import DataLoaderHFManifest

__all__ = [
    "LoaderProfile",
    "create_data_loader",
    "get_loader_profile_notes",
    "get_loader_profiles",
    "get_loader_registry",
]


@dataclass(frozen=True)
class LoaderProfile:
    """Describe why a loader implementation exists."""

    name: str
    loader_class: type[DataLoaderBase]
    note: str


LOADER_PROFILES: dict[str, LoaderProfile] = {
    "hf_manifest": LoaderProfile(
        name="hf_manifest",
        loader_class=DataLoaderHFManifest,
        note="HuggingFace-style manifest/parquet dataset contract.",
    ),
}


def get_loader_profiles() -> dict[str, LoaderProfile]:
    """Return the supported loader profiles."""
    return dict(LOADER_PROFILES)


def get_loader_registry() -> Dict[str, type[DataLoaderBase]]:
    """Return loader names mapped to implementation classes."""
    return {
        name: profile.loader_class
        for name, profile in LOADER_PROFILES.items()
    }


def get_loader_profile_notes() -> Dict[str, str]:
    """Return human-readable notes about why each loader exists."""
    return {
        name: profile.note
        for name, profile in LOADER_PROFILES.items()
    }


def create_data_loader(
    data_root: str | Path,
    loader_type: str = "hf_manifest",
    loader_config: Dict[str, Any] | None = None,
) -> DataLoaderBase:
    """Create a data loader from the registry-driven loader family."""
    normalized_loader_type = str(loader_type).strip().lower()

    if normalized_loader_type not in LOADER_PROFILES:
        available = ", ".join(sorted(LOADER_PROFILES))
        raise ValueError(
            f"Unknown loader type: '{normalized_loader_type}'. "
            f"Available loaders: {available}"
        )

    profile = LOADER_PROFILES[normalized_loader_type]
    resolved_loader_config = loader_config or {}
    return profile.loader_class(data_root, **resolved_loader_config)
