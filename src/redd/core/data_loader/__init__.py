"""Internal data-loader implementations for ReDD.

Public callers should prefer `redd.create_data_loader` and `redd.DataLoaderBase`.
The registry module owns concrete loader profiles so dataset-specific layout
choices stay behind one internal factory boundary.
"""

from .data_loader_basic import DataLoaderBase
from .data_loader_hf_manifest import DataLoaderHFManifest
from .registry import (
    LoaderProfile,
    create_data_loader,
    get_loader_profile_notes,
    get_loader_profiles,
    get_loader_registry,
)

__all__ = [
    "DataLoaderBase",
    "DataLoaderHFManifest",
    "LoaderProfile",
    "create_data_loader",
    "get_loader_profile_notes",
    "get_loader_profiles",
    "get_loader_registry",
]
