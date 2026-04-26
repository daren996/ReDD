"""Internal stage orchestration services.

The public API in `redd.api` is intentionally small. These modules own the
runtime loops that bind configs, loaders, low-level implementations, and
optional strategies together.
"""

from __future__ import annotations

from .data_extraction import (
    build_loader_config,
    resolve_general_schema_source,
    resolve_query_schema_source,
    resolve_schema_source_mode,
    run_data_extraction,
)
from .pipeline import run_pipeline
from .schema import run_schema, run_schema_preprocessing, run_schema_refinement

__all__ = [
    "build_loader_config",
    "resolve_general_schema_source",
    "resolve_query_schema_source",
    "resolve_schema_source_mode",
    "run_data_extraction",
    "run_pipeline",
    "run_schema",
    "run_schema_preprocessing",
    "run_schema_refinement",
]
