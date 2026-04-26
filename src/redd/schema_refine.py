from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .core.schema_tailor import QueryDocumentFilter, SchemaTailor

if TYPE_CHECKING:
    from .api import SchemaGenerator


def schema_refine(
    *,
    schema_generator: SchemaGenerator | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    exp: str | None = None,
    datasets: Sequence[str] | None = None,
    api_key: str | None = None,
    configure_logging: bool = True,
) -> list[dict[str, Any]]:
    """Run query-specific schema refinement."""
    generator = _resolve_schema_generator(
        schema_generator=schema_generator,
        config=config,
        config_path=config_path,
        exp=exp,
        api_key=api_key,
        configure_logging=configure_logging,
    )
    return generator.schema_refine(datasets=datasets)


def schema_refinement(
    *,
    schema_generator: SchemaGenerator | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    exp: str | None = None,
    datasets: Sequence[str] | None = None,
    api_key: str | None = None,
    configure_logging: bool = True,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for `schema_refine`."""
    return schema_refine(
        schema_generator=schema_generator,
        config=config,
        config_path=config_path,
        exp=exp,
        datasets=datasets,
        api_key=api_key,
        configure_logging=configure_logging,
    )


def _resolve_schema_generator(
    *,
    schema_generator: SchemaGenerator | None,
    config: Mapping[str, Any] | None,
    config_path: str | Path | None,
    exp: str | None,
    api_key: str | None,
    configure_logging: bool,
) -> SchemaGenerator:
    from .api import SchemaGenerator

    if schema_generator is not None:
        return schema_generator

    if config is not None:
        return SchemaGenerator(
            refinement_config=config,
            api_key=api_key,
            configure_logging=configure_logging,
        )

    if config_path is not None and exp is not None:
        return SchemaGenerator.from_experiment(
            config_path,
            exp,
            api_key=api_key,
            configure_logging=configure_logging,
        )

    raise ValueError(
        "schema_refine requires either `schema_generator=`, "
        "`config=`, or both `config_path=` and `exp=`."
    )


__all__ = [
    "QueryDocumentFilter",
    "SchemaTailor",
    "schema_refine",
    "schema_refinement",
]
