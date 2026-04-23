from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import SchemaGenerator


def preprocessing(
    *,
    schema_generator: SchemaGenerator | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    exp: str | None = None,
    datasets: Sequence[str] | None = None,
    api_key: str | None = None,
    configure_logging: bool = True,
) -> list[dict[str, Any]]:
    """Run the query-independent preprocessing stage."""
    generator = _resolve_schema_generator(
        schema_generator=schema_generator,
        config=config,
        config_path=config_path,
        exp=exp,
        api_key=api_key,
        configure_logging=configure_logging,
    )
    return generator.preprocessing(datasets=datasets)


def _resolve_schema_generator(
    *,
    schema_generator: SchemaGenerator | None,
    config: Mapping[str, Any] | None,
    config_path: str | Path | None,
    exp: str | None,
    api_key: str | None,
    configure_logging: bool,
) -> SchemaGenerator:
    if schema_generator is not None:
        return schema_generator

    if config is not None:
        return SchemaGenerator(
            preprocessing_config=config,
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
        "preprocessing requires either `schema_generator=`, "
        "`config=`, or both `config_path=` and `exp=`."
    )


__all__ = ["preprocessing"]
