from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import DataPopulator, SchemaGenerator


def data_extraction(
    *,
    data_populator: DataPopulator | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    exp: str | None = None,
    datasets: Sequence[str] | None = None,
    schema_generator: SchemaGenerator | None = None,
    schema_config: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    configure_logging: bool = True,
) -> list[dict[str, Any]]:
    """Run the data extraction stage."""
    populator = _resolve_data_populator(
        data_populator=data_populator,
        config=config,
        config_path=config_path,
        exp=exp,
        api_key=api_key,
        configure_logging=configure_logging,
    )
    return populator.data_extraction(
        datasets=datasets,
        schema_generator=schema_generator,
        schema_config=schema_config,
    )


def _resolve_data_populator(
    *,
    data_populator: DataPopulator | None,
    config: Mapping[str, Any] | None,
    config_path: str | Path | None,
    exp: str | None,
    api_key: str | None,
    configure_logging: bool,
) -> DataPopulator:
    if data_populator is not None:
        return data_populator

    if config is not None:
        return DataPopulator(
            config,
            api_key=api_key,
            configure_logging=configure_logging,
        )

    if config_path is not None and exp is not None:
        return DataPopulator.from_experiment(
            config_path,
            exp,
            api_key=api_key,
            configure_logging=configure_logging,
        )

    raise ValueError(
        "data_extraction requires either `data_populator=`, "
        "`config=`, or both `config_path=` and `exp=`."
    )


__all__ = ["data_extraction"]
