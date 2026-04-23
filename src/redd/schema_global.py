from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .core.utils.constants import PATH_TEMPLATES

if TYPE_CHECKING:
    from .api import SchemaGenerator


@dataclass(frozen=True)
class GlobalSchemaArtifacts:
    schema_path: Path
    source_result_path: Path
    qid: str | None = None
    query: str | None = None


def schema_global(
    *,
    schema_generator: SchemaGenerator | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    exp: str | None = None,
    datasets: Sequence[str] | None = None,
    api_key: str | None = None,
    configure_logging: bool = True,
) -> list[dict[str, Any]]:
    """Run query-independent schema extraction."""
    generator = _resolve_schema_generator(
        schema_generator=schema_generator,
        config=config,
        config_path=config_path,
        exp=exp,
        api_key=api_key,
        configure_logging=configure_logging,
    )
    return generator.schema_global(datasets=datasets)


def resolve_global_schema_path(
    out_root: str | Path,
    param_str: str,
    *,
    qid: str | None = None,
) -> Path:
    return Path(out_root) / PATH_TEMPLATES.schema_general(param_str, qid)


def discover_global_schema(
    *,
    config: Mapping[str, Any],
    res_dict: Mapping[str, Any],
    doc_dict: Mapping[str, Any],
    out_root: str | Path,
    param_str: str,
    qid: str | None = None,
    query: str | None = None,
) -> GlobalSchemaArtifacts:
    from .core.utils import output_utils

    out_root_path = Path(out_root)
    schema_path = resolve_global_schema_path(out_root_path, param_str, qid=qid)
    source_result_path = out_root_path / PATH_TEMPLATES.schema_gen_result_general(param_str)

    output_utils.create_general_schema(
        dict(config),
        dict(res_dict),
        dict(doc_dict),
        str(out_root_path),
        param_str,
        qid=qid,
        query=query,
    )
    return GlobalSchemaArtifacts(
        schema_path=schema_path,
        source_result_path=source_result_path,
        qid=qid,
        query=query,
    )


def load_global_schema(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


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
        "schema_global requires either `schema_generator=`, "
        "`config=`, or both `config_path=` and `exp=`."
    )


__all__ = [
    "GlobalSchemaArtifacts",
    "discover_global_schema",
    "load_global_schema",
    "resolve_global_schema_path",
    "schema_global",
]
