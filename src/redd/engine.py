"""Programmatic extraction engine entry point."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from redd.core.data_extraction.data_extraction import DataExtraction
from redd.core.utils.prompt_registry import DEFAULT_DATA_EXTRACTION_PROMPTS


@dataclass(frozen=True)
class ExtractionRequest:
    documents: Sequence[Mapping[str, Any]]
    schema: Any
    queries: Any | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    output_dir: str | Path | None = None
    dataset_id: str = "engine"


@dataclass(frozen=True)
class ExtractionResult:
    output_dir: Path
    result_paths: list[Path]

    def load_results(self) -> dict[str, Any]:
        return {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in self.result_paths
        }


def run_extraction(
    request: ExtractionRequest,
    *,
    api_key: str | None = None,
) -> ExtractionResult:
    """Run ReDD extraction from in-memory documents and schema payloads."""
    output_dir = (
        Path(request.output_dir)
        if request.output_dir is not None
        else Path(tempfile.mkdtemp(prefix="redd-engine-"))
    )
    data_root = output_dir / "source"
    data_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_engine_extraction_config(request)
    extractor = DataExtraction(config, api_key=api_key)
    extractor._process_dataset(data_root, output_dir)
    result_paths = sorted(
        path
        for path in output_dir.glob("res_tabular_data_*.json")
        if not path.name.endswith("_proxy_decisions.json")
    )
    return ExtractionResult(output_dir=output_dir, result_paths=result_paths)


def build_engine_extraction_config(request: ExtractionRequest) -> dict[str, Any]:
    config = dict(request.config or {})
    if "llm_model" not in config and config.get("model"):
        config["llm_model"] = config["model"]
    if "mode" not in config and config.get("provider"):
        config["mode"] = config["provider"]

    disable_llm = bool(config.get("disable_llm") or config.get("use_ground_truth"))
    if disable_llm:
        config.setdefault("mode", "ground_truth")
        config["disable_llm"] = True
        config["use_ground_truth"] = True
    else:
        config.setdefault("mode", "deepseek")
        config.setdefault("llm_model", "deepseek-chat")
        config.setdefault("prompts", dict(DEFAULT_DATA_EXTRACTION_PROMPTS))

    config.setdefault("res_param_str", "engine")
    config.setdefault("training_data_count", 0)
    config.setdefault("force_rerun", True)
    config["data_loader_type"] = "memory"
    config["data_loader_config"] = {
        "documents": [dict(document) for document in request.documents],
        "schema_json": request.schema,
        "query_json": request.queries,
        "dataset_id": request.dataset_id,
        **dict(config.get("data_loader_config") or {}),
    }
    return config


__all__ = [
    "ExtractionRequest",
    "ExtractionResult",
    "build_engine_extraction_config",
    "run_extraction",
]
