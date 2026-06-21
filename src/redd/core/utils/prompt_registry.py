from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PromptSpec:
    id: str
    filename: str
    purpose: str
    input_fields: tuple[str, ...]
    output_schema: str
    used_by: tuple[str, ...]
    owner: str
    aliases: tuple[str, ...] = ()

    @property
    def metadata_filename(self) -> str:
        return f"{Path(self.filename).stem}.yaml"


SCHEMA_GENERATION_PROMPT_ID = "schemagen"
DATA_EXTRACTION_TABLE_PROMPT_ID = "data_extraction_table"
DATA_EXTRACTION_ATTR_PROMPT_ID = "data_extraction_attr"
SEMANTIC_COMPARISON_PROMPT_ID = "data_extraction_cmp_str"
GENERAL_SCHEMA_REVISE_PROMPT_ID = "general_schema_revise"
SCHEMA_TAILOR_PROMPT_ID = "schema_tailor"


PROMPT_SPECS: Mapping[str, PromptSpec] = {
    SCHEMA_GENERATION_PROMPT_ID: PromptSpec(
        id=SCHEMA_GENERATION_PROMPT_ID,
        filename="schemagen.txt",
        purpose="Iteratively generate or refine relational schemas from natural-language documents.",
        input_fields=("Document", "Query", "Record of Schema", "General Schema"),
        output_schema="redd.core.utils.structured_outputs.SchemaGenDocumentOutput",
        used_by=("schema_generation", "schema_refinement"),
        owner="redd.schema",
        aliases=("schema_generation",),
    ),
    DATA_EXTRACTION_TABLE_PROMPT_ID: PromptSpec(
        id=DATA_EXTRACTION_TABLE_PROMPT_ID,
        filename="data_extraction_table_json.txt",
        purpose="Assign a document to the best matching table in a provided schema.",
        input_fields=("Document", "Schema"),
        output_schema="redd.core.utils.structured_outputs.TableAssignmentOutput",
        used_by=("data_extraction.table_assignment",),
        owner="redd.data_extraction",
        aliases=("data_extraction_table_json", "table_assignment"),
    ),
    DATA_EXTRACTION_ATTR_PROMPT_ID: PromptSpec(
        id=DATA_EXTRACTION_ATTR_PROMPT_ID,
        filename="data_extraction_attr_json.txt",
        purpose="Extract one target attribute value from a document under a selected schema.",
        input_fields=("Document", "Schema", "Target Attribute"),
        output_schema="redd.core.utils.structured_outputs.AttributeExtractionOutput",
        used_by=("data_extraction.attribute_extraction", "proxy_runtime.oracle"),
        owner="redd.data_extraction",
        aliases=("data_extraction_attr_json", "attribute_extraction"),
    ),
    SEMANTIC_COMPARISON_PROMPT_ID: PromptSpec(
        id=SEMANTIC_COMPARISON_PROMPT_ID,
        filename="eval_data_extraction_cmp_str.txt",
        purpose="Judge whether a predicted attribute value semantically matches ground truth.",
        input_fields=("Prediction", "Ground Truth"),
        output_schema="json_object[Reasoning,result]",
        used_by=("evaluation.semantic_comparison",),
        owner="redd.evaluation",
        aliases=("eval_data_extraction_cmp_str", "cmp_str", "semantic_comparison"),
    ),
    GENERAL_SCHEMA_REVISE_PROMPT_ID: PromptSpec(
        id=GENERAL_SCHEMA_REVISE_PROMPT_ID,
        filename="general_schema_revise.txt",
        purpose="Revise generated general schema names and attributes using example documents.",
        input_fields=("Schema", "Example Query"),
        output_schema="redd.core.utils.structured_outputs.SchemaUpdateOutput",
        used_by=("schema_generation.general_schema_revise",),
        owner="redd.schema",
        aliases=("schema_revise",),
    ),
    SCHEMA_TAILOR_PROMPT_ID: PromptSpec(
        id=SCHEMA_TAILOR_PROMPT_ID,
        filename="schema_tailor.txt",
        purpose="Remove query-irrelevant attributes from generated schemas.",
        input_fields=("Schema", "Query"),
        output_schema="redd.core.utils.structured_outputs.SchemaUpdateOutput",
        used_by=("schema_generation.schema_tailor", "schema_tailor.engine"),
        owner="redd.schema",
        aliases=("tailor_schema",),
    ),
}


DEFAULT_SCHEMA_PROMPT_ID = SCHEMA_GENERATION_PROMPT_ID
DEFAULT_DATA_EXTRACTION_PROMPTS = {
    "prompt_table": DATA_EXTRACTION_TABLE_PROMPT_ID,
    "prompt_attr": DATA_EXTRACTION_ATTR_PROMPT_ID,
}
DEFAULT_EVALUATION_PROMPTS = {
    "data_extraction_cmp_str": SEMANTIC_COMPARISON_PROMPT_ID,
}


def _reference_aliases(spec: PromptSpec) -> set[str]:
    stem = Path(spec.filename).stem
    return {
        spec.id,
        spec.filename,
        stem,
        f"prompts/{spec.filename}",
        f"resources/prompts/{spec.filename}",
        *spec.aliases,
    }


_PROMPT_ALIAS_MAP: dict[str, PromptSpec] = {
    alias: spec
    for spec in PROMPT_SPECS.values()
    for alias in _reference_aliases(spec)
}


def get_prompt_spec(prompt_reference: str | Path) -> PromptSpec | None:
    if isinstance(prompt_reference, Path) and prompt_reference.is_absolute():
        return None
    return _PROMPT_ALIAS_MAP.get(str(prompt_reference).replace("\\", "/"))


def iter_prompt_specs() -> tuple[PromptSpec, ...]:
    return tuple(PROMPT_SPECS.values())


__all__ = [
    "DATA_EXTRACTION_ATTR_PROMPT_ID",
    "DATA_EXTRACTION_TABLE_PROMPT_ID",
    "DEFAULT_DATA_EXTRACTION_PROMPTS",
    "DEFAULT_EVALUATION_PROMPTS",
    "DEFAULT_SCHEMA_PROMPT_ID",
    "GENERAL_SCHEMA_REVISE_PROMPT_ID",
    "PROMPT_SPECS",
    "PromptSpec",
    "SCHEMA_GENERATION_PROMPT_ID",
    "SCHEMA_TAILOR_PROMPT_ID",
    "SEMANTIC_COMPARISON_PROMPT_ID",
    "get_prompt_spec",
    "iter_prompt_specs",
]
