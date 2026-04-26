from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel

from .constants import TABLE_ASSIGNMENT_KEY


class TableAssignmentOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    table_assignment: str | None = Field(default=None, alias=TABLE_ASSIGNMENT_KEY)


class AttributeExtractionOutput(RootModel[dict[str, Any]]):
    pass


class SchemaUpdateOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    updated_schema: list[dict[str, Any]] = Field(default_factory=list, alias="Updated Schema")


class SchemaGenDocumentOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    res: Any | None = None
    log: list[dict[str, Any]] | None = None


__all__ = [
    "AttributeExtractionOutput",
    "SchemaGenDocumentOutput",
    "SchemaUpdateOutput",
    "TableAssignmentOutput",
]
