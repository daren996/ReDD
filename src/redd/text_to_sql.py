from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TextToSQLRequest:
    query: str
    schema: Any
    context: Any | None = None


class TextToSQLAdapter(Protocol):
    def translate(self, request: TextToSQLRequest) -> str:
        """Translate a natural-language request into SQL."""


class StaticSQLAdapter:
    """Minimal adapter boundary for externally managed text-to-SQL integrations."""

    def __init__(self, sql: str) -> None:
        self.sql = sql

    def translate(self, request: TextToSQLRequest) -> str:
        _ = request
        return self.sql


__all__ = ["StaticSQLAdapter", "TextToSQLAdapter", "TextToSQLRequest"]
