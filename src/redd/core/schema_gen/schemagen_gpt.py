"""Backward-compatible alias for the canonical schema generator."""

from .schemagen import SchemaGen


class SchemaGenGPT(SchemaGen):
    """Legacy alias kept for internal compatibility."""


__all__ = ["SchemaGenGPT"]
