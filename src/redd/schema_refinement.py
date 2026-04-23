from __future__ import annotations

from importlib import import_module

from .schema_refine import schema_refinement

__all__ = ["QueryDocumentFilter", "SchemaTailor", "schema_refinement"]

_LAZY_EXPORTS = {
    "QueryDocumentFilter": ".schema_refine",
    "SchemaTailor": ".schema_refine",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
