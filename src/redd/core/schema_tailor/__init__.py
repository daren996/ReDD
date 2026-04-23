"""Internal query-aware schema tailoring components."""

from importlib import import_module

__all__ = ["QueryDocumentFilter", "SchemaTailor"]

_EXPORT_MAP = {
    "QueryDocumentFilter": ".schema_tailor",
    "SchemaTailor": ".schema_tailor",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
