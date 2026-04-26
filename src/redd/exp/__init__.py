"""Future-facing namespace for evaluation and experiments."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "EvalBasic",
    "EvalDataExtraction",
]

_EXPORT_MAP = {
    "EvalBasic": ".evaluation",
    "EvalDataExtraction": ".evaluation",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
