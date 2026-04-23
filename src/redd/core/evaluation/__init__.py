"""Compatibility wrapper for evaluation helpers.

Prefer importing from `redd.exp.evaluation`.
"""

from importlib import import_module

__all__ = ["EvalBasic", "EvalDataPop"]

_EXPORT_MAP = {
    "EvalBasic": "redd.exp.evaluation",
    "EvalDataPop": "redd.exp.evaluation",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
