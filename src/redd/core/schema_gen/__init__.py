"""Internal schema-generation implementations for ReDD.

These classes back the public `redd.SchemaGenerator` and stage entry points.
They are not a stable external API.
"""

from importlib import import_module

__all__ = [
    "SchemaGen",
    "create_schema_generator",
]


_EXPORT_MAP = {
    "SchemaGen": ".schemagen",
    "create_schema_generator": ".factory",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
