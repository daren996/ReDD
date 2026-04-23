"""Internal data-population implementations for ReDD.

These classes back the public `redd.DataPopulator` and `redd.data_extraction`
entry point. They are not a stable external API.
"""

from importlib import import_module

__all__ = ["DataPop", "create_data_populator"]


_EXPORT_MAP = {
    "DataPop": ".datapop",
    "create_data_populator": ".factory",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
