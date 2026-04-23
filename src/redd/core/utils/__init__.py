"""Internal utility package for ReDD.

Keep this package lightweight at import time so higher-level public modules can
be imported without eagerly pulling optional provider dependencies.
"""

__all__: list[str] = []
