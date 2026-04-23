from __future__ import annotations

from typing import Any

__all__ = ["run_pipeline"]


def run_pipeline(*args: Any, **kwargs: Any):
    from .api import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
