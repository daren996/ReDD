from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def _load_model_catalog() -> dict[str, Any]:
    catalog_resource = resources.files("redd.resources").joinpath("model_catalog.json")
    with catalog_resource.open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    if not isinstance(catalog, dict):
        raise TypeError("Model catalog root must be a mapping.")
    return catalog


def get_model_catalog() -> dict[str, Any]:
    """Return the packaged catalog of model choices supported by the UI."""
    return deepcopy(_load_model_catalog())
