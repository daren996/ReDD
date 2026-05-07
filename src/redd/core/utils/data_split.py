"""
Utilities for global training/test document split.

This module centralizes:
- global config parsing for ``training_data_count``
- deterministic document split (first N for training)
- validation to reject deprecated split keys
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_TRAINING_DATA_COUNT = 100
DEFAULT_TRAINING_DATA_SPLIT = "prefix"
DEFAULT_TRAINING_DATA_SPLIT_SEED = 42


def validate_no_legacy_split_keys(config: Dict[str, Any]) -> None:
    """
    Reject deprecated split-related config keys.

    Deprecated keys:
    - ``train_ratio`` (doc filtering)
    - nested ``training_size`` keys inside legacy split sub-sections
    """
    legacy_paths: List[str] = []

    def _walk(node: Any, path: List[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                cur_path = path + [str(key)]
                if key == "train_ratio":
                    legacy_paths.append(".".join(cur_path))
                if key == "training_size" and path:
                    legacy_paths.append(".".join(cur_path))
                _walk(value, cur_path)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _walk(item, path + [str(idx)])

    _walk(config, [])

    if legacy_paths:
        legacy_desc = ", ".join(sorted(legacy_paths))
        raise ValueError(
            "Deprecated split keys detected: "
            f"{legacy_desc}. Use top-level 'training_data_count' only."
        )


def resolve_training_data_count(
    config: Dict[str, Any],
    default: int = DEFAULT_TRAINING_DATA_COUNT,
) -> int:
    """
    Resolve global training document count from config.

    Args:
        config: Configuration dictionary.
        default: Default count when key is not set.

    Returns:
        Non-negative integer training document count.
    """
    validate_no_legacy_split_keys(config)

    raw_value = config.get("training_data_count", default)
    if raw_value is None:
        return default

    try:
        count = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid training_data_count={raw_value!r}; expected non-negative integer."
        ) from exc

    if count < 0:
        raise ValueError(
            f"Invalid training_data_count={count}; expected non-negative integer."
        )
    return count


def resolve_training_data_split(
    config: Dict[str, Any],
    default: str = DEFAULT_TRAINING_DATA_SPLIT,
) -> str:
    """Resolve the global training document split strategy."""
    raw_value = config.get("training_data_split", default)
    strategy = str(raw_value or default).strip().lower()
    if strategy in {"prefix", "first_n", "first-n"}:
        return "prefix"
    if strategy in {"hash", "hashed", "deterministic_hash"}:
        return "hash"
    raise ValueError(
        "training_data_split must be one of: prefix, hash; "
        f"got {raw_value!r}."
    )


def resolve_training_data_split_seed(
    config: Dict[str, Any],
    default: int = DEFAULT_TRAINING_DATA_SPLIT_SEED,
) -> int:
    """Resolve the seed for deterministic non-prefix training splits."""
    raw_value = config.get("training_data_split_seed")
    if raw_value is None:
        raw_value = config.get("project_seed")
    if raw_value is None and isinstance(config.get("project"), dict):
        raw_value = config["project"].get("seed")
    if raw_value is None:
        raw_value = default

    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid training_data_split_seed={raw_value!r}; expected integer."
        ) from exc


def split_doc_ids(
    doc_ids: Sequence[str],
    training_data_count: int,
    *,
    strategy: str = "prefix",
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Deterministically split doc IDs into training and test sets.

    Supported strategies:
    - prefix: first N documents are training
    - hash: deterministic hash sample spread across document IDs
    """
    if training_data_count < 0:
        raise ValueError(
            f"training_data_count must be non-negative, got {training_data_count}."
        )

    ordered_doc_ids = list(doc_ids)
    n_train = min(training_data_count, len(ordered_doc_ids))
    normalized_strategy = str(strategy or "prefix").strip().lower()
    if normalized_strategy in {"prefix", "first_n", "first-n"}:
        train_doc_ids = ordered_doc_ids[:n_train]
    elif normalized_strategy in {"hash", "hashed", "deterministic_hash"}:
        ranked = sorted(
            ordered_doc_ids,
            key=lambda doc_id: hashlib.sha256(
                f"{int(seed)}:{doc_id}".encode("utf-8")
            ).hexdigest(),
        )
        train_doc_ids = ranked[:n_train]
    else:
        raise ValueError(
            "training_data_split must be one of: prefix, hash; "
            f"got {strategy!r}."
        )
    train_set = set(train_doc_ids)
    test_doc_ids = [doc_id for doc_id in ordered_doc_ids if doc_id not in train_set]
    return train_doc_ids, test_doc_ids
