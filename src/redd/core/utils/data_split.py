"""
Utilities for global training/test document split.

This module centralizes:
- global config parsing for ``training_data_count``
- deterministic document split (first N for training)
- validation to reject deprecated split keys
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_TRAINING_DATA_COUNT = 100


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


def split_doc_ids(
    doc_ids: Sequence[str],
    training_data_count: int,
) -> Tuple[List[str], List[str]]:
    """
    Deterministically split doc IDs into training and test sets.

    Uses a prefix split for reproducibility:
    - first N documents: training
    - remaining documents: test
    """
    if training_data_count < 0:
        raise ValueError(
            f"training_data_count must be non-negative, got {training_data_count}."
        )

    ordered_doc_ids = list(doc_ids)
    n_train = min(training_data_count, len(ordered_doc_ids))
    train_doc_ids = ordered_doc_ids[:n_train]
    test_doc_ids = ordered_doc_ids[n_train:]
    return train_doc_ids, test_doc_ids
