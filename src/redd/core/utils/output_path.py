"""
Output path helpers for task-based modules.
"""

from pathlib import Path
from typing import Any, Dict

from redd.runtime import resolve_stage_output_root


def build_task_output_root(
    config: Dict[str, Any],
    dataset_task: str,
    module_name: str,
) -> Path:
    """
    Build output root path for a dataset task.
    """
    return resolve_stage_output_root(config, dataset_task, module_name=module_name)
