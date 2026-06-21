"""Shared CLI helpers for ReDD command modules."""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from redd.orchestration.experiment import normalize_selection

DEFAULT_CONFIG = "configs/examples/ground_truth_demo.yaml"


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    default_config: str = DEFAULT_CONFIG,
    experiment_required: bool = True,
    experiment_default: str | None = None,
    experiment_flag: str = "--experiment",
) -> None:
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument(experiment_flag, type=str, required=experiment_required, default=experiment_default)
    parser.add_argument("--api-key", type=str, default=None)


def add_selection_args(
    parser: argparse.ArgumentParser,
    *,
    include_stages: bool = False,
) -> None:
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        action="append",
        default=None,
        help="Dataset ID to run. Repeat or comma-separate for multiple datasets.",
    )
    parser.add_argument(
        "--query-id",
        "--query-ids",
        dest="query_ids",
        action="append",
        default=None,
        help="Query ID to run. Repeat or comma-separate for multiple queries.",
    )
    if include_stages:
        parser.add_argument(
            "--stage",
            "--stages",
            dest="stages",
            action="append",
            default=None,
            help="Stage to run. Repeat or comma-separate; accepts preprocess/refine/extract aliases.",
        )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the runner payload as JSON.",
    )


def selected_values(args: argparse.Namespace, name: str) -> list[str] | None:
    return normalize_selection(getattr(args, name, None))


def run_and_print(
    args: argparse.Namespace,
    runner: Callable[..., dict[str, Any]],
    *,
    include_stages: bool = False,
) -> int:
    kwargs: dict[str, Any] = {
        "api_key": args.api_key,
        "datasets": selected_values(args, "datasets"),
        "query_ids": selected_values(args, "query_ids"),
    }
    if include_stages:
        kwargs["stages"] = selected_values(args, "stages")

    result = runner(args.config, args.experiment, **kwargs)
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "DEFAULT_CONFIG",
    "add_experiment_args",
    "add_output_args",
    "add_selection_args",
    "run_and_print",
    "selected_values",
]
