from __future__ import annotations

import argparse

from .common import add_experiment_args, add_output_args, add_selection_args, run_and_print


def build_evaluation_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    add_experiment_args(
        parser,
        default_config="configs/siliconflow_qwen30B.yaml",
        experiment_required=False,
        experiment_default="wine_reddv0",
        experiment_flag="--exp",
    )
    add_selection_args(parser)
    add_output_args(parser)
    return parser


def run_evaluation(args: argparse.Namespace) -> int:
    from redd.orchestration.runners import run_evaluation

    args.experiment = args.exp
    return run_and_print(args, run_evaluation)


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    evaluation_parser = subparsers.add_parser(
        "evaluation",
        parents=[build_evaluation_parser(add_help=False)],
        help="Run experiment-side evaluation workflows",
    )
    evaluation_parser.set_defaults(handler=run_evaluation)

    return parser


def run(args: argparse.Namespace) -> int:
    handler = getattr(args, "handler", None)
    if handler is None:
        raise ValueError("Missing exp workflow handler")
    return handler(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
