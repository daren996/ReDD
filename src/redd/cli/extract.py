from __future__ import annotations

import argparse

from .common import add_experiment_args, add_output_args, add_selection_args, run_and_print


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    add_experiment_args(parser)
    add_selection_args(parser)
    add_output_args(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    from redd.orchestration.runners import run_extract

    return run_and_print(args, run_extract)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
