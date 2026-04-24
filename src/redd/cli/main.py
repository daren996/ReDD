from __future__ import annotations

import argparse

from . import datapop, exp, preprocessing, schema_refinement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="redd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocessing_parser = subparsers.add_parser(
        "preprocessing",
        parents=[preprocessing.build_parser(add_help=False)],
        help="Run offline query-independent preprocessing",
    )
    preprocessing_parser.set_defaults(handler=preprocessing.run)

    schema_refinement_parser = subparsers.add_parser(
        "schema-refinement",
        parents=[schema_refinement.build_parser(add_help=False)],
        help="Run query-aware schema refinement",
    )
    schema_refinement_parser.set_defaults(handler=schema_refinement.run)

    datapop_parser = subparsers.add_parser(
        "datapop",
        parents=[datapop.build_parser(add_help=False)],
        help="Run final data extraction",
    )
    datapop_parser.set_defaults(handler=datapop.run)

    exp_parser = subparsers.add_parser(
        "exp",
        parents=[exp.build_parser(add_help=False)],
        help="Run experiment and evaluation workflows",
    )
    exp_parser.set_defaults(handler=exp.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("Missing subcommand handler")
    return handler(args)
