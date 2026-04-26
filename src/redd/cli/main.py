from __future__ import annotations

import argparse

from . import dataset, extract, preprocessing, run, schema_refinement, web


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="redd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        parents=[run.build_parser(add_help=False)],
        help="Run a strict v2 experiment",
    )
    run_parser.set_defaults(handler=run.run)

    preprocessing_parser = subparsers.add_parser(
        "preprocess",
        parents=[preprocessing.build_parser(add_help=False)],
        help="Run offline query-independent preprocessing",
    )
    preprocessing_parser.set_defaults(handler=preprocessing.run)

    schema_refinement_parser = subparsers.add_parser(
        "refine",
        parents=[schema_refinement.build_parser(add_help=False)],
        help="Run query-aware schema refinement",
    )
    schema_refinement_parser.set_defaults(handler=schema_refinement.run)

    extract_parser = subparsers.add_parser(
        "extract",
        parents=[extract.build_parser(add_help=False)],
        help="Run final data extraction",
    )
    extract_parser.set_defaults(handler=extract.run)

    dataset_parser = subparsers.add_parser(
        "dataset",
        parents=[dataset.build_parser(add_help=False)],
        help="Validate and inspect ReDD dataset contracts",
    )
    dataset_parser.set_defaults(handler=dataset.run)

    web_parser = subparsers.add_parser(
        "web",
        parents=[web.build_parser(add_help=False)],
        help="Serve the ReDD web demo",
    )
    web_parser.set_defaults(handler=web.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("Missing subcommand handler")
    return handler(args)
