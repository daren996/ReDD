from __future__ import annotations

import argparse

from . import correction, datapop, schemagen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="redd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schemagen_parser = subparsers.add_parser(
        "schemagen",
        parents=[schemagen.build_parser(add_help=False)],
        help="Run schema generation",
    )
    schemagen_parser.set_defaults(handler=schemagen.run)

    datapop_parser = subparsers.add_parser(
        "datapop",
        parents=[datapop.build_parser(add_help=False)],
        help="Run data population",
    )
    datapop_parser.set_defaults(handler=datapop.run)

    correction_parser = subparsers.add_parser(
        "correction",
        parents=[correction.build_parser(add_help=False)],
        help="Run correction/classifier workflows",
    )
    correction_parser.set_defaults(handler=correction.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("Missing subcommand handler")
    return handler(args)
