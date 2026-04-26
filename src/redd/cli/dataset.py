from __future__ import annotations

import argparse


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    subparsers = parser.add_subparsers(dest="dataset_command", required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate a ReDD dataset registry")
    validate_parser.add_argument("manifest", type=str)
    validate_parser.set_defaults(handler=validate)
    return parser


def validate(args: argparse.Namespace) -> int:
    from redd.dataset_contract import print_validation_report, validate_registry

    report = validate_registry(args.manifest)
    print_validation_report(report)
    return 0 if report["valid"] else 1


def run(args: argparse.Namespace) -> int:
    handler = getattr(args, "handler", None)
    if handler is None:
        raise ValueError("Missing dataset subcommand handler")
    return handler(args)
