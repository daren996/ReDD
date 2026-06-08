from __future__ import annotations

from .exp import build_evaluation_parser, run_evaluation


def main(argv: list[str] | None = None) -> int:
    args = build_evaluation_parser().parse_args(argv)
    return run_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
