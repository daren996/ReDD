#!/usr/bin/env python3
"""Run the full ReDD paper verification stack for an output root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main_from_test(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/paper_claim_run_hash_train100_v2")
    parser.add_argument("--dataset-root", default="dataset/canonical")
    parser.add_argument(
        "--evidence-mode",
        choices=("exact", "analogous"),
        default="exact",
        help="Use exact paper evidence only, or allow explicitly marked analogous paper-like experiment evidence.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    commands = [
        [
            python,
            str(repo_root / "scripts" / "redd_paper_experiment_suite.py"),
            "--output-root",
            args.output_root,
            "--dataset-root",
            args.dataset_root,
        ],
        [
            python,
            str(repo_root / "scripts" / "redd_paper_claim_audit.py"),
            "--output-root",
            args.output_root,
        ],
        [
            python,
            str(repo_root / "scripts" / "redd_paper_completion_gate.py"),
            "--output-root",
            args.output_root,
            "--evidence-mode",
            args.evidence_mode,
        ],
        [
            python,
            str(repo_root / "scripts" / "redd_paper_completion_audit.py"),
            "--output-root",
            args.output_root,
            "--dataset-root",
            args.dataset_root,
            "--evidence-mode",
            args.evidence_mode,
        ],
    ]

    codes = [_run(cmd) for cmd in commands]
    if codes[-2] == 0 and codes[-1] == 0:
        return 0
    return 2


def main() -> int:
    return main_from_test()


if __name__ == "__main__":
    raise SystemExit(main())
