#!/usr/bin/env python3
"""Completion gate for validating ReDD paper experiment coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXACT_PASSING_STATUSES = {"supported", "not_experimental"}
ANALOGOUS_PASSING_STATUSES = {"supported", "not_experimental", "analogous_supported"}
NON_PASSING_STATUSES = {"partial", "surrogate_only", "blocked", "missing", "unsupported", "implemented_needs_aggregation"}

CLAIM_TO_EXPERIMENT = {
    "paper.table1_dataset_setup": "table1_dataset_setup",
    "paper.table2_tdp_accuracy": "table2_data_population_accuracy",
    "paper.table3_fpr_overhead": "table3_false_positive_overhead",
    "paper.fig2_accuracy_cost_tradeoff": "fig2_accuracy_cost_tradeoff",
    "paper.fig3_lambda_sweep": "fig3_lambda_sweep",
    "paper.fig4_calibration_size": "fig4_calibration_size_sweep",
    "paper.fig5_training_size": "fig5_training_size_sweep",
    "paper.fig6_label_source": "fig6_human_vs_llm_labels",
    "paper.table4_schema_discovery": "table4_schema_discovery",
    "paper.sec641_one_to_many": "one_to_many_chunk_to_table",
    "paper.sec642_density": "density_sweep",
    "paper.sec643_runtime_tokens": "runtime_token_accounting",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_items(payload: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    key = "results" if kind == "suite" else "claims"
    items = payload.get(key, [])
    return items if isinstance(items, list) else []


def _item_id(item: dict[str, Any], kind: str) -> str:
    return str(item.get("experiment_id") or item.get("claim_id") or f"unknown_{kind}")


def _passing_statuses(evidence_mode: str) -> set[str]:
    if evidence_mode == "analogous":
        return ANALOGOUS_PASSING_STATUSES
    return EXACT_PASSING_STATUSES


def _analogous_overrides(output_root: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(output_root / "reports" / "redd_paper_analogous_results.json")
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return {}
    return {
        str(item["experiment_id"]): item
        for item in results
        if isinstance(item, dict) and item.get("experiment_id")
    }


def _apply_claim_analogous_overrides(
    claim_items: list[dict[str, Any]],
    analogous: dict[str, dict[str, Any]],
    evidence_mode: str,
) -> list[dict[str, Any]]:
    if evidence_mode != "analogous" or not analogous:
        return claim_items
    updated = []
    for item in claim_items:
        claim_id = str(item.get("claim_id") or "")
        experiment_id = CLAIM_TO_EXPERIMENT.get(claim_id, claim_id)
        override = analogous.get(experiment_id) or analogous.get(claim_id)
        if not override or override.get("status") != "analogous_supported":
            updated.append(item)
            continue
        updated_item = dict(item)
        updated_item["status"] = "analogous_supported"
        updated_item["evidence"] = str(override.get("observed") or item.get("evidence") or "")
        updated_item["required_action"] = str(override.get("next_action") or item.get("required_action") or "")
        updated_item["local_blocker"] = str(
            override.get("comparison")
            or "Paper claim covered by analogous evidence; not exact paper reproduction."
        )
        updated_item["analogous_experiment_id"] = experiment_id
        updated.append(updated_item)
    return updated


def _render_markdown(
    output_root: Path,
    suite_items: list[dict[str, Any]],
    claim_items: list[dict[str, Any]],
    evidence_mode: str,
) -> str:
    all_items = [("suite", item) for item in suite_items] + [("claim", item) for item in claim_items]
    status_counts = Counter(str(item.get("status", "unknown")) for _, item in all_items)
    passing_statuses = _passing_statuses(evidence_mode)
    failing = [
        (kind, item)
        for kind, item in all_items
        if str(item.get("status", "unknown")) not in passing_statuses
    ]
    lines = [
        "# ReDD Paper Completion Gate",
        "",
        f"Output root: `{output_root}`",
        "",
        f"Evidence mode: `{evidence_mode}`",
        "",
        "## Objective",
        "",
        "Validate whether the current output can support all FastReDD/ReDD paper claims, including experiments, figures, tables, and major Section 6 conclusions. If it cannot, the report must identify concrete artifact or environment blockers.",
        "",
        "## Success Criteria",
        "",
        "- Every experimental table, figure, and Section 6 claim is represented by a suite or claim item.",
        "- Numeric paper claims are backed by real output artifacts, not oracle/surrogate-only signals.",
        "- The completion gate passes only when every item has a passing status for the selected evidence mode.",
        f"- Passing statuses for this run are: `{sorted(passing_statuses)}`.",
        "- `partial`, `surrogate_only`, `blocked`, `missing`, and `unsupported` are failures.",
        "",
        "## Evidence Sources",
        "",
        "- `redd_paper_experiment_suite.json`: table/figure/experiment coverage and observed metrics.",
        "- `redd_paper_claim_audit.json`: optimizer and paper-claim support audit.",
        "- Current output tree artifact counts, global artifact counts, and generated report files.",
        "",
        f"Pass: `{not failing}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Failing Items",
            "",
            "| Source | Item | Status | Evidence | Required action |",
            "|---|---|---:|---|---|",
        ]
    )
    for kind, item in failing:
        evidence = str(item.get("observed") or item.get("evidence") or "").replace("|", "\\|")
        action = str(item.get("next_action") or item.get("required_action") or "").replace("|", "\\|")
        lines.append(f"| {kind} | `{_item_id(item, kind)}` | {item.get('status')} | {evidence} | {action} |")
    lines.append("")
    return "\n".join(lines)


def main_from_test(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/paper_claim_run_hash_train100_v2")
    parser.add_argument(
        "--evidence-mode",
        choices=("exact", "analogous"),
        default="exact",
        help="Use exact paper evidence only, or allow explicitly marked paper-like analogous experiment evidence.",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    suite = _read_json(reports_dir / "redd_paper_experiment_suite.json")
    audit = _read_json(reports_dir / "redd_paper_claim_audit.json")
    suite_items = _load_items(suite, "suite")
    analogous = _analogous_overrides(output_root)
    claim_items = _apply_claim_analogous_overrides(
        _load_items(audit, "claim"),
        analogous,
        args.evidence_mode,
    )
    all_items = [("suite", item) for item in suite_items] + [("claim", item) for item in claim_items]
    passing_statuses = _passing_statuses(args.evidence_mode)
    failing = [
        {"source": kind, **item}
        for kind, item in all_items
        if str(item.get("status", "unknown")) not in passing_statuses
    ]
    payload = {
        "output_root": str(output_root),
        "evidence_mode": args.evidence_mode,
        "pass": not failing,
        "passing_statuses": sorted(passing_statuses),
        "non_passing_statuses": sorted(NON_PASSING_STATUSES),
        "status_counts": dict(Counter(str(item.get("status", "unknown")) for _, item in all_items)),
        "failing": failing,
    }
    json_path = reports_dir / "redd_paper_completion_gate.json"
    md_path = reports_dir / "redd_paper_completion_gate.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(_render_markdown(output_root, suite_items, claim_items, args.evidence_mode))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Pass: {payload['pass']}")
    print(f"Status counts: {payload['status_counts']}")
    return 0 if payload["pass"] else 2


def main() -> int:
    return main_from_test()


if __name__ == "__main__":
    raise SystemExit(main())
