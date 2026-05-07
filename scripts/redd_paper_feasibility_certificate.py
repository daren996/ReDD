#!/usr/bin/env python3
"""Certify whether the current output verifies the paper or proves it cannot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORRECTION_REQUIREMENTS = {
    "table2_cuad_chunk_merge",
    "table3_false_positive_overhead",
    "sec623_alpha_threshold_effect",
    "fig2_accuracy_cost_tradeoff",
    "fig3_lambda_sweep",
    "fig4_calibration_size_sweep",
    "fig5_training_size_sweep",
    "fig6_human_vs_llm_labels",
}
VARIANT_REQUIREMENTS = {"one_to_many_chunk_to_table", "density_sweep"}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def build_certificate(output_root: Path) -> dict[str, Any]:
    reports = output_root / "reports"
    audit = _read_json(reports / "redd_paper_completion_audit.json", {})
    if not isinstance(audit, dict):
        audit = {}
    hard_blocker_ids = {item.get("id") for item in audit.get("hard_blockers", []) if isinstance(item, dict)}
    failing_ids = {item.get("experiment_id") for item in audit.get("failing", []) if isinstance(item, dict)}

    correction_blocked = bool(
        CORRECTION_REQUIREMENTS & failing_ids
        and {
            "missing_hidden_states",
            "missing_correction_eval",
            "missing_scape_outputs",
        }
        <= hard_blocker_ids
    )
    variants_blocked = bool(VARIANT_REQUIREMENTS & failing_ids)
    exact_dataset_blocked = "non_exact_dataset_setup" in hard_blocker_ids

    proves_cannot_verify_current_output = bool(
        not audit.get("achieved")
        and correction_blocked
        and variants_blocked
        and exact_dataset_blocked
    )
    return {
        "output_root": str(output_root),
        "paper_verification_achieved": bool(audit.get("achieved")),
        "fallback_cannot_verify_current_output": proves_cannot_verify_current_output,
        "conclusion": (
            "paper_verified"
            if audit.get("achieved")
            else (
                "current_output_cannot_verify_all_paper_claims"
                if proves_cannot_verify_current_output
                else "inconclusive"
            )
        ),
        "evidence": {
            "completion_audit": str(reports / "redd_paper_completion_audit.json"),
            "failing_requirements": sorted(failing_ids),
            "hard_blockers": sorted(str(item) for item in hard_blocker_ids if item),
            "correction_blocked": correction_blocked,
            "variants_blocked": variants_blocked,
            "exact_dataset_blocked": exact_dataset_blocked,
        },
        "required_to_verify": [
            "Exact paper dataset split/artifacts.",
            "Hidden-state extraction artifacts.",
            "Classifier training and classifier evaluation artifacts.",
            "SCAPE/SCAPE-Hyb correction evaluation artifacts with ACCpop/FPRpop.",
            "Alpha, lambda, calibration-size, training-size, and label-source sweeps.",
            "CUAD chunk-merge, one-to-many, and density variant outputs.",
        ],
    }


def _write_markdown(path: Path, certificate: dict[str, Any]) -> None:
    evidence = certificate["evidence"]
    lines = [
        "# FastReDD Feasibility Certificate",
        "",
        f"Conclusion: `{certificate['conclusion']}`",
        f"Paper verification achieved: `{certificate['paper_verification_achieved']}`",
        f"Fallback cannot-verify certificate: `{certificate['fallback_cannot_verify_current_output']}`",
        "",
        "## Evidence",
        "",
        f"- Completion audit: `{evidence['completion_audit']}`",
        f"- Correction blocked: {evidence['correction_blocked']}",
        f"- Variants blocked: {evidence['variants_blocked']}",
        f"- Exact dataset blocked: {evidence['exact_dataset_blocked']}",
        f"- Hard blockers: {evidence['hard_blockers']}",
        "",
        "## Failing Requirements",
        "",
    ]
    lines.extend(f"- `{item}`" for item in evidence["failing_requirements"])
    lines.extend(["", "## Required To Verify", ""])
    lines.extend(f"- {item}" for item in certificate["required_to_verify"])
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    certificate = build_certificate(output_root)
    json_path = reports / "redd_paper_feasibility_certificate.json"
    md_path = reports / "redd_paper_feasibility_certificate.md"
    json_path.write_text(json.dumps(certificate, indent=2) + "\n")
    _write_markdown(md_path, certificate)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Conclusion: {certificate['conclusion']}")
    return 0 if certificate["paper_verification_achieved"] or certificate["fallback_cannot_verify_current_output"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
