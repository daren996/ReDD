#!/usr/bin/env python3
"""Build a prompt-to-artifact completion audit for FastReDD paper claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_EXPERIMENTS = [
    ("fig1_pipeline_overview", "Figure 1", "Pipeline implementation/documentation exists."),
    ("table1_dataset_setup", "Table 1 / Sec. 6.1", "Exact dataset/query setup is available."),
    ("table2_data_extraction_accuracy", "Table 2", "Data-extraction accuracy can be measured."),
    ("table2_cuad_chunk_merge", "Table 2 / Sec. 6.2.1", "CUAD chunk merge result exists."),
    ("table3_false_positive_overhead", "Table 3", "False-positive correction overhead is measured."),
    ("sec623_alpha_threshold_effect", "Sec. 6.2.3 / Figure 2", "Alpha threshold effect is measured."),
    ("fig2_accuracy_cost_tradeoff", "Figure 2", "Accuracy-cost curves are measured."),
    ("fig3_lambda_sweep", "Figure 3", "Lambda sweep is measured."),
    ("fig4_calibration_size_sweep", "Figure 4", "Calibration-size sweep is measured."),
    ("fig5_training_size_sweep", "Figure 5", "Training-size sweep is measured."),
    ("fig6_human_vs_llm_labels", "Figure 6", "Human-vs-LLM label comparison is measured."),
    ("table4_schema_discovery", "Table 4 / Sec. 6.3", "Schema-discovery ablation is measured."),
    ("one_to_many_chunk_to_table", "Figure 7 / Sec. 6.4.1", "One-to-many chunk/table experiment is measured."),
    ("density_sweep", "Figure 8 / Sec. 6.4.2", "Information-density sweep is measured."),
    ("runtime_token_accounting", "Sec. 6.4.3", "Runtime and token usage are measured."),
]

PASSING_STATUSES = {"supported", "not_experimental", "analogous_supported"}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _by_id(items: list[dict[str, Any]], key: str = "experiment_id") -> dict[str, dict[str, Any]]:
    return {str(item[key]): item for item in items if isinstance(item, dict) and item.get(key)}


def build_audit(output_root: Path, dataset_root: Path, evidence_mode: str) -> dict[str, Any]:
    reports = output_root / "reports"
    suite = _read_json(reports / "redd_paper_experiment_suite.json", {})
    claims = _read_json(reports / "redd_paper_claim_audit.json", {})
    gate = _read_json(reports / "redd_paper_completion_gate.json", {})
    analogous = _read_json(reports / "redd_paper_analogous_results.json", {})
    provider_smoke = _read_json(reports / "provider_smoke" / "provider_smoke.json", {})

    suite_by_id = _by_id(suite.get("results", []) if isinstance(suite, dict) else [])
    analogous_by_id = _by_id(analogous.get("results", []) if isinstance(analogous, dict) else [])
    claim_items = claims.get("claims", []) if isinstance(claims, dict) else []

    checklist = []
    for experiment_id, paper_ref, requirement in EXPECTED_EXPERIMENTS:
        item = suite_by_id.get(experiment_id, {})
        status = str(item.get("status") or "missing")
        checklist.append(
            {
                "experiment_id": experiment_id,
                "paper_ref": paper_ref,
                "requirement": requirement,
                "status": status,
                "evidence_mode": evidence_mode,
                "passing": status in PASSING_STATUSES if evidence_mode == "analogous" else status in {"supported", "not_experimental"},
                "observed": item.get("observed"),
                "comparison": item.get("comparison"),
                "next_action": item.get("next_action"),
                "analogous_evidence": analogous_by_id.get(experiment_id),
            }
        )

    missing_from_suite = [
        experiment_id for experiment_id, _, _ in EXPECTED_EXPERIMENTS if experiment_id not in suite_by_id
    ]
    failing = [item for item in checklist if not item["passing"]]
    artifact_counts = suite.get("artifact_counts", {}) if isinstance(suite, dict) else {}
    global_artifact_counts = suite.get("global_artifact_counts", {}) if isinstance(suite, dict) else {}
    env = suite.get("env", {}) if isinstance(suite, dict) else {}
    hard_blockers = _hard_blockers(
        artifact_counts=artifact_counts,
        global_artifact_counts=global_artifact_counts,
        env=env,
        provider_smoke=provider_smoke,
        checklist=checklist,
    )
    return {
        "objective": (
            "Verify FastReDD paper main experimental conclusions, including all experiment conclusions, figures, "
            "tables, density, one-to-many, and runtime-token claims; exact evidence is preferred, analogous evidence "
            "is allowed only when explicitly marked."
        ),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "evidence_mode": evidence_mode,
        "achieved": not failing and not missing_from_suite and bool(gate.get("pass")),
        "completion_gate": {
            "pass": gate.get("pass"),
            "status_counts": gate.get("status_counts"),
            "path": str(reports / "redd_paper_completion_gate.json"),
        },
        "source_reports": {
            "suite": str(reports / "redd_paper_experiment_suite.json"),
            "claim_audit": str(reports / "redd_paper_claim_audit.json"),
            "analogous_results": str(reports / "redd_paper_analogous_results.json"),
            "provider_smoke": str(reports / "provider_smoke" / "provider_smoke.json"),
        },
        "artifact_counts": artifact_counts,
        "global_artifact_counts": global_artifact_counts,
        "env": env,
        "provider_smoke": provider_smoke,
        "hard_blockers": hard_blockers,
        "missing_from_suite": missing_from_suite,
        "checklist": checklist,
        "claim_audit_status_counts": _status_counts(claim_items, "status"),
        "failing_count": len(failing),
        "failing": failing,
    }


def _hard_blockers(
    *,
    artifact_counts: dict[str, Any],
    global_artifact_counts: dict[str, Any],
    env: dict[str, Any],
    provider_smoke: dict[str, Any],
    checklist: list[dict[str, Any]],
) -> list[dict[str, str]]:
    blockers = []
    if int(artifact_counts.get("hidden_state_files") or 0) == 0:
        blockers.append(
            {
                "id": "missing_hidden_states",
                "evidence": "output_root hidden_state_files=0",
                "impact": "Cannot verify hidden-state classifier, SCAPE, or SCAPE-Hyb correction claims from current output.",
            }
        )
    if int(artifact_counts.get("classifier_eval_json") or 0) == 0 and int(artifact_counts.get("correction_eval_json") or 0) == 0:
        blockers.append(
            {
                "id": "missing_correction_eval",
                "evidence": "classifier_eval_json=0 and correction_eval_json=0",
                "impact": "Cannot compute Table 3 FPRpop or Figures 2-6 correction sweeps.",
            }
        )
    if int(global_artifact_counts.get("scape_or_hyb_json") or 0) == 0:
        blockers.append(
            {
                "id": "missing_scape_outputs",
                "evidence": "global scape_or_hyb_json=0",
                "impact": "No local artifact names or metrics indicate a completed SCAPE/SCAPE-Hyb run anywhere in the repo.",
            }
        )
    if env and not env.get("cuda_available"):
        blockers.append(
            {
                "id": "cuda_unavailable",
                "evidence": "torch importable but cuda_available=false",
                "impact": "Local hidden-state extraction/training workflows cannot be completed on this host as configured.",
            }
        )
    blocked_providers = provider_smoke.get("blocked_providers", []) if isinstance(provider_smoke, dict) else []
    if blocked_providers:
        blockers.append(
            {
                "id": "provider_blockers",
                "evidence": json.dumps(blocked_providers, sort_keys=True),
                "impact": "Some provider-backed reproduction paths are unavailable; use an OK provider or fix credentials/quota.",
            }
        )
    if any(item["experiment_id"] == "table1_dataset_setup" and item["status"] != "supported" for item in checklist):
        blockers.append(
            {
                "id": "non_exact_dataset_setup",
                "evidence": "Table 1 is not exact-supported in the current suite.",
                "impact": "Exact paper tables cannot be reproduced without the original paper split/artifacts.",
            }
        )
    return blockers


def _status_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get(key) or "missing")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# FastReDD Paper Completion Audit",
        "",
        f"Output root: `{audit['output_root']}`",
        f"Dataset root: `{audit['dataset_root']}`",
        f"Evidence mode: `{audit['evidence_mode']}`",
        f"Achieved: `{audit['achieved']}`",
        "",
        "## Gate",
        "",
        f"- Pass: {audit['completion_gate']['pass']}",
        f"- Status counts: {audit['completion_gate']['status_counts']}",
        f"- Failing requirements: {audit['failing_count']}",
        f"- Hard blockers: {len(audit['hard_blockers'])}",
        "",
        "## Hard Blockers",
        "",
    ]
    for blocker in audit["hard_blockers"]:
        lines.append(f"- `{blocker['id']}`: {blocker['evidence']}; {blocker['impact']}")
    lines.extend(
        [
            "",
            "## Checklist",
            "",
            "| Requirement | Paper ref | Status | Passing | Next action |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in audit["checklist"]:
        lines.append(
            "| "
            f"{item['experiment_id']} | {item['paper_ref']} | {item['status']} | "
            f"{item['passing']} | {item.get('next_action') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Missing Or Weak Evidence",
            "",
        ]
    )
    for item in audit["failing"]:
        lines.append(f"- `{item['experiment_id']}` ({item['paper_ref']}): {item['status']}; {item.get('next_action') or ''}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--evidence-mode", choices=("exact", "analogous"), default="exact")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = build_audit(output_root, Path(args.dataset_root), args.evidence_mode)
    json_path = reports / "redd_paper_completion_audit.json"
    md_path = reports / "redd_paper_completion_audit.md"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")
    _write_markdown(md_path, audit)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Achieved: {audit['achieved']}")
    print(f"Failing requirements: {audit['failing_count']}")
    return 0 if audit["achieved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
