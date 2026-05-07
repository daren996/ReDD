from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_metric_parser_extracts_nested_correction_metrics() -> None:
    suite = _load_script("redd_paper_experiment_suite.py")

    metrics = suite._metric_values_from_obj(
        {
            "model": {
                "threshold2correctedaccuracy": {"0": 0.91, "1": 0.97},
                "threshold2extracostrate": {"0": 0.04, "1": 0.12},
                "coverage": 0.95,
            }
        }
    )

    assert metrics["corrected_accuracy"] == [0.91, 0.97]
    assert metrics["extra_cost_rate"] == [0.04, 0.12]
    assert metrics["coverage"] == [0.95]


def test_dataset_setup_reports_insufficient_canonical_split(tmp_path: Path) -> None:
    suite = _load_script("redd_paper_experiment_suite.py")
    dataset_root = tmp_path / "dataset" / "canonical"
    spider_queries = dataset_root / "spider.demo" / "metadata"
    spider_queries.mkdir(parents=True)
    (spider_queries / "queries.json").write_text(json.dumps({"Q1": {}, "Q2": {}}))

    sweep_rows = [
        {
            "artifact_id": "current-001-baseline",
            "rows": [
                {"dataset": "spider.demo", "answer_recall": 1.0, "cell": [1, 1]},
                {"dataset": "spider.demo", "answer_recall": 1.0, "cell": [1, 1]},
            ],
        }
    ]

    result = suite._dataset_setup_result(sweep_rows, dataset_root)

    assert result.status == "partial"
    assert "canonical_query_counts={'spider': 2}" in result.observed
    assert "not sufficient" in result.next_action


def test_completion_gate_fails_on_partial_suite_item(tmp_path: Path) -> None:
    gate = _load_script("redd_paper_completion_gate.py")
    output_root = tmp_path / "outputs" / "paper"
    reports = output_root / "reports"
    reports.mkdir(parents=True)
    (reports / "redd_paper_experiment_suite.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "experiment_id": "table2",
                        "status": "blocked",
                        "observed": "no classifier artifacts",
                        "next_action": "run correction",
                    }
                ]
            }
        )
    )
    (reports / "redd_paper_claim_audit.json").write_text(json.dumps({"claims": []}))

    assert gate.main_from_test(["--output-root", str(output_root)]) == 2
    gate_output = json.loads((reports / "redd_paper_completion_gate.json").read_text())
    assert gate_output["pass"] is False
    assert gate_output["failing"][0]["experiment_id"] == "table2"


def test_completion_gate_passes_when_all_items_are_supported(tmp_path: Path) -> None:
    gate = _load_script("redd_paper_completion_gate.py")
    output_root = tmp_path / "outputs" / "paper"
    reports = output_root / "reports"
    reports.mkdir(parents=True)
    (reports / "redd_paper_experiment_suite.json").write_text(
        json.dumps(
            {
                "results": [
                    {"experiment_id": "fig1", "status": "not_experimental"},
                    {"experiment_id": "table2", "status": "supported"},
                ]
            }
        )
    )
    (reports / "redd_paper_claim_audit.json").write_text(
        json.dumps({"claims": [{"claim_id": "optimizer.baseline", "status": "supported"}]})
    )

    assert gate.main_from_test(["--output-root", str(output_root)]) == 0
    gate_output = json.loads((reports / "redd_paper_completion_gate.json").read_text())
    assert gate_output["pass"] is True
    assert gate_output["failing"] == []


def test_completion_gate_allows_analogous_supported_only_in_analogous_mode(tmp_path: Path) -> None:
    gate = _load_script("redd_paper_completion_gate.py")
    output_root = tmp_path / "outputs" / "paper"
    reports = output_root / "reports"
    reports.mkdir(parents=True)
    (reports / "redd_paper_experiment_suite.json").write_text(
        json.dumps({"results": [{"experiment_id": "table2", "status": "analogous_supported"}]})
    )
    (reports / "redd_paper_claim_audit.json").write_text(json.dumps({"claims": []}))

    assert gate.main_from_test(["--output-root", str(output_root)]) == 2
    assert gate.main_from_test(["--output-root", str(output_root), "--evidence-mode", "analogous"]) == 0


def test_suite_applies_analogous_result_overrides(tmp_path: Path) -> None:
    suite = _load_script("redd_paper_experiment_suite.py")
    output_root = tmp_path / "outputs" / "paper"
    reports = output_root / "reports"
    reports.mkdir(parents=True)
    (reports / "redd_paper_analogous_results.json").write_text(
        json.dumps(
            {
                "evidence_mode": "analogous",
                "results": [
                    {
                        "experiment_id": "table2_data_population_accuracy",
                        "status": "analogous_supported",
                        "observed": "ACCpop=0.91 from analogous LLM run",
                    }
                ],
            }
        )
    )
    original = [
        suite.ExperimentResult(
            "table2_data_population_accuracy",
            "Table 2",
            "blocked",
            "claim",
            "missing",
            "blocked",
            "run",
        )
    ]

    updated = suite._apply_analogous_results(output_root, original)

    assert updated[0].status == "analogous_supported"
    assert "ACCpop=0.91" in updated[0].observed


def test_extraction_analogous_summary_merges_results(tmp_path: Path) -> None:
    summarize = _load_script("redd_paper_analogous_summarize.py")
    path = tmp_path / "redd_paper_analogous_results.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "experiment_id": "table4_schema_discovery",
                        "status": "partial",
                    }
                ]
            }
        )
    )

    summarize._merge_analogous_result(
        path,
        {
            "experiment_id": "table2_data_population_accuracy",
            "status": "analogous_supported",
            "observed": "full extraction",
        },
    )

    payload = json.loads(path.read_text())
    result_ids = {item["experiment_id"] for item in payload["results"]}
    assert result_ids == {"table2_data_population_accuracy", "table4_schema_discovery"}


def test_schema_analogous_summary_reports_partial_attribute_match(tmp_path: Path) -> None:
    summarize = _load_script("redd_paper_analogous_schema_summarize.py")
    dataset_root = tmp_path / "dataset"
    metadata = dataset_root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "schema.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "name": "wine",
                        "columns": [
                            {"name": "winery"},
                            {"name": "appelation"},
                            {"name": "score"},
                        ],
                    }
                ]
            }
        )
    )
    run_root = tmp_path / "run"
    out = run_root / "demo" / "preprocessing" / "artifact"
    out.mkdir(parents=True)
    (out / "res_artifact.json").write_text(
        json.dumps(
            {
                "doc-1": {
                    "res": "Wines",
                    "log": [
                        {
                            "Schema Name": "Wines",
                            "Attributes": [
                                {"winery": "Name of winery."},
                                {"wine_name": "Name of wine."},
                                {"score": "Score."},
                            ],
                        }
                    ],
                }
            }
        )
    )

    summary = summarize.summarize_run(run_root, dataset_root)
    result = summarize._analogous_result(summary)

    assert summary["metrics"]["table_recall"] == 1.0
    assert summary["metrics"]["attribute_recall"] == 2 / 3
    assert summary["metrics"]["semantic_attribute_recall"] == 1.0
    assert summary["metrics"]["missing_attributes"] == ["appelation"]
    assert result["status"] == "analogous_supported"


def test_llm_usage_summary_marks_runtime_token_accounting_supported(tmp_path: Path) -> None:
    summarize = _load_script("redd_paper_llm_usage_summarize.py")
    run_root = tmp_path / "run"
    reports = run_root / "reports"
    reports.mkdir(parents=True)
    (reports / "llm_usage.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "provider": "deepseek",
                        "configured_model": "deepseek-chat",
                        "response_model": "deepseek-v4-flash",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                            "total_tokens": 13,
                        },
                    }
                ),
                json.dumps(
                    {
                        "provider": "deepseek",
                        "configured_model": "deepseek-chat",
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 4,
                            "total_tokens": 24,
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    summary = summarize.summarize_run(run_root)
    result = summarize._analogous_result(summary)

    assert summary["totals"]["calls"] == 2
    assert summary["totals"]["prompt_tokens"] == 30
    assert summary["totals"]["total_tokens"] == 37
    assert result["experiment_id"] == "runtime_token_accounting"
    assert result["status"] == "analogous_supported"


def test_dataset_setup_summary_counts_current_canonical_queries(tmp_path: Path) -> None:
    summarize = _load_script("redd_paper_dataset_setup_summarize.py")
    dataset_root = tmp_path / "dataset" / "canonical"
    meta = dataset_root / "spider.demo" / "metadata"
    meta.mkdir(parents=True)
    (dataset_root / "spider.demo" / "manifest.yaml").write_text("dataset_id: spider.demo\n")
    (meta / "queries.json").write_text(json.dumps({"Q1": {}, "Q2": {}}))

    summary = summarize.summarize_datasets(dataset_root)
    result = summarize._analogous_result({**summary, "summary_path": "summary.json"})

    assert summary["dataset_count"] == 1
    assert summary["total_queries"] == 2
    assert summary["query_counts_by_family"] == {"spider": 2}
    assert result["status"] == "analogous_supported"


def test_completion_audit_maps_expected_artifacts_to_gate_status(tmp_path: Path) -> None:
    audit_mod = _load_script("redd_paper_completion_audit.py")
    output_root = tmp_path / "outputs" / "paper"
    reports = output_root / "reports"
    reports.mkdir(parents=True)
    (reports / "redd_paper_experiment_suite.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "experiment_id": "fig1_pipeline_overview",
                        "paper_ref": "Figure 1",
                        "status": "not_experimental",
                    },
                    {
                        "experiment_id": "table2_data_population_accuracy",
                        "paper_ref": "Table 2",
                        "status": "analogous_supported",
                    },
                ]
            }
        )
    )
    (reports / "redd_paper_claim_audit.json").write_text(
        json.dumps({"claims": [{"claim_id": "optimizer.baseline", "status": "supported"}]})
    )
    (reports / "redd_paper_completion_gate.json").write_text(
        json.dumps({"pass": False, "status_counts": {"missing": 1}})
    )

    audit = audit_mod.build_audit(output_root, tmp_path / "dataset", "analogous")
    by_id = {item["experiment_id"]: item for item in audit["checklist"]}

    assert audit["achieved"] is False
    assert by_id["table2_data_population_accuracy"]["passing"] is True
    assert by_id["table1_dataset_setup"]["status"] == "missing"
    assert audit["failing_count"] > 0


def test_completion_gate_applies_analogous_overrides_to_claim_audit(tmp_path: Path) -> None:
    gate = _load_script("redd_paper_completion_gate.py")
    output_root = tmp_path / "outputs" / "paper"
    reports = output_root / "reports"
    reports.mkdir(parents=True)
    (reports / "redd_paper_experiment_suite.json").write_text(
        json.dumps({"results": [{"experiment_id": "table3_false_positive_overhead", "status": "analogous_supported"}]})
    )
    (reports / "redd_paper_claim_audit.json").write_text(
        json.dumps({"claims": [{"claim_id": "paper.table3_fpr_overhead", "status": "blocked", "evidence": "missing"}]})
    )
    (reports / "redd_paper_analogous_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "experiment_id": "table3_false_positive_overhead",
                        "status": "analogous_supported",
                        "observed": "controlled FPRpop evidence",
                    }
                ]
            }
        )
    )

    assert gate.main_from_test(["--output-root", str(output_root), "--evidence-mode", "analogous"]) == 0
    payload = json.loads((reports / "redd_paper_completion_gate.json").read_text())
    assert payload["pass"] is True
    assert payload["status_counts"]["analogous_supported"] == 2


def test_controlled_analogous_experiments_cover_missing_claims() -> None:
    controlled = _load_script("redd_paper_controlled_analogous_experiments.py")
    summary = controlled._run_controlled()
    results = controlled._analogous_results(summary, Path("summary.json"))
    result_ids = {item["experiment_id"] for item in results}

    assert "fig2_accuracy_cost_tradeoff" in result_ids
    assert "density_sweep" in result_ids
    assert "optimizer.alpha_allocation" in result_ids
    assert all(item["status"] == "analogous_supported" for item in results)


def test_feasibility_certificate_accepts_cannot_verify_fallback(tmp_path: Path) -> None:
    cert_mod = _load_script("redd_paper_feasibility_certificate.py")
    output_root = tmp_path / "outputs" / "paper"
    reports = output_root / "reports"
    reports.mkdir(parents=True)
    (reports / "redd_paper_completion_audit.json").write_text(
        json.dumps(
            {
                "achieved": False,
                "hard_blockers": [
                    {"id": "missing_hidden_states"},
                    {"id": "missing_correction_eval"},
                    {"id": "missing_scape_outputs"},
                    {"id": "non_exact_dataset_setup"},
                ],
                "failing": [
                    {"experiment_id": "table3_false_positive_overhead"},
                    {"experiment_id": "density_sweep"},
                ],
            }
        )
    )

    certificate = cert_mod.build_certificate(output_root)

    assert certificate["paper_verification_achieved"] is False
    assert certificate["fallback_cannot_verify_current_output"] is True
    assert certificate["conclusion"] == "current_output_cannot_verify_all_paper_claims"


def test_verify_all_uses_completion_gate_status(monkeypatch) -> None:
    verify = _load_script("redd_paper_verify_all.py")
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        if cmd[1].endswith("redd_paper_completion_gate.py"):
            return 2
        return 0

    monkeypatch.setattr(verify, "_run", fake_run)

    assert verify.main_from_test(["--output-root", "out", "--dataset-root", "data"]) == 2
    assert len(calls) == 4
    assert calls[-2][1].endswith("redd_paper_completion_gate.py")
    assert calls[-1][1].endswith("redd_paper_completion_audit.py")
