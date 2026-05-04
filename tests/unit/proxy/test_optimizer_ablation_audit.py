from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_optimizer_ablation_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "optimizer_ablation.py"
    spec = importlib.util.spec_from_file_location("optimizer_ablation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_consistency_audit_reports_text_pass_gt_fail(tmp_path: Path) -> None:
    module = _load_optimizer_ablation_module()
    root = tmp_path / "demo"
    (root / "data").mkdir(parents=True)
    (root / "metadata").mkdir()

    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "d1",
                "doc_text": "The school reached an average of 513 in math.",
            }
        ]
    ).to_parquet(root / "data" / "documents.parquet", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "d1",
                "record_id": "1",
                "table_id": "scores",
                "column_id": "scores.avg_scr_math",
                "column_name": "avg_scr_math",
                "value": "448",
                "value_type": "string",
                "source_row_id": "1",
            }
        ]
    ).to_parquet(root / "data" / "ground_truth.parquet", index=False)
    (root / "metadata" / "queries.json").write_text(
        """
        {
          "schema_version": "redd.queries.v1",
          "dataset_id": "demo",
          "queries": [
            {
              "query_id": "q1",
              "sql": "SELECT avg_scr_math FROM scores WHERE avg_scr_math >= 473;"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    audit = module._audit_dataset_consistency(
        "demo",
        {"root": str(root), "query_ids": ["q1"]},
    )

    assert audit["conflicts_by_type"] == {"text_pass_gt_fail": 1}
    assert audit["conflicts"][0]["doc_id"] == "d1"
    assert audit["conflicts"][0]["text_values"] == [513.0]
    assert audit["conflicts"][0]["ground_truth_values"] == [448.0]


def test_dataset_consistency_audit_reports_string_gt_pass_text_fail(tmp_path: Path) -> None:
    module = _load_optimizer_ablation_module()
    root = tmp_path / "demo"
    (root / "data").mkdir(parents=True)
    (root / "metadata").mkdir()

    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "d1",
                "doc_text": "Madelyn Nicholson attended the Football game.",
                "source_table": "attendance",
                "source_row_id": "77",
            }
        ]
    ).to_parquet(root / "data" / "documents.parquet", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "d1",
                "record_id": "77",
                "table_id": "attendance",
                "column_id": "attendance.event_name",
                "column_name": "event_name",
                "value": "October Speaker",
                "value_type": "string",
                "source_row_id": "77",
            }
        ]
    ).to_parquet(root / "data" / "ground_truth.parquet", index=False)
    (root / "metadata" / "queries.json").write_text(
        """
        {
          "schema_version": "redd.queries.v1",
          "dataset_id": "demo",
          "queries": [
            {
              "query_id": "q1",
              "sql": "SELECT member_first_name FROM attendance WHERE event_name = 'October Speaker';"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    audit = module._audit_dataset_consistency(
        "demo",
        {"root": str(root), "query_ids": ["q1"]},
    )

    assert audit["conflicts_by_type"] == {"gt_pass_text_fail": 1}
    assert audit["conflicts"][0]["attribute"] == "event_name"
    assert audit["conflicts"][0]["source_table"] == "attendance"
    assert audit["conflicts"][0]["source_row_id"] == "77"
    assert audit["conflicts"][0]["text_values"] == []
    assert audit["conflicts"][0]["ground_truth_values"] == ["October Speaker"]


def test_dataset_consistency_audit_does_not_flag_string_like_when_gt_matches(
    tmp_path: Path,
) -> None:
    module = _load_optimizer_ablation_module()
    root = tmp_path / "demo"
    (root / "data").mkdir(parents=True)
    (root / "metadata").mkdir()

    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "d1",
                "doc_text": "Madelyn Nicholson attended the September Speaker.",
                "source_table": "attendance",
                "source_row_id": "77",
            }
        ]
    ).to_parquet(root / "data" / "documents.parquet", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "d1",
                "record_id": "77",
                "table_id": "attendance",
                "column_id": "attendance.event_name",
                "column_name": "event_name",
                "value": "October Speaker",
                "value_type": "string",
                "source_row_id": "77",
            }
        ]
    ).to_parquet(root / "data" / "ground_truth.parquet", index=False)
    (root / "metadata" / "queries.json").write_text(
        """
        {
          "schema_version": "redd.queries.v1",
          "dataset_id": "demo",
          "queries": [
            {
              "query_id": "q1",
              "sql": "SELECT member_first_name FROM attendance WHERE event_name LIKE '%speaker%';"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    audit = module._audit_dataset_consistency(
        "demo",
        {"root": str(root), "query_ids": ["q1"]},
    )

    assert audit["conflicts_by_type"] == {}
    assert audit["conflicts"] == []


def test_dataset_consistency_audit_uses_nearby_numbers_for_generic_numeric_fields(
    tmp_path: Path,
) -> None:
    module = _load_optimizer_ablation_module()
    root = tmp_path / "demo"
    (root / "data").mkdir(parents=True)
    (root / "metadata").mkdir()

    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "p1",
                "doc_text": "The player's height is 175 cm and weight is 160 lbs.",
            }
        ]
    ).to_parquet(root / "data" / "documents.parquet", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "doc_id": "p1",
                "record_id": "1",
                "table_id": "player",
                "column_id": "player.height",
                "column_name": "height",
                "value": "185",
                "value_type": "string",
                "source_row_id": "1",
            }
        ]
    ).to_parquet(root / "data" / "ground_truth.parquet", index=False)
    (root / "metadata" / "queries.json").write_text(
        """
        {
          "schema_version": "redd.queries.v1",
          "dataset_id": "demo",
          "queries": [
            {
              "query_id": "q1",
              "sql": "SELECT player_name FROM player WHERE height > 180;"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    audit = module._audit_dataset_consistency(
        "demo",
        {"root": str(root), "query_ids": ["q1"]},
    )

    assert audit["conflicts_by_type"] == {"gt_pass_text_fail": 1}
    assert audit["conflicts"][0]["attribute"] == "height"
    assert audit["conflicts"][0]["text_values"] == [175.0]
    assert audit["conflicts"][0]["ground_truth_values"] == [185.0]


def test_dataset_consistency_audit_writes_flat_conflict_exports(tmp_path: Path) -> None:
    module = _load_optimizer_ablation_module()
    audit = {
        "total_conflicts": 1,
        "conflicts_by_type": {"text_pass_gt_fail": 1},
        "datasets": [
            {
                "dataset": "demo",
                "checked_doc_predicates": 1,
                "conflict_count": 1,
                "conflicts_by_type": {"text_pass_gt_fail": 1},
                "conflicts": [
                    {
                        "dataset": "demo",
                        "query_id": "q1",
                        "doc_id": "d1",
                        "attribute": "avg_scr_math",
                        "operator": ">=",
                        "expected": 473,
                        "conflict_type": "text_pass_gt_fail",
                        "text_values": [513.0],
                        "ground_truth_values": [448.0],
                        "text_excerpt": "average of 513 in math",
                    }
                ],
            }
        ],
    }

    module.write_dataset_consistency_audit(tmp_path, audit)

    jsonl_path = tmp_path / "dataset_consistency_audit_conflicts.jsonl"
    csv_path = tmp_path / "dataset_consistency_audit_conflicts.csv"
    jsonl_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert jsonl_rows[0]["conflict_ref"] == "demo::q1::d1::avg_scr_math"
    assert jsonl_rows[0]["text_values"] == [513.0]
    assert csv_rows[0]["conflict_ref"] == "demo::q1::d1::avg_scr_math"
    assert csv_rows[0]["text_values"] == "[513.0]"


def test_run_journal_records_terminal_current_dataset_cycle(tmp_path: Path) -> None:
    module = _load_optimizer_ablation_module()
    record = {
        "artifact_id": "current-002-proxy-cache-tablecache-pt0p505-strict",
        "llm_docs": 37,
        "saved_rate": 0.875,
        "answer_recall": 1.0,
        "cell_recall": 1.0,
        "table_recall": 1.0,
        "can_answer": [8, 8],
        "full_recall": True,
        "use_oracle_predicate_proxy": False,
        "use_gt_text_consistency_guard": False,
        "cross_query_extraction_cache": True,
        "table_assignment_calls_before": 400,
        "table_assignment_cache": {
            "cache_hits": 300,
            "cache_misses": 0,
            "source_table_metadata_hits": 100,
            "saved_rate": 1.0,
        },
    }

    journal = module._build_run_journal(
        results=[record],
        ranked=[record],
        run_summary={
            "baseline_llm_docs": 295,
            "oracle_upper_bound_llm_docs": 37,
        },
        best_deployable=record,
        best_table_cache=record,
        dataset_audit={"total_conflicts": 0, "conflicts_by_type": {}},
        output_dir=tmp_path,
    )

    assert any("offline upper bound" in item for item in journal["avoid_repeating"])
    assert any("per-conflict" in item for item in journal["solved_this_cycle"])
    assert any("JSONL and CSV" in item for item in journal["solved_this_cycle"])
    assert not any("Compact sweep comparison" in item for item in journal["solved_this_cycle"])
    assert journal["next_distinct_targets"][0].startswith(
        "Run the same deployable configuration on a larger held-out dataset"
    )
    assert not any("per-conflict links" in item for item in journal["next_distinct_targets"])
    assert not any("CSV/JSONL" in item for item in journal["next_distinct_targets"])

    run_summary = {
        "baseline_llm_docs": 295,
        "oracle_upper_bound_llm_docs": 37,
        "comparison_report_paths": {"markdown": "current_sweep_comparison.md"},
    }
    journal = module._build_run_journal(
        results=[record],
        ranked=[record],
        run_summary=run_summary,
        best_deployable=record,
        best_table_cache=record,
        dataset_audit={"total_conflicts": 0, "conflicts_by_type": {}},
        output_dir=tmp_path,
    )
    assert any("Compact sweep comparison" in item for item in journal["solved_this_cycle"])


def test_source_metadata_ablation_summary_compares_nometa_variant() -> None:
    module = _load_optimizer_ablation_module()
    common = {
        "llm_docs": 37,
        "saved_rate": 0.875,
        "answer_recall": 1.0,
        "cell_recall": 1.0,
        "table_recall": 1.0,
        "can_answer": [8, 8],
        "full_recall": True,
        "table_assignment_cache_enabled": True,
    }
    without_metadata = {
        **common,
        "artifact_id": "current-001-proxy-cache-tablecache-pt0p505-nometa-strict",
        "table_assignment_cache_source_table_metadata": False,
        "table_assignment_cache": {
            "cache_hits": 300,
            "cache_misses": 100,
            "source_table_metadata_hits": 0,
            "source_table_metadata_misses": 0,
            "saved_rate": 0.75,
        },
    }
    with_metadata = {
        **common,
        "artifact_id": "current-002-proxy-cache-tablecache-pt0p505-strict",
        "table_assignment_cache_source_table_metadata": True,
        "table_assignment_cache": {
            "cache_hits": 300,
            "cache_misses": 0,
            "source_table_metadata_hits": 100,
            "source_table_metadata_misses": 0,
            "saved_rate": 1.0,
        },
    }

    summary = module._build_source_metadata_ablation_summary(
        [without_metadata, with_metadata]
    )

    assert summary == {
        "with_metadata_artifact_id": "current-002-proxy-cache-tablecache-pt0p505-strict",
        "without_metadata_artifact_id": "current-001-proxy-cache-tablecache-pt0p505-nometa-strict",
        "with_metadata_calls_after": 0,
        "without_metadata_calls_after": 100,
        "calls_saved_by_metadata": 100,
        "with_metadata_hits": 100,
        "with_metadata_misses": 0,
        "without_metadata_cache_hits": 300,
        "without_metadata_cache_misses": 100,
    }


def test_compact_comparison_rows_include_metadata_ablation() -> None:
    module = _load_optimizer_ablation_module()
    common = {
        "llm_docs": 37,
        "saved_rate": 0.875,
        "answer_recall": 1.0,
        "cell_recall": 1.0,
        "table_recall": 1.0,
        "can_answer": [8, 8],
        "full_recall": True,
        "use_oracle_predicate_proxy": False,
        "use_gt_text_consistency_guard": False,
        "table_assignment_cache_enabled": True,
        "table_assignment_calls_before": 400,
    }
    with_metadata = {
        **common,
        "artifact_id": "with-meta",
        "table_assignment_cache_source_table_metadata": True,
        "table_assignment_cache": {
            "cache_hits": 300,
            "cache_misses": 0,
            "source_table_metadata_hits": 100,
            "source_table_metadata_misses": 0,
            "saved_rate": 1.0,
        },
    }
    without_metadata = {
        **common,
        "artifact_id": "without-meta",
        "table_assignment_cache_source_table_metadata": False,
        "table_assignment_cache": {
            "cache_hits": 300,
            "cache_misses": 100,
            "source_table_metadata_hits": 0,
            "source_table_metadata_misses": 0,
            "saved_rate": 0.75,
        },
    }

    rows = module._build_compact_comparison_rows(
        ranked=[with_metadata, without_metadata],
        best_deployable=with_metadata,
        best_table_cache=with_metadata,
        run_summary={
            "baseline_llm_docs": 295,
            "oracle_upper_bound_llm_docs": 37,
            "source_table_metadata_ablation": {
                "with_metadata_artifact_id": "with-meta",
                "without_metadata_artifact_id": "without-meta",
            },
        },
    )

    rows_by_label = {row["label"]: row for row in rows}
    assert rows_by_label["baseline"]["llm_docs"] == 295
    assert rows_by_label["oracle_upper_bound"]["oracle_gap"] == 0
    assert rows_by_label["source_metadata_on"]["table_assignment_calls_after"] == 0
    assert rows_by_label["source_metadata_off"]["table_assignment_calls_after"] == 100
    assert rows_by_label["source_metadata_on"]["saved_vs_baseline"] == 258


def test_offline_upper_bound_ignores_non_full_recall_records() -> None:
    module = _load_optimizer_ablation_module()
    lower_but_bad = {
        "artifact_id": "bad-upper",
        "llm_docs": 298,
        "full_recall": False,
        "use_audit_conflict_quarantine": True,
    }
    full_recall = {
        "artifact_id": "good-upper",
        "llm_docs": 389,
        "full_recall": True,
        "use_audit_conflict_quarantine": True,
    }

    assert module._best_offline_upper_record([lower_but_bad, full_recall]) is full_recall
