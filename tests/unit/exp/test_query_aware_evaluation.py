from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from redd.core.data_loader import DataLoaderBase
from redd.core.utils.constants import RESULT_DATA_KEY, RESULT_RECORDS_KEY, RESULT_TABLE_KEY
from redd.exp.evaluation import EvalDataExtraction


class QueryAwareLoader(DataLoaderBase):
    def __init__(self) -> None:
        super().__init__(Path.cwd())
        self._doc_ids = ["course-1", "teach-1", "extra-1", "missing-1"]
        self._gt = {
            "course-1": {
                "doc": "Course catalog note: DB Systems is the short title used for Databases. Credits: 4.",
                "table": "course",
                "data": {"title": "Databases", "credits": "4", "dept_name": "CS"},
                "data_records": [
                    {"table_name": "course", "data": {"title": "Databases", "credits": "4", "dept_name": "CS"}}
                ],
            },
            "teach-1": {
                "doc": "",
                "table": "teaches",
                "data": {"course_title": "Databases", "semester": "Fall", "year": "2026"},
                "data_records": [
                    {
                        "table_name": "teaches",
                        "data": {"course_title": "Databases", "semester": "Fall", "year": "2026"},
                    }
                ],
            },
            "extra-1": {
                "doc": "",
                "table": "instructor",
                "data": {"name": "Ada"},
                "data_records": [{"table_name": "instructor", "data": {"name": "Ada"}}],
            },
            "missing-1": {
                "doc": "",
                "table": "course",
                "data": {"title": "Compilers", "credits": "4"},
                "data_records": [{"table_name": "course", "data": {"title": "Compilers", "credits": "4"}}],
            },
        }

    @property
    def num_docs(self) -> int:
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    def iter_docs(self):
        for doc_id in self._doc_ids:
            yield self.get_doc(doc_id)

    def get_doc(self, doc_id: str):
        return "", doc_id, {}

    def get_doc_info(self, doc_id: str) -> dict[str, Any] | None:
        return self._gt.get(doc_id)

    def load_schema_query(self, qid: str | int):
        return [
            {
                "Schema Name": "course",
                "Description": "University course catalog records.",
                "Attributes": [
                    {
                        "Attribute Name": "title",
                        "Description": "Canonical course title.",
                        "type": "string",
                        "column_id": "course.title",
                    },
                    {
                        "Attribute Name": "credits",
                        "Description": "Number of credit hours.",
                        "type": "number",
                        "column_id": "course.credits",
                    },
                ],
            },
            {
                "Schema Name": "teaches",
                "Description": "Course offering records.",
                "Attributes": [
                    {"Attribute Name": "course_title", "Description": "Title of the taught course."},
                    {"Attribute Name": "semester", "Description": "Academic semester."},
                    {"Attribute Name": "year", "Description": "Academic year."},
                ],
            },
        ]


class SemanticQueryAwareEvaluator(EvalDataExtraction):
    def _semantic_eval_enabled(self) -> bool:
        return True

    def _semantic_match_attribute(
        self,
        *,
        pred_attr: str,
        pred_value: Any,
        gt_attr: str,
        gt_value: Any,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.seen_contexts = getattr(self, "seen_contexts", [])
        self.seen_contexts.append(context)
        if self._compare_attribute_values(pred_value, gt_value):
            return {
                "result": True,
                "method": "strict",
                "reasoning": "strict",
                "cached": False,
            }
        return {
            "result": str(pred_value) == "DB Systems" and str(gt_value) == "Databases",
            "method": "llm",
            "reasoning": "fake semantic judge",
            "cached": False,
        }


def test_query_aware_recall_uses_answer_row_provenance_for_cell_denominator() -> None:
    loader = QueryAwareLoader()
    evaluator = EvalDataExtraction({"res_param_str": "unit", "training_data_count": 0})
    query_info = {
        "query": "Which Fall 2026 four-credit courses were taught?",
        "sql": (
            "SELECT T1.title FROM course AS T1 JOIN teaches AS T2 "
            "ON T1.title = T2.course_title "
            "WHERE T2.semester = 'Fall' AND T2.year = '2026' AND T1.credits = '4';"
        ),
        "tables": ["course", "teaches"],
        "attributes": [
            "course.title",
            "course.credits",
            "teaches.course_title",
            "teaches.semester",
            "teaches.year",
        ],
        "output_columns": ["course.title"],
    }
    result = {
        "course-1": {"res": "course", "data": {"title": "Databases", "credits": "4", "dept_name": "CS"}},
        "teach-1": {
            "res": "teaches",
            "data": {"course_title": "Databases", "semester": "Fall", "year": "2026"},
        },
        "extra-1": {"res": "instructor", "data": {"name": "Ada"}},
        "missing-1": {"res": "course", "data": {"title": "Compilers"}},
    }

    stats = evaluator.compute_query_aware_statistics(loader, result, "q1", query_info).to_dict()

    assert stats["table_assignment"]["recall"] == 1.0
    assert stats["cell_recall"]["covered"] == 5
    assert stats["cell_recall"]["total"] == 5
    assert stats["cell_recall"]["missing"] == 0
    assert stats["summary"]["redundant_cells"] == 3
    assert stats["answer_recall"]["recall"] == 1.0
    assert stats["summary"]["can_answer_query"] is True
    layers = stats["required_cell_layers"]["layers"]
    assert layers["answer"]["total"] == 1
    assert layers["predicate"]["total"] == 3
    assert layers["join"]["total"] == 1
    assert layers["other_required"]["total"] == 0


def test_query_aware_metadata_only_query_treats_nil_as_null() -> None:
    loader = QueryAwareLoader()
    evaluator = EvalDataExtraction({"res_param_str": "unit", "training_data_count": 0})
    query_info = {
        "query": "",
        "sql": "",
        "tables": ["course"],
        "attributes": ["course.title", "course.credits", "course.dept_name"],
    }
    result = {
        "course-1": {
            "res": "course",
            "data": {"title": "Databases", "credits": "4", "dept_name": "None"},
        },
        "missing-1": {"res": "course", "data": {"title": "Compilers", "credits": "4"}},
    }
    loader._gt["course-1"]["data"]["dept_name"] = "nil"
    loader._gt["course-1"]["data_records"][0]["data"]["dept_name"] = "nil"

    stats = evaluator.compute_query_aware_statistics(loader, result, "q1", query_info).to_dict()

    assert stats["cell_recall"]["covered"] == 4
    assert stats["cell_recall"]["total"] == 4
    assert stats["cell_recall"]["null_gt_skipped"] == 2
    assert stats["answer_recall"]["reason"] == "query_has_no_sql"
    assert stats["summary"]["can_answer_query"] is True


def test_query_aware_semantic_cell_accuracy_uses_required_attrvalue_cells() -> None:
    loader = QueryAwareLoader()
    evaluator = SemanticQueryAwareEvaluator({"res_param_str": "unit", "training_data_count": 0})
    query_info = {
        "query": "",
        "sql": "",
        "tables": ["course"],
        "attributes": ["course.title", "course.credits"],
    }
    result = {
        "course-1": {"res": "course", "data": {"title": "DB Systems", "credits": "4"}},
        "missing-1": {"res": "course", "data": {"title": "Compilers", "credits": "4"}},
    }

    stats = evaluator.compute_query_aware_statistics(loader, result, "q1", query_info).to_dict()

    assert stats["cell_recall"]["covered"] == 3
    assert stats["cell_recall"]["total"] == 4
    assert stats["semantic_cell_accuracy"]["scope"] == "query_required_answer_cells"
    assert stats["semantic_cell_accuracy"]["correct"] == 4
    assert stats["semantic_cell_accuracy"]["total"] == 4
    assert stats["semantic_cell_accuracy"]["llm_judged"] == 1
    assert stats["semantic_cell_accuracy"]["accuracy"] == 1.0
    assert {cell["attr"] for cell in stats["semantic_cell_accuracy"]["cells"]} == {"title", "credits"}


def test_query_aware_semantic_cell_accuracy_passes_context() -> None:
    loader = QueryAwareLoader()
    evaluator = SemanticQueryAwareEvaluator(
        {
            "res_param_str": "unit",
            "training_data_count": 0,
            "eval": {
                "semantic_context": {
                    "enabled": True,
                    "include_doc_text": "on_mismatch",
                    "doc_text_max_chars": 120,
                }
            },
        },
        api_key="test-key",
    )
    query_info = {
        "question": "Which four-credit courses match the requested title?",
        "sql": "SELECT title FROM course WHERE credits = '4';",
        "tables": ["course"],
        "attributes": ["course.title", "course.credits"],
        "output_columns": ["course.title"],
    }
    result = {
        "course-1": {"res": "course", "data": {"title": "DB Systems", "credits": "4"}},
        "missing-1": {"res": "course", "data": {"title": "Compilers", "credits": "4"}},
    }

    stats = evaluator.compute_query_aware_statistics(loader, result, "q1", query_info).to_dict()

    contexts = [context for context in evaluator.seen_contexts if context]
    assert len(contexts) == 1
    context = contexts[0]
    assert context["Cell Role"] == "answer"
    assert context["Schema"]["Table Name"] == "course"
    assert context["Schema"]["Attribute"]["Attribute Description"] == "Canonical course title."
    assert context["Query"]["Question"] == "Which four-credit courses match the requested title?"
    assert context["Document"]["Doc ID"] == "course-1"
    assert "Databases" in context["Document"]["Text Excerpt"]
    assert stats["semantic_cell_accuracy"]["context"]["include_doc_text"] == "on_mismatch"


def test_query_aware_recall_reads_multi_record_prediction_entries() -> None:
    loader = QueryAwareLoader()
    loader._doc_ids = ["bundle-1"]
    loader._gt = {
        "bundle-1": {
            "doc": "",
            "table": "course",
            "data": {"title": "Algorithms", "credits": "4"},
            "data_records": [
                {"table_name": "course", "data": {"title": "Algorithms", "credits": "4"}},
                {"table_name": "course", "data": {"title": "Databases", "credits": "3"}},
            ],
        }
    }
    evaluator = EvalDataExtraction({"res_param_str": "unit", "training_data_count": 0})
    query_info = {
        "query": "",
        "sql": "",
        "tables": ["course"],
        "attributes": ["course.title", "course.credits"],
    }
    result = {
        "bundle-1": {
            RESULT_TABLE_KEY: "course",
            RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
            RESULT_RECORDS_KEY: [
                {
                    "table": "course",
                    "record_id": "course-a",
                    "data": {"title": "Algorithms", "credits": "4"},
                },
                {
                    "table": "course",
                    "record_id": "course-b",
                    "data": {"title": "Databases", "credits": "3"},
                },
            ],
        }
    }

    stats = evaluator.compute_query_aware_statistics(loader, result, "q1", query_info).to_dict()

    assert stats["table_assignment"]["covered"] == 2
    assert stats["table_assignment"]["total"] == 2
    assert stats["cell_recall"]["covered"] == 4
    assert stats["cell_recall"]["total"] == 4
    assert stats["summary"]["can_answer_query"] is True


def test_query_optimization_summary_reads_usage_and_stage_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("REDD_LLM_USAGE_LOG", raising=False)
    out_root = tmp_path
    usage_path = out_root / "llm_usage.jsonl"
    usage_rows = [
        {
            "provider": "deepseek",
            "configured_model": "deepseek-chat",
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
            "context": {"stage": "table_assignment", "query_id": "q1", "doc_id": "d1"},
        },
        {
            "provider": "deepseek",
            "configured_model": "deepseek-chat",
            "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
            "context": {"stage": "proxy_runtime_oracle", "query_id": "q1", "doc_id": "d1"},
        },
        {
            "provider": "deepseek",
            "configured_model": "deepseek-chat",
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            "context": {"stage": "semantic_evaluation", "query_id": "q1"},
        },
        {
            "provider": "deepseek",
            "configured_model": "deepseek-chat",
            "usage": {"prompt_tokens": 999, "completion_tokens": 1, "total_tokens": 1000},
            "context": {"stage": "table_assignment", "query_id": "q2"},
        },
    ]
    usage_path.write_text("\n".join(json.dumps(row) for row in usage_rows), encoding="utf-8")

    doc_filter_dir = out_root / "doc_filter"
    doc_filter_dir.mkdir()
    (doc_filter_dir / "doc_filter_q1.json").write_text(
        json.dumps(
            {
                "query_id": "q1",
                "excluded_doc_ids": ["d3"],
                "kept_doc_ids": ["d1", "d2"],
                "metadata": {
                    "num_docs_input": 3,
                    "num_docs_kept": 2,
                    "num_docs_excluded": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (out_root / "table_assignment_cache.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "query_id": "q1",
                        "input_docs": 3,
                        "excluded": 0,
                        "cache_hits": 2,
                        "cache_misses": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (out_root / "res_tabular_data_q1_demo_proxy_decisions.json").write_text(
        json.dumps(
            {
                "course": {
                    "all_doc_ids": ["d1", "d2", "d3"],
                    "passed_doc_ids": ["d1"],
                    "extracted_doc_ids": ["d1"],
                    "proxy_stats": {"p": {"evaluated": 3, "passed": 1, "rejected": 2}},
                }
            }
        ),
        encoding="utf-8",
    )

    evaluator = EvalDataExtraction({"res_param_str": "demo", "training_data_count": 0})
    evaluator.out_root = out_root

    summary = evaluator._collect_query_optimization_summary("q1")

    assert summary["has_metrics"] is True
    assert summary["llm_usage"]["calls"] == 3
    assert summary["llm_usage"]["total_tokens"] == 170
    assert summary["llm_usage"]["by_stage"]["table_assignment"]["calls"] == 1
    assert summary["doc_call_optimization"]["calls_before"] == 9
    assert summary["doc_call_optimization"]["calls_after"] == 4
    assert summary["doc_call_optimization"]["calls_saved"] == 5
    assert summary["doc_call_optimization"]["by_stage"]["doc_filter"]["calls_saved"] == 1
    assert summary["doc_call_optimization"]["by_stage"]["table_assignment_cache"]["calls_saved"] == 2
    assert summary["doc_call_optimization"]["by_stage"]["proxy_runtime"]["calls_saved"] == 2
    assert summary["token_optimization"]["status"] == "estimated"
    assert summary["token_optimization"]["estimated_tokens_saved"] > 0
    enabled = {item["id"]: item for item in summary["enabled_optimizations"]}
    assert set(enabled) == {"doc_filter", "table_assignment_cache", "proxy_runtime"}
    assert enabled["doc_filter"]["optimized_part"].startswith("Phase 0")
    assert enabled["table_assignment_cache"]["optimized_part"] == "Phase 1: table assignment"
    assert enabled["proxy_runtime"]["optimized_part"].startswith("Phase 2")
    assert enabled["proxy_runtime"]["doc_call_savings"]["calls_saved"] == 2
    assert enabled["proxy_runtime"]["token_savings"]["estimated_tokens_saved"] > 0
    assert "summary" in enabled["doc_filter"]


def test_query_aware_sql_provenance_filters_multi_record_gt_rows() -> None:
    loader = QueryAwareLoader()
    loader._doc_ids = ["bundle-1"]
    loader._gt = {
        "bundle-1": {
            "doc": "",
            "table": "course",
            "data": {"title": "Algorithms", "credits": "4"},
            "data_records": [
                {
                    "table_name": "course",
                    "record_id": "course-a",
                    "row_id": "course-a",
                    "data": {"title": "Algorithms", "credits": "4"},
                },
                {
                    "table_name": "course",
                    "record_id": "course-b",
                    "row_id": "course-b",
                    "data": {"title": "Databases", "credits": "3"},
                },
            ],
        }
    }
    evaluator = EvalDataExtraction({"res_param_str": "unit", "training_data_count": 0})
    query_info = {
        "query": "",
        "sql": "SELECT title FROM course WHERE credits = '4';",
        "tables": ["course"],
        "attributes": ["course.title", "course.credits"],
        "output_columns": ["course.title"],
    }
    result = {
        "bundle-1": {
            RESULT_TABLE_KEY: "course",
            RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
            RESULT_RECORDS_KEY: [
                {
                    "table": "course",
                    "record_id": "course-a",
                    "data": {"title": "Algorithms", "credits": "4"},
                },
                {
                    "table": "course",
                    "record_id": "course-b",
                    "data": {"title": "Databases", "credits": "3"},
                },
            ],
        }
    }

    stats = evaluator.compute_query_aware_statistics(loader, result, "q1", query_info).to_dict()

    assert stats["table_assignment"]["covered"] == 1
    assert stats["table_assignment"]["total"] == 1
    assert stats["cell_recall"]["covered"] == 2
    assert stats["cell_recall"]["total"] == 2
    assert stats["answer_recall"]["recall"] == 1.0
    assert stats["summary"]["can_answer_query"] is True


def test_full_table_semantic_can_be_disabled_independently() -> None:
    default_evaluator = EvalDataExtraction({"res_param_str": "unit", "training_data_count": 0})
    disabled_evaluator = EvalDataExtraction(
        {
            "res_param_str": "unit",
            "training_data_count": 0,
            "eval": {"full_table_semantic": False},
        },
        api_key="test-key",
    )

    assert default_evaluator.full_table_semantic_enabled is True
    assert disabled_evaluator.full_table_semantic_enabled is False
