from __future__ import annotations

from pathlib import Path
from typing import Any

from redd.core.data_loader import DataLoaderBase
from redd.exp.evaluation import EvalDataExtraction


class QueryAwareLoader(DataLoaderBase):
    def __init__(self) -> None:
        super().__init__(Path.cwd())
        self._doc_ids = ["course-1", "teach-1", "extra-1", "missing-1"]
        self._gt = {
            "course-1": {
                "doc": "",
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
                "Attributes": [{"Attribute Name": "title"}, {"Attribute Name": "credits"}],
            },
            {
                "Schema Name": "teaches",
                "Attributes": [
                    {"Attribute Name": "course_title"},
                    {"Attribute Name": "semester"},
                    {"Attribute Name": "year"},
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
    ) -> dict[str, Any]:
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
