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
