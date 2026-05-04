import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from redd.core.data_population.data_extraction import DataExtraction
from redd.core.utils.constants import (
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTES_KEY,
    NULL_VALUE,
    RESULT_DATA_KEY,
    RESULT_TABLE_KEY,
    SCHEMA_NAME_KEY,
)


class FakeManifestLoader:
    doc_ids = ["course-4", "course-3", "instructor-1"]

    def load_query_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "q1": {
                "sql": "SELECT title FROM course WHERE credits = '4';",
                "required_tables": ["course"],
                "required_columns": ["course.title", "course.credits"],
                "output_columns": ["course.title"],
            }
        }

    def load_schema_general(self) -> List[Dict[str, Any]]:
        return [
            {
                SCHEMA_NAME_KEY: "course",
                ATTRIBUTES_KEY: [
                    {ATTRIBUTE_NAME_KEY: "title"},
                    {ATTRIBUTE_NAME_KEY: "credits"},
                ],
            },
            {
                SCHEMA_NAME_KEY: "instructor",
                ATTRIBUTES_KEY: [{ATTRIBUTE_NAME_KEY: "name"}],
            },
        ]

    def load_schema_query(self, query_id: str) -> List[Dict[str, Any]]:
        assert query_id == "q1"
        return [
            {
                SCHEMA_NAME_KEY: "course",
                ATTRIBUTES_KEY: [
                    {ATTRIBUTE_NAME_KEY: "title"},
                    {ATTRIBUTE_NAME_KEY: "credits"},
                ],
            }
        ]

    def load_name_map(self, query_id: str) -> Dict[str, Dict[str, Any]]:
        assert query_id == "q1"
        return {"table": {}, "attribute": {}}

    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        records = {
            "course-4": [
                {
                    "table_name": "course",
                    "data": {"title": "Algorithms", "credits": "4"},
                }
            ],
            "course-3": [
                {
                    "table_name": "course",
                    "data": {"title": "Databases", "credits": "3"},
                }
            ],
            "instructor-1": [
                {
                    "table_name": "instructor",
                    "data": {"name": "Ada"},
                }
            ],
        }
        return {"data_records": records[doc_id]}


def test_data_extraction_persists_all_docs_and_required_attrs(tmp_path: Path) -> None:
    extractor = DataExtraction(
        {
            "mode": "ground_truth",
            "res_param_str": "unit",
            "exp_query_id_list": ["q1"],
            "training_data_count": 0,
        }
    )

    with (
        patch(
            "redd.core.data_population.data_extraction.create_data_loader",
            return_value=FakeManifestLoader(),
        ),
        patch.object(
            DataExtraction,
            "_materialize_query_output",
            side_effect=AssertionError("data extraction must not materialize query output"),
        ),
    ):
        extractor._process_dataset(tmp_path / "dataset", tmp_path / "out")

    [result_path] = list((tmp_path / "out").glob("res_tabular_data_q1_unit.json"))
    result = json.loads(result_path.read_text())

    assert set(result) == {"course-4", "course-3", "instructor-1"}
    assert result["course-4"] == {
        RESULT_TABLE_KEY: "course",
        RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
    }
    assert result["course-3"] == {
        RESULT_TABLE_KEY: "course",
        RESULT_DATA_KEY: {"title": "Databases", "credits": "3"},
    }
    assert result["instructor-1"] == {
        RESULT_TABLE_KEY: NULL_VALUE,
        RESULT_DATA_KEY: {},
    }


def test_failed_table_assignment_records_null_doc(tmp_path: Path) -> None:
    class UnknownTablePrompt:
        def complete_model(self, *_args: Any, **_kwargs: Any) -> Any:
            return type("Result", (), {"table_assignment": "instructor"})()

    class TextLoader:
        doc_ids = ["doc-1"]

        def get_doc_text(self, doc_id: str) -> str:
            assert doc_id == "doc-1"
            return "Instructor profile"

    extractor = DataExtraction.__new__(DataExtraction)
    extractor.config = {}
    extractor.disable_llm = False
    extractor.loader = TextLoader()
    extractor.prompt_table = UnknownTablePrompt()
    extractor.retry_params = {}
    extractor.schema_general = [
        {SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []},
        {SCHEMA_NAME_KEY: "instructor", ATTRIBUTES_KEY: []},
    ]
    extractor.pause_controller = None

    res_data: Dict[str, Any] = {}
    extractor._process_table_assignment(
        schema_query=[{SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}],
        res_data=res_data,
        res_path=tmp_path / "res.json",
        pgbar_name="unit",
        max_table_retries=0,
        target_doc_ids=["doc-1"],
    )

    assert res_data == {"doc-1": {RESULT_TABLE_KEY: NULL_VALUE, RESULT_DATA_KEY: {}}}


def test_data_extraction_reuses_upstream_schema_doc_filter(tmp_path: Path) -> None:
    upstream = tmp_path / "schema"
    (upstream / "doc_filter").mkdir(parents=True)
    (upstream / "doc_filter" / "doc_filter_q1_schema_recall095.json").write_text(
        json.dumps(
            {
                "query_id": "q1",
                "excluded_doc_ids": ["d2"],
                "kept_doc_ids": ["d1"],
                "metadata": {
                    "target_recall": 0.95,
                    "num_docs_input": 2,
                    "num_docs_excluded": 1,
                    "num_docs_kept": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    extractor = DataExtraction(
        {
            "mode": "ground_truth",
            "res_param_str": "unit",
            "training_data_count": 0,
            "upstream_doc_filter_root": str(upstream),
            "doc_filter": {"enabled": True},
        }
    )
    extractor.test_doc_ids = ["d1", "d2"]
    extractor.train_doc_ids = []
    extractor.out_root = tmp_path / "data"
    extractor.out_root.mkdir()

    excluded = extractor._excluded_doc_ids_for_query(
        query_id="q1",
        schema_query=[],
        target_recall_override=None,
    )

    assert excluded == {"d2"}
    [reuse_artifact] = list((tmp_path / "data" / "doc_filter").glob("*.json"))
    payload = json.loads(reuse_artifact.read_text(encoding="utf-8"))
    assert payload["metadata"]["reused_from_stage"] == "schema_refinement"


def test_table_assignment_uses_query_schema_and_accepts_null() -> None:
    class NullPrompt:
        last_payload: Dict[str, Any] = {}

        def complete_model(self, payload: str, *_args: Any, **_kwargs: Any) -> Any:
            self.last_payload = json.loads(payload)
            return type("Result", (), {"table_assignment": None})()

    prompt = NullPrompt()
    extractor = DataExtraction.__new__(DataExtraction)
    extractor.prompt_table = prompt
    extractor.retry_params = {}
    extractor.schema_general = [
        {SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []},
        {SCHEMA_NAME_KEY: "instructor", ATTRIBUTES_KEY: []},
    ]

    table, failed, reason = extractor._assign_table_single_doc(
        doc_id="doc-1",
        doc_text="Instructor salary profile",
        all_tables=["course"],
        prompt_schema=[{SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}],
        max_retries=0,
    )

    assert table is None
    assert failed is False
    assert reason is None
    assert prompt.last_payload["Schema"] == [
        {SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}
    ]


def test_table_assignment_general_schema_cache_reuses_out_of_query_tables(tmp_path: Path) -> None:
    class CountingPrompt:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def complete_model(self, payload: str, *_args: Any, **_kwargs: Any) -> Any:
            parsed = json.loads(payload)
            self.calls.append(parsed)
            document = parsed["Document"]
            table = "instructor" if "Instructor" in document else "course"
            return type("Result", (), {"table_assignment": table})()

    class TextLoader:
        doc_ids = ["course-doc", "instructor-doc"]

        def get_doc_text(self, doc_id: str) -> str:
            return {
                "course-doc": "Course catalog entry",
                "instructor-doc": "Instructor profile",
            }[doc_id]

    prompt = CountingPrompt()
    extractor = DataExtraction.__new__(DataExtraction)
    extractor.config = {}
    extractor.disable_llm = False
    extractor.loader = TextLoader()
    extractor.prompt_table = prompt
    extractor.retry_params = {}
    extractor.schema_general = [
        {SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []},
        {SCHEMA_NAME_KEY: "instructor", ATTRIBUTES_KEY: []},
    ]
    extractor.pause_controller = None
    extractor.table_assignment_cache_enabled = True
    extractor.table_assignment_cache_general_schema = True
    extractor._table_assignment_cache = {}
    extractor._table_assignment_null_cache = set()
    extractor._table_assignment_cache_events = []
    extractor.data_path = Path("dataset")
    extractor.out_root = tmp_path

    course_results: Dict[str, Any] = {}
    extractor._process_table_assignment(
        schema_query=[{SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}],
        res_data=course_results,
        res_path=tmp_path / "course.json",
        pgbar_name="course",
        target_doc_ids=["course-doc", "instructor-doc"],
        query_id="q-course",
    )

    assert len(prompt.calls) == 2
    assert course_results["course-doc"][RESULT_TABLE_KEY] == "course"
    assert course_results["instructor-doc"][RESULT_TABLE_KEY] == NULL_VALUE
    assert extractor._table_assignment_cache == {
        "course-doc": "course",
        "instructor-doc": "instructor",
    }

    instructor_results: Dict[str, Any] = {}
    extractor._process_table_assignment(
        schema_query=[{SCHEMA_NAME_KEY: "instructor", ATTRIBUTES_KEY: []}],
        res_data=instructor_results,
        res_path=tmp_path / "instructor.json",
        pgbar_name="instructor",
        target_doc_ids=["course-doc", "instructor-doc"],
        query_id="q-instructor",
    )

    assert len(prompt.calls) == 2
    assert instructor_results["course-doc"][RESULT_TABLE_KEY] == NULL_VALUE
    assert instructor_results["instructor-doc"][RESULT_TABLE_KEY] == "instructor"

    cache_payload = json.loads((tmp_path / "table_assignment_cache.json").read_text())
    assert cache_payload["totals"] == {
        "input_docs": 4,
        "cache_hits": 2,
        "cache_misses": 2,
        "source_table_metadata_hits": 0,
        "source_table_metadata_misses": 0,
        "excluded": 0,
    }


def test_table_assignment_uses_source_table_metadata_shortcut(tmp_path: Path) -> None:
    class FailingPrompt:
        def complete_model(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("source_table metadata should avoid prompt calls")

    class MetadataLoader:
        doc_ids = ["course-doc", "instructor-doc"]

        def get_doc(self, doc_id: str) -> tuple[str, str, Dict[str, Any]]:
            table = "course" if doc_id == "course-doc" else "instructor"
            return "Document", doc_id, {"table_name": table}

    extractor = DataExtraction.__new__(DataExtraction)
    extractor.config = {}
    extractor.disable_llm = False
    extractor.loader = MetadataLoader()
    extractor.prompt_table = FailingPrompt()
    extractor.retry_params = {}
    extractor.schema_general = [
        {SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []},
        {SCHEMA_NAME_KEY: "instructor", ATTRIBUTES_KEY: []},
    ]
    extractor.pause_controller = None
    extractor.table_assignment_cache_enabled = True
    extractor.table_assignment_cache_general_schema = True
    extractor.table_assignment_source_table_metadata = True
    extractor._table_assignment_cache = {}
    extractor._table_assignment_null_cache = set()
    extractor._table_assignment_cache_events = []
    extractor.data_path = Path("dataset")
    extractor.out_root = tmp_path

    results: Dict[str, Any] = {}
    extractor._process_table_assignment(
        schema_query=[{SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}],
        res_data=results,
        res_path=tmp_path / "course.json",
        pgbar_name="course",
        target_doc_ids=["course-doc", "instructor-doc"],
        query_id="q-course",
    )

    assert results["course-doc"][RESULT_TABLE_KEY] == "course"
    assert results["instructor-doc"][RESULT_TABLE_KEY] == NULL_VALUE
    cache_payload = json.loads((tmp_path / "table_assignment_cache.json").read_text())
    assert cache_payload["totals"] == {
        "input_docs": 2,
        "cache_hits": 0,
        "cache_misses": 0,
        "source_table_metadata_hits": 2,
        "source_table_metadata_misses": 0,
        "excluded": 0,
    }


def test_table_assignment_records_source_table_metadata_misses(tmp_path: Path) -> None:
    class CountingPrompt:
        def __init__(self) -> None:
            self.calls = 0

        def complete_model(self, payload: str, *_args: Any, **_kwargs: Any) -> Any:
            self.calls += 1
            parsed = json.loads(payload)
            table = "course" if "course" in parsed["Document"].lower() else "instructor"
            return type("Result", (), {"table_assignment": table})()

    class PartialMetadataLoader:
        doc_ids = ["course-doc", "instructor-doc"]

        def get_doc(self, doc_id: str) -> tuple[str, str, Dict[str, Any]]:
            if doc_id == "course-doc":
                return "Course catalog entry", doc_id, {"table_name": "course"}
            return "Instructor profile", doc_id, {}

        def get_doc_text(self, doc_id: str) -> str:
            return self.get_doc(doc_id)[0]

    prompt = CountingPrompt()
    extractor = DataExtraction.__new__(DataExtraction)
    extractor.config = {}
    extractor.disable_llm = False
    extractor.loader = PartialMetadataLoader()
    extractor.prompt_table = prompt
    extractor.retry_params = {}
    extractor.schema_general = [
        {SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []},
        {SCHEMA_NAME_KEY: "instructor", ATTRIBUTES_KEY: []},
    ]
    extractor.pause_controller = None
    extractor.table_assignment_cache_enabled = True
    extractor.table_assignment_cache_general_schema = True
    extractor.table_assignment_source_table_metadata = True
    extractor._table_assignment_cache = {}
    extractor._table_assignment_null_cache = set()
    extractor._table_assignment_cache_events = []
    extractor.data_path = Path("dataset")
    extractor.out_root = tmp_path

    results: Dict[str, Any] = {}
    extractor._process_table_assignment(
        schema_query=[{SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}],
        res_data=results,
        res_path=tmp_path / "course.json",
        pgbar_name="course",
        target_doc_ids=["course-doc", "instructor-doc"],
        query_id="q-course",
    )

    assert prompt.calls == 1
    assert results["course-doc"][RESULT_TABLE_KEY] == "course"
    assert results["instructor-doc"][RESULT_TABLE_KEY] == NULL_VALUE
    cache_payload = json.loads((tmp_path / "table_assignment_cache.json").read_text())
    assert cache_payload["totals"] == {
        "input_docs": 2,
        "cache_hits": 0,
        "cache_misses": 1,
        "source_table_metadata_hits": 1,
        "source_table_metadata_misses": 1,
        "excluded": 0,
    }


def test_table_assignment_writes_empty_result_when_all_docs_excluded(tmp_path: Path) -> None:
    class TextLoader:
        doc_ids = ["doc-1"]

        def get_doc_text(self, doc_id: str) -> str:
            raise AssertionError(f"excluded doc should not be loaded: {doc_id}")

    extractor = DataExtraction.__new__(DataExtraction)
    extractor.config = {}
    extractor.disable_llm = False
    extractor.loader = TextLoader()
    extractor.retry_params = {}
    extractor.schema_general = [{SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}]
    extractor.pause_controller = None

    res_path = tmp_path / "res.json"
    res_data: Dict[str, Any] = {}
    extractor._process_table_assignment(
        schema_query=[{SCHEMA_NAME_KEY: "course", ATTRIBUTES_KEY: []}],
        res_data=res_data,
        res_path=res_path,
        pgbar_name="unit",
        excluded_doc_ids={"doc-1"},
        target_doc_ids=["doc-1"],
    )

    assert res_path.exists()
    assert json.loads(res_path.read_text()) == {}
