import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from redd.core.data_extraction.data_extraction import DataExtraction
from redd.core.utils.constants import (
    ATTRIBUTE_DESC_KEY,
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTES_KEY,
    NULL_VALUE,
    PATH_TEMPLATES,
    RESULT_DATA_KEY,
    RESULT_RECORDS_KEY,
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


class FakeLLMManifestLoader:
    doc_ids = ["course-4", "course-3", "instructor-1"]

    documents = {
        "course-4": "Course title: Algorithms. Credits: 4.",
        "course-3": "Course title: Databases. Credits: 3.",
        "instructor-1": "Instructor name: Ada.",
    }

    def load_query_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "q1": {
                "query": "List course titles.",
                "sql": "",
                "required_tables": ["course"],
                "required_columns": ["course.title"],
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
                ATTRIBUTES_KEY: [{ATTRIBUTE_NAME_KEY: "title"}],
            }
        ]

    def get_doc_text(self, doc_id: str) -> str:
        return self.documents[doc_id]


class FortunePostprocessManifestLoader:
    doc_ids = ["company-1", "company-2"]

    documents = {
        "company-1": (
            "Acme is a public company. Its growth in jobs reflects workforce "
            "expansion and job creation. It is one of the World's Most Admired "
            "Companies. The updated figures show a market value of $12.3 billion "
            "as of June 4, 2024. It is listed under the ticker symbol ACME."
        ),
        "company-2": (
            "Beta is a public company. It is not one of the World's Most Admired "
            "Companies and does not show job growth. Its market value was not "
            "updated in the document."
        ),
    }

    def load_query_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "q1": {
                "query": "List company names.",
                "sql": "",
                "required_tables": ["fortune500_companies"],
                "required_columns": ["fortune500_companies.company"],
                "output_columns": ["fortune500_companies.company"],
            }
        }

    def load_schema_general(self) -> List[Dict[str, Any]]:
        return [
            {
                SCHEMA_NAME_KEY: "fortune500_companies",
                ATTRIBUTES_KEY: [
                    {ATTRIBUTE_NAME_KEY: "company"},
                    {
                        ATTRIBUTE_NAME_KEY: "growth_in_jobs",
                        ATTRIBUTE_DESC_KEY: (
                            "Closed-world boolean indicating whether the company "
                            "shows growth in jobs or workforce expansion."
                        ),
                    },
                    {
                        ATTRIBUTE_NAME_KEY: "worlds_most_admired_companies",
                        ATTRIBUTE_DESC_KEY: (
                            "Closed-world boolean for World's Most Admired "
                            "Companies membership."
                        ),
                    },
                    {ATTRIBUTE_NAME_KEY: "market_cap_updated_m"},
                    {ATTRIBUTE_NAME_KEY: "ticker"},
                ],
            }
        ]

    def load_schema_query(self, query_id: str) -> List[Dict[str, Any]]:
        assert query_id == "q1"
        return [
            {
                SCHEMA_NAME_KEY: "fortune500_companies",
                ATTRIBUTES_KEY: [{ATTRIBUTE_NAME_KEY: "company"}],
            }
        ]

    def get_doc_text(self, doc_id: str) -> str:
        return self.documents[doc_id]


class MultiRecordGTManifestLoader(FakeManifestLoader):
    doc_ids = ["bundle-1"]

    def get_doc_info(self, doc_id: str) -> Dict[str, Any]:
        assert doc_id == "bundle-1"
        return {
            "data_records": [
                {
                    "table_name": "course",
                    "data": {"title": "Algorithms", "credits": "4"},
                },
                {
                    "table_name": "course",
                    "data": {"title": "Databases", "credits": "3"},
                },
            ]
        }


class FourDocLLMManifestLoader(FakeLLMManifestLoader):
    doc_ids = ["course-4", "course-3", "instructor-1", "course-5"]
    documents = {
        **FakeLLMManifestLoader.documents,
        "course-5": "Course title: Systems. Credits: 5.",
    }


class CountingTablePrompt:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def complete_model(self, payload: str, *_args: Any, **_kwargs: Any) -> Any:
        document = json.loads(payload)["Document"]
        self.calls.append(document)
        table = "instructor" if "Instructor" in document else "course"
        return type("Result", (), {"table_assignment": table})()


class CountingAttrPrompt:
    def __init__(self) -> None:
        self.calls: List[tuple[str, str]] = []

    def complete_model(self, payload: str, *_args: Any, **_kwargs: Any) -> Any:
        parsed = json.loads(payload)
        document = parsed["Document"]
        attr = parsed["Target Attribute"]
        self.calls.append((document, attr))
        values = {
            ("Course title: Algorithms. Credits: 4.", "title"): "Algorithms",
            ("Course title: Algorithms. Credits: 4.", "credits"): "4",
            ("Course title: Databases. Credits: 3.", "title"): "Databases",
            ("Course title: Databases. Credits: 3.", "credits"): "3",
            ("Instructor name: Ada.", "name"): "Ada",
        }
        return type("Result", (), {"root": {attr: values[(document, attr)]}})()


class RuntimeBackedPrompt:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime


class CountingFullDocBatchRuntime:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def complete_model(self, request: Any, *_args: Any, **_kwargs: Any) -> Any:
        content = request.messages[0]["content"]
        payload = json.loads(content.split("Input:\n", 1)[1])
        doc_ids = [item["doc_id"] for item in payload["Documents"]]
        self.calls.append(doc_ids)
        results: Dict[str, Dict[str, Any]] = {}
        for item in payload["Documents"]:
            doc_id = item["doc_id"]
            document = item["Document"]
            if "Instructor" in document:
                results[doc_id] = {
                    "Table Assignment": "instructor",
                    "Data Extracted": {"name": "Ada"},
                }
            elif "Algorithms" in document:
                results[doc_id] = {
                    "Table Assignment": "course",
                    "Data Extracted": {"title": "Algorithms", "credits": "4"},
                }
            elif "Systems" in document:
                results[doc_id] = {
                    "Table Assignment": "course",
                    "Data Extracted": {"title": "Systems", "credits": "5"},
                }
            else:
                results[doc_id] = {
                    "Table Assignment": "course",
                    "Data Extracted": {"title": "Databases", "credits": "3"},
                }
        return type("Result", (), {"root": results})()


class MultiRecordFullDocBatchRuntime:
    def complete_model(self, request: Any, *_args: Any, **_kwargs: Any) -> Any:
        content = request.messages[0]["content"]
        payload = json.loads(content.split("Input:\n", 1)[1])
        return type(
            "Result",
            (),
            {
                "root": {
                    item["doc_id"]: {
                        "Records": [
                            {
                                "Table Assignment": "course",
                                "Record ID": "course-a",
                                "Data Extracted": {"title": "Algorithms", "credits": "4"},
                            },
                            {
                                "Table Assignment": "course",
                                "Record ID": "course-b",
                                "Data Extracted": {"title": "Databases", "credits": "3"},
                            },
                        ]
                    }
                    for item in payload["Documents"]
                }
            },
        )()


class FailingPrompt:
    def complete_model(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("materialized query execution should not call prompts")


def _materialized_extractor(
    table_prompt: Any,
    attr_prompt: Any,
    *,
    only: bool = False,
    multi_record: bool = False,
) -> DataExtraction:
    extractor = DataExtraction.__new__(DataExtraction)
    extractor.config = {
        "res_param_str": "unit",
        "exp_query_id_list": ["q1"],
        "training_data_count": 0,
        "materialized_full_extraction": True,
        "materialized_full_extraction_only": only,
        "multi_record_extraction": multi_record,
    }
    extractor.training_data_count = 0
    extractor.training_data_split = "prefix"
    extractor.training_data_split_seed = 42
    extractor.train_doc_ids = []
    extractor.test_doc_ids = []
    extractor._train_doc_ids_set = set()
    extractor._test_doc_ids_set = set()
    extractor.mode = "deepseek"
    extractor.disable_llm = False
    extractor.param_str = "unit"
    extractor.pause_controller = None
    extractor.loader_type = "hf_manifest"
    extractor.loader_config = {}
    extractor.retry_params = {}
    extractor.prompt_table = table_prompt
    extractor.prompt_attr = attr_prompt
    extractor.proxy_runtime_config = {}
    extractor.use_proxy_runtime = False
    extractor.doc_filter_strategy = None
    extractor.doc_filter_config = {}
    extractor.doc_filter_enabled = False
    extractor.doc_filter_only = False
    extractor.result_save_interval = 1
    extractor.materialized_full_extraction_only = only
    extractor.materialized_full_extraction_enabled = True
    extractor.materialized_full_extraction_batch_size = 16
    extractor.materialized_full_extraction_batch_max_chars = 24000
    extractor.materialized_full_extraction_concurrency = 1
    extractor.multi_record_extraction = multi_record
    extractor._materialized_full_extraction_data = None
    extractor._materialized_full_extraction_path = None
    extractor.table_assignment_cache_general_schema = False
    extractor.table_assignment_source_table_metadata = False
    extractor.table_assignment_cache_enabled = False
    return extractor


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
            "redd.core.data_extraction.data_extraction.create_data_loader",
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
        RESULT_RECORDS_KEY: [
            {
                "table": "course",
                "data": {"title": "Algorithms", "credits": "4"},
            }
        ],
    }
    assert result["course-3"] == {
        RESULT_TABLE_KEY: "course",
        RESULT_DATA_KEY: {"title": "Databases", "credits": "3"},
        RESULT_RECORDS_KEY: [
            {
                "table": "course",
                "data": {"title": "Databases", "credits": "3"},
            }
        ],
    }
    assert result["instructor-1"] == {
        RESULT_TABLE_KEY: NULL_VALUE,
        RESULT_DATA_KEY: {},
        RESULT_RECORDS_KEY: [],
    }


def test_ground_truth_extraction_persists_multiple_records_per_doc(tmp_path: Path) -> None:
    extractor = DataExtraction(
        {
            "mode": "ground_truth",
            "res_param_str": "unit",
            "exp_query_id_list": ["q1"],
            "training_data_count": 0,
            "multi_record_extraction": True,
        }
    )

    with (
        patch(
            "redd.core.data_extraction.data_extraction.create_data_loader",
            return_value=MultiRecordGTManifestLoader(),
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

    assert result["bundle-1"][RESULT_TABLE_KEY] == "course"
    assert result["bundle-1"][RESULT_DATA_KEY] == {"title": "Algorithms", "credits": "4"}
    assert result["bundle-1"][RESULT_RECORDS_KEY] == [
        {
            "table": "course",
            "data": {"title": "Algorithms", "credits": "4"},
            "record_id": "bundle-1#0",
        },
        {
            "table": "course",
            "data": {"title": "Databases", "credits": "3"},
            "record_id": "bundle-1#1",
        },
    ]


def test_ground_truth_extraction_defaults_to_primary_record_per_doc(tmp_path: Path) -> None:
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
            "redd.core.data_extraction.data_extraction.create_data_loader",
            return_value=MultiRecordGTManifestLoader(),
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

    assert result["bundle-1"] == {
        RESULT_TABLE_KEY: "course",
        RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
        RESULT_RECORDS_KEY: [
            {
                "table": "course",
                "data": {"title": "Algorithms", "credits": "4"},
                "record_id": "bundle-1#0",
            }
        ],
    }


def test_materialized_full_extraction_first_run_uses_configured_llm_then_queries_lookup(
    tmp_path: Path,
) -> None:
    loader = FakeLLMManifestLoader()
    table_prompt = CountingTablePrompt()
    attr_prompt = CountingAttrPrompt()
    extractor = _materialized_extractor(table_prompt, attr_prompt)

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=loader,
    ):
        extractor._process_dataset(tmp_path / "dataset", tmp_path / "out")

    materialized_path = tmp_path / "out" / PATH_TEMPLATES.materialized_full_extraction("unit")
    materialized = json.loads(materialized_path.read_text())
    assert materialized == {
        "course-4": {
            RESULT_TABLE_KEY: "course",
            RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
            RESULT_RECORDS_KEY: [
                {
                    "table": "course",
                    "data": {"title": "Algorithms", "credits": "4"},
                }
            ],
        },
        "course-3": {
            RESULT_TABLE_KEY: "course",
            RESULT_DATA_KEY: {"title": "Databases", "credits": "3"},
            RESULT_RECORDS_KEY: [
                {
                    "table": "course",
                    "data": {"title": "Databases", "credits": "3"},
                }
            ],
        },
        "instructor-1": {
            RESULT_TABLE_KEY: "instructor",
            RESULT_DATA_KEY: {"name": "Ada"},
            RESULT_RECORDS_KEY: [
                {
                    "table": "instructor",
                    "data": {"name": "Ada"},
                }
            ],
        },
    }

    query_result = json.loads((tmp_path / "out" / "res_tabular_data_q1_unit.json").read_text())
    assert query_result == {
        "course-4": {
            RESULT_TABLE_KEY: "course",
            RESULT_DATA_KEY: {"title": "Algorithms"},
            RESULT_RECORDS_KEY: [
                {
                    "table": "course",
                    "data": {"title": "Algorithms"},
                }
            ],
        },
        "course-3": {
            RESULT_TABLE_KEY: "course",
            RESULT_DATA_KEY: {"title": "Databases"},
            RESULT_RECORDS_KEY: [
                {
                    "table": "course",
                    "data": {"title": "Databases"},
                }
            ],
        },
        "instructor-1": {
            RESULT_TABLE_KEY: NULL_VALUE,
            RESULT_DATA_KEY: {},
            RESULT_RECORDS_KEY: [],
        },
    }
    assert len(table_prompt.calls) == 3
    assert len(attr_prompt.calls) == 5


def test_materialized_full_extraction_uses_runtime_batch_output(
    tmp_path: Path,
) -> None:
    runtime = CountingFullDocBatchRuntime()
    prompt = RuntimeBackedPrompt(runtime)
    extractor = _materialized_extractor(prompt, prompt, only=True)
    extractor.materialized_full_extraction_batch_size = 3
    extractor.config["materialized_full_extraction_batch_size"] = 3

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FakeLLMManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", tmp_path / "out")

    materialized = json.loads(
        (tmp_path / "out" / PATH_TEMPLATES.materialized_full_extraction("unit")).read_text()
    )
    assert materialized["course-4"][RESULT_DATA_KEY] == {
        "title": "Algorithms",
        "credits": "4",
    }
    assert materialized["course-3"][RESULT_DATA_KEY] == {
        "title": "Databases",
        "credits": "3",
    }
    assert materialized["instructor-1"][RESULT_DATA_KEY] == {"name": "Ada"}
    assert runtime.calls == [["course-4", "course-3", "instructor-1"]]


def test_materialized_full_extraction_supports_multi_record_runtime_output(
    tmp_path: Path,
) -> None:
    prompt = RuntimeBackedPrompt(MultiRecordFullDocBatchRuntime())
    extractor = _materialized_extractor(prompt, prompt, only=True, multi_record=True)

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FakeLLMManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", tmp_path / "out")

    materialized = json.loads(
        (tmp_path / "out" / PATH_TEMPLATES.materialized_full_extraction("unit")).read_text()
    )
    assert materialized["course-4"][RESULT_TABLE_KEY] == "course"
    assert materialized["course-4"][RESULT_DATA_KEY] == {
        "title": "Algorithms",
        "credits": "4",
    }
    assert materialized["course-4"][RESULT_RECORDS_KEY] == [
        {
            "table": "course",
            "data": {"title": "Algorithms", "credits": "4"},
            "record_id": "course-a",
        },
        {
            "table": "course",
            "data": {"title": "Databases", "credits": "3"},
            "record_id": "course-b",
        },
    ]


def test_materialized_full_extraction_defaults_to_primary_record_output(
    tmp_path: Path,
) -> None:
    prompt = RuntimeBackedPrompt(MultiRecordFullDocBatchRuntime())
    extractor = _materialized_extractor(prompt, prompt, only=True)

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FakeLLMManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", tmp_path / "out")

    materialized = json.loads(
        (tmp_path / "out" / PATH_TEMPLATES.materialized_full_extraction("unit")).read_text()
    )
    assert materialized["course-4"] == {
        RESULT_TABLE_KEY: "course",
        RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
        RESULT_RECORDS_KEY: [
            {
                "table": "course",
                "data": {"title": "Algorithms", "credits": "4"},
                "record_id": "course-a",
            }
        ],
    }


def test_materialized_full_extraction_can_run_batches_concurrently(
    tmp_path: Path,
) -> None:
    runtime = CountingFullDocBatchRuntime()
    prompt = RuntimeBackedPrompt(runtime)
    extractor = _materialized_extractor(prompt, prompt, only=True)
    extractor.materialized_full_extraction_batch_size = 2
    extractor.materialized_full_extraction_concurrency = 2
    extractor.config["materialized_full_extraction_batch_size"] = 2
    extractor.config["materialized_full_extraction_concurrency"] = 2

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FourDocLLMManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", tmp_path / "out")

    materialized = json.loads(
        (tmp_path / "out" / PATH_TEMPLATES.materialized_full_extraction("unit")).read_text()
    )
    assert materialized["course-5"][RESULT_DATA_KEY] == {
        "title": "Systems",
        "credits": "5",
    }
    assert sorted(runtime.calls) == sorted(
        [["course-4", "course-3"], ["instructor-1", "course-5"]]
    )


def test_materialized_full_extraction_existing_complete_artifact_skips_llm(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / PATH_TEMPLATES.materialized_full_extraction("unit")).write_text(
        json.dumps(
            {
                "course-4": {
                    RESULT_TABLE_KEY: "course",
                    RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
                },
                "course-3": {
                    RESULT_TABLE_KEY: "course",
                    RESULT_DATA_KEY: {"title": "Databases", "credits": "3"},
                },
                "instructor-1": {
                    RESULT_TABLE_KEY: "instructor",
                    RESULT_DATA_KEY: {"name": "Ada"},
                },
            }
        ),
        encoding="utf-8",
    )
    extractor = _materialized_extractor(FailingPrompt(), FailingPrompt())

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FakeLLMManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", out_root)

    query_result = json.loads((out_root / "res_tabular_data_q1_unit.json").read_text())
    assert query_result["course-4"][RESULT_DATA_KEY] == {"title": "Algorithms"}
    assert query_result["course-3"][RESULT_DATA_KEY] == {"title": "Databases"}
    assert query_result["instructor-1"] == {
        RESULT_TABLE_KEY: NULL_VALUE,
        RESULT_DATA_KEY: {},
        RESULT_RECORDS_KEY: [],
    }


def test_materialized_full_extraction_postprocesses_existing_complete_artifact(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    materialized_path = out_root / PATH_TEMPLATES.materialized_full_extraction("unit")
    materialized_path.write_text(
        json.dumps(
            {
                "company-1": {
                    RESULT_TABLE_KEY: "fortune500_companies",
                    RESULT_DATA_KEY: {
                        "company": "Acme",
                        "growth_in_jobs": NULL_VALUE,
                        "worlds_most_admired_companies": NULL_VALUE,
                        "market_cap_updated_m": NULL_VALUE,
                        "ticker": NULL_VALUE,
                    },
                    RESULT_RECORDS_KEY: [
                        {
                            "table": "fortune500_companies",
                            RESULT_DATA_KEY: {
                                "company": "Acme",
                                "growth_in_jobs": NULL_VALUE,
                                "worlds_most_admired_companies": NULL_VALUE,
                                "market_cap_updated_m": NULL_VALUE,
                                "ticker": NULL_VALUE,
                            },
                        }
                    ],
                },
                "company-2": {
                    RESULT_TABLE_KEY: "fortune500_companies",
                    RESULT_DATA_KEY: {
                        "company": "Beta",
                        "growth_in_jobs": NULL_VALUE,
                        "worlds_most_admired_companies": NULL_VALUE,
                        "market_cap_updated_m": NULL_VALUE,
                        "ticker": NULL_VALUE,
                    },
                    RESULT_RECORDS_KEY: [
                        {
                            "table": "fortune500_companies",
                            RESULT_DATA_KEY: {
                                "company": "Beta",
                                "growth_in_jobs": NULL_VALUE,
                                "worlds_most_admired_companies": NULL_VALUE,
                                "market_cap_updated_m": NULL_VALUE,
                                "ticker": NULL_VALUE,
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    extractor = _materialized_extractor(FailingPrompt(), FailingPrompt())

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FortunePostprocessManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", out_root)

    materialized = json.loads(materialized_path.read_text())
    company_1 = materialized["company-1"][RESULT_DATA_KEY]
    assert company_1["growth_in_jobs"] == "1"
    assert company_1["worlds_most_admired_companies"] == "1"
    assert company_1["market_cap_updated_m"] == "12300"
    assert company_1["ticker"] == "ACME"

    company_2 = materialized["company-2"][RESULT_DATA_KEY]
    assert company_2["growth_in_jobs"] == "0"
    assert company_2["worlds_most_admired_companies"] == "0"
    assert company_2["market_cap_updated_m"] == NULL_VALUE
    assert company_2["ticker"] == NULL_VALUE


def test_materialized_full_extraction_partial_artifact_is_completed_before_queries(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / PATH_TEMPLATES.materialized_full_extraction("unit")).write_text(
        json.dumps(
            {
                "course-4": {
                    RESULT_TABLE_KEY: "course",
                    RESULT_DATA_KEY: {"title": "Algorithms", "credits": "4"},
                },
                "course-3": {
                    RESULT_TABLE_KEY: "course",
                    RESULT_DATA_KEY: {"title": "Databases"},
                },
                "instructor-1": {
                    RESULT_TABLE_KEY: "instructor",
                    RESULT_DATA_KEY: {"name": "Ada"},
                },
            }
        ),
        encoding="utf-8",
    )
    attr_prompt = CountingAttrPrompt()
    extractor = _materialized_extractor(FailingPrompt(), attr_prompt)

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FakeLLMManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", out_root)

    materialized = json.loads(
        (out_root / PATH_TEMPLATES.materialized_full_extraction("unit")).read_text()
    )
    assert materialized["course-3"][RESULT_DATA_KEY] == {
        "title": "Databases",
        "credits": "3",
    }
    assert attr_prompt.calls == [("Course title: Databases. Credits: 3.", "credits")]

    query_result = json.loads((out_root / "res_tabular_data_q1_unit.json").read_text())
    assert query_result["course-3"][RESULT_DATA_KEY] == {"title": "Databases"}


def test_materialized_full_extraction_only_skips_query_outputs(tmp_path: Path) -> None:
    table_prompt = CountingTablePrompt()
    attr_prompt = CountingAttrPrompt()
    extractor = _materialized_extractor(table_prompt, attr_prompt, only=True)

    with patch(
        "redd.core.data_extraction.data_extraction.create_data_loader",
        return_value=FakeLLMManifestLoader(),
    ):
        extractor._process_dataset(tmp_path / "dataset", tmp_path / "out")

    materialized_path = tmp_path / "out" / PATH_TEMPLATES.materialized_full_extraction("unit")
    assert materialized_path.exists()
    assert not (tmp_path / "out" / "res_tabular_data_q1_unit.json").exists()
    assert len(table_prompt.calls) == 3
    assert len(attr_prompt.calls) == 5


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

    assert res_data == {
        "doc-1": {
            RESULT_TABLE_KEY: NULL_VALUE,
            RESULT_DATA_KEY: {},
            RESULT_RECORDS_KEY: [],
        }
    }


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
