from __future__ import annotations

from redd.core.data_extraction.res_to_db import ResToDBConverter
from redd.core.utils.constants import RESULT_DATA_KEY, RESULT_RECORDS_KEY, RESULT_TABLE_KEY


def test_group_no_chunking_preserves_multiple_records_per_document() -> None:
    converter = ResToDBConverter.__new__(ResToDBConverter)

    table_data = converter._group_no_chunking(
        {
            "doc-1": {
                RESULT_TABLE_KEY: "course",
                RESULT_DATA_KEY: {"title": "Algorithms"},
                RESULT_RECORDS_KEY: [
                    {
                        "table": "course",
                        "record_id": "course-a",
                        "data": {"title": "Algorithms"},
                    },
                    {
                        "table": "course",
                        "record_id": "course-b",
                        "data": {"title": "Databases"},
                    },
                ],
            }
        },
        {},
    )

    assert table_data == {
        "course": {
            "course-a": {"title": "Algorithms"},
            "course-b": {"title": "Databases"},
        }
    }


def test_group_with_chunking_merges_single_record_entries_with_records_key() -> None:
    converter = ResToDBConverter.__new__(ResToDBConverter)
    converter.merge_strategy = "first_non_null"

    table_data = converter._group_with_chunking(
        {
            "doc-1#0": {
                RESULT_TABLE_KEY: "course",
                RESULT_DATA_KEY: {"title": "Algorithms"},
                RESULT_RECORDS_KEY: [
                    {
                        "table": "course",
                        "data": {"title": "Algorithms"},
                    }
                ],
            },
            "doc-1#1": {
                RESULT_TABLE_KEY: "course",
                RESULT_DATA_KEY: {"credits": "4"},
                RESULT_RECORDS_KEY: [
                    {
                        "table": "course",
                        "data": {"credits": "4"},
                    }
                ],
            },
        },
        {
            "doc-1#0": {"parent_doc_id": "doc-1", "chunk_index": 0},
            "doc-1#1": {"parent_doc_id": "doc-1", "chunk_index": 1},
        },
    )

    assert table_data == {
        "course": {
            "doc-1": {
                "title": "Algorithms",
                "credits": "4",
            }
        }
    }
