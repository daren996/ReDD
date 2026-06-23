from __future__ import annotations

from redd.core.utils.constants import RESULT_RECORDS_KEY
from redd.core.utils.extraction_records import make_result_entry


def test_make_result_entry_does_not_persist_internal_record_index() -> None:
    entry = make_result_entry(
        [
            {
                "table": "course",
                "data": {"title": "Algorithms"},
                "_record_index": 0,
            }
        ]
    )

    assert "_record_index" not in entry[RESULT_RECORDS_KEY][0]
