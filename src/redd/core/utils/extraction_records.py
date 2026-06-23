"""Utilities for legacy and multi-record extraction result entries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from .constants import (
    DATA_EXTRACTED_KEY,
    NULL_VALUE,
    RESULT_DATA_KEY,
    RESULT_RECORD_ID_KEY,
    RESULT_RECORD_TABLE_KEY,
    RESULT_RECORDS_KEY,
    RESULT_TABLE_KEY,
    TABLE_ASSIGNMENT_KEY,
)
from .utils import is_none_value


def make_result_record(
    table: Any,
    data: Optional[Dict[str, Any]] = None,
    *,
    record_id: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create one normalized record object for result entries."""
    record: Dict[str, Any] = {
        RESULT_RECORD_TABLE_KEY: NULL_VALUE if is_none_value(table) else str(table),
        RESULT_DATA_KEY: data if isinstance(data, dict) else {},
    }
    if record_id is not None and str(record_id) != "":
        record[RESULT_RECORD_ID_KEY] = str(record_id)
    if extra:
        for key, value in extra.items():
            if key not in record:
                record[key] = value
    return record


def make_result_entry(
    records: Iterable[Dict[str, Any]] | None = None,
    *,
    reason: Any = None,
    include_records_for_single: bool = True,
) -> Dict[str, Any]:
    """Create a result entry with primary fields plus the canonical records list."""
    clean_records = [
        make_result_record(
            record.get(RESULT_RECORD_TABLE_KEY, record.get(RESULT_TABLE_KEY)),
            record.get(RESULT_DATA_KEY, {}),
            record_id=record.get(RESULT_RECORD_ID_KEY),
            extra={
                key: value
                for key, value in record.items()
                if (
                    key
                    and not str(key).startswith("_")
                    and key
                    not in {
                        RESULT_RECORD_TABLE_KEY,
                        RESULT_TABLE_KEY,
                        RESULT_DATA_KEY,
                        RESULT_RECORD_ID_KEY,
                    }
                )
            },
        )
        for record in (records or [])
        if isinstance(record, dict)
        and not is_none_value(record.get(RESULT_RECORD_TABLE_KEY, record.get(RESULT_TABLE_KEY)))
        and record.get(RESULT_RECORD_TABLE_KEY, record.get(RESULT_TABLE_KEY)) != NULL_VALUE
    ]
    entry: Dict[str, Any] = {}
    if clean_records:
        primary = clean_records[0]
        entry[RESULT_TABLE_KEY] = primary.get(RESULT_RECORD_TABLE_KEY)
        entry[RESULT_DATA_KEY] = deepcopy(primary.get(RESULT_DATA_KEY, {}))
        if len(clean_records) > 1 or include_records_for_single:
            entry[RESULT_RECORDS_KEY] = clean_records
    else:
        entry[RESULT_TABLE_KEY] = NULL_VALUE
        entry[RESULT_DATA_KEY] = {}
        if include_records_for_single:
            entry[RESULT_RECORDS_KEY] = []
    if reason is not None:
        entry["reason"] = reason
    return entry


def make_legacy_result_entry(table: Any, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a single-record result entry in the canonical output shape."""
    return make_result_entry(
        [make_result_record(table, data)],
        include_records_for_single=True,
    )


def result_entry_records(entry: Any) -> List[Dict[str, Any]]:
    """Return normalized records from either the new or legacy result shape."""
    if not isinstance(entry, dict):
        return []

    raw_records = entry.get(RESULT_RECORDS_KEY)
    if isinstance(raw_records, list):
        records: List[Dict[str, Any]] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, dict):
                continue
            table = raw_record.get(RESULT_RECORD_TABLE_KEY)
            if table is None:
                table = raw_record.get(RESULT_TABLE_KEY)
            if table is None:
                table = raw_record.get(TABLE_ASSIGNMENT_KEY)
            data = raw_record.get(RESULT_DATA_KEY)
            if data is None:
                data = raw_record.get(DATA_EXTRACTED_KEY)
            if not isinstance(data, dict):
                data = {}
            record_id = raw_record.get(RESULT_RECORD_ID_KEY, raw_record.get("Record ID"))
            record = make_result_record(table, data, record_id=record_id)
            record["_record_index"] = index
            records.append(record)
        return records

    if RESULT_TABLE_KEY not in entry and RESULT_DATA_KEY not in entry:
        return []
    table = entry.get(RESULT_TABLE_KEY)
    data = entry.get(RESULT_DATA_KEY, {})
    if not isinstance(data, dict):
        data = {}
    record = make_result_record(table, data, record_id=entry.get(RESULT_RECORD_ID_KEY))
    record["_record_index"] = 0
    return [record]


def active_result_records(entry: Any) -> List[Dict[str, Any]]:
    """Return records whose table is not null/None."""
    return [
        record
        for record in result_entry_records(entry)
        if not is_none_value(record.get(RESULT_RECORD_TABLE_KEY))
        and record.get(RESULT_RECORD_TABLE_KEY) != NULL_VALUE
    ]


def sync_legacy_primary_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the first active record into legacy ``res``/``data`` fields."""
    records = active_result_records(entry)
    if not records:
        entry[RESULT_TABLE_KEY] = NULL_VALUE
        entry[RESULT_DATA_KEY] = {}
        return entry
    primary = records[0]
    entry[RESULT_TABLE_KEY] = primary.get(RESULT_RECORD_TABLE_KEY)
    data = primary.get(RESULT_DATA_KEY, {})
    entry[RESULT_DATA_KEY] = data if isinstance(data, dict) else {}
    return entry


def update_result_record_data(
    entry: Dict[str, Any],
    record_index: int,
    attr: str,
    value: Any,
) -> None:
    """Set a record attribute in either a multi-record or legacy entry."""
    raw_records = entry.get(RESULT_RECORDS_KEY)
    if isinstance(raw_records, list):
        if 0 <= record_index < len(raw_records) and isinstance(raw_records[record_index], dict):
            data = raw_records[record_index].setdefault(RESULT_DATA_KEY, {})
            if not isinstance(data, dict):
                data = {}
                raw_records[record_index][RESULT_DATA_KEY] = data
            data[attr] = value
            sync_legacy_primary_fields(entry)
        return

    data = entry.setdefault(RESULT_DATA_KEY, {})
    if not isinstance(data, dict):
        data = {}
        entry[RESULT_DATA_KEY] = data
    data[attr] = value


def replace_result_records(entry: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replace an entry's records and keep legacy primary fields coherent."""
    next_entry = make_result_entry(records, include_records_for_single=True)
    entry.clear()
    entry.update(next_entry)
    return entry
