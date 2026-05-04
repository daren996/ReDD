from __future__ import annotations

from redd.core.utils.sql_filter_parser import SQLFilterParser


def test_sql_filter_parser_unwraps_casted_identifier_predicates() -> None:
    parser = SQLFilterParser(strip_table_aliases=True)

    predicates = parser.parse(
        'SELECT "row_id" FROM "major" WHERE (CAST("row_id" AS REAL) <= 89);'
    )

    assert len(predicates) == 1
    assert predicates[0].attribute == "row_id"
    assert predicates[0].operator == "<="
    assert predicates[0].value == 89


def test_sql_filter_parser_unwraps_lower_casted_string_predicates() -> None:
    parser = SQLFilterParser(strip_table_aliases=True)

    predicates = parser.parse(
        'SELECT * FROM "attendance" WHERE (LOWER(CAST("event_name" AS TEXT)) = "Party");'
    )

    assert len(predicates) == 1
    assert predicates[0].attribute == "event_name"
    assert predicates[0].operator == "="
    assert predicates[0].value == "Party"
