from __future__ import annotations

from redd.core.utils.sql_filter_parser import (
    SQLFilterParser,
    analyze_sql_predicates,
    group_predicates_by_table,
    parse_alias_mapping,
    parse_join_conditions,
)


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


def test_sql_filter_parser_does_not_split_and_inside_string_literal() -> None:
    parser = SQLFilterParser(strip_table_aliases=False)

    predicates = parser.parse(
        'SELECT "sname" FROM "satscores" WHERE "dname" <> \'Oakland Unified\';'
    )

    assert len(predicates) == 1
    assert predicates[0].attribute == "dname"
    assert predicates[0].operator == "<>"
    assert predicates[0].value == "Oakland Unified"


def test_sql_filter_parser_preserves_quoted_table_aliases() -> None:
    parser = SQLFilterParser(strip_table_aliases=False)

    predicates = parser.parse(
        'SELECT "income"."member_last_name" FROM "income" '
        'JOIN "member" ON "income"."member_first_name" = "member"."first_name" '
        'WHERE "member"."major_department" IN ('
        "'School of Applied Sciences, Technology and Education', "
        "'Civil and Environmental Engineering Department');"
    )

    assert len(predicates) == 1
    assert predicates[0].table_alias == "member"
    assert predicates[0].attribute == "major_department"
    assert predicates[0].operator == "IN"


def test_quoted_alias_mapping_predicate_grouping_and_join_conditions() -> None:
    sql = (
        'SELECT "course"."title" FROM "course" '
        'JOIN "instructor" ON "course"."dept_name" = "instructor"."dept_name" '
        'WHERE "instructor"."salary" <= 121141.99;'
    )
    schema = [
        {"Schema Name": "course", "Attributes": [{"Attribute Name": "title"}, {"Attribute Name": "dept_name"}]},
        {"Schema Name": "instructor", "Attributes": [{"Attribute Name": "dept_name"}, {"Attribute Name": "salary"}]},
    ]

    assert parse_alias_mapping(sql) == {
        "course": "course",
        "instructor": "instructor",
    }

    grouped = group_predicates_by_table(sql, schema, query_tables=["course", "instructor"])
    assert list(grouped) == ["instructor"]
    assert grouped["instructor"][0].attribute == "salary"

    joins = parse_join_conditions(sql)
    assert len(joins) == 1
    assert str(joins[0]) == "course.dept_name = instructor.dept_name"


def test_left_join_alias_mapping_and_join_conditions() -> None:
    sql = (
        'SELECT "apartments"."apt_number" FROM "apartments" '
        'LEFT JOIN "apartment_bookings" '
        'ON "apartments"."apt_number" = "apartment_bookings"."apt_number" '
        'WHERE "apartments"."bathroom_count" >= 1;'
    )

    assert parse_alias_mapping(sql) == {
        "apartments": "apartments",
        "apartment_bookings": "apartment_bookings",
    }

    joins = parse_join_conditions(sql)
    assert len(joins) == 1
    assert str(joins[0]) == "apartments.apt_number = apartment_bookings.apt_number"


def test_full_outer_join_alias_mapping_and_join_conditions() -> None:
    sql = (
        'SELECT "schools"."school" FROM "schools" '
        'FULL OUTER JOIN "frpm" '
        'ON "schools"."school" = "frpm"."school_name" '
        'FULL OUTER JOIN "satscores" '
        'ON "schools"."school" = "satscores"."sname";'
    )

    assert parse_alias_mapping(sql) == {
        "schools": "schools",
        "frpm": "frpm",
        "satscores": "satscores",
    }

    joins = parse_join_conditions(sql)
    assert len(joins) == 2
    assert str(joins[0]) == "schools.school = frpm.school_name"
    assert str(joins[1]) == "schools.school = satscores.sname"


def test_set_operation_predicates_are_unsafe_for_row_proxy() -> None:
    schema = [
        {
            "Schema Name": "routes",
            "Attributes": [
                {"Attribute Name": "airline_name"},
                {"Attribute Name": "codeshare"},
            ],
        }
    ]
    sql = (
        "SELECT DISTINCT airline_name FROM routes "
        "EXCEPT SELECT DISTINCT airline_name FROM routes WHERE codeshare = 'None';"
    )

    analysis = analyze_sql_predicates(sql, schema, query_tables=["routes"])

    assert analysis.has_set_operation is True
    assert analysis.requires_full_join_reasoning is True
    assert analysis.safe_predicates_by_table == {}
    assert analysis.unsafe_predicates[0]["table"] == "routes"
    assert "set_operation_query" in analysis.unsafe_predicates[0]["reasons"]


def test_subquery_and_column_comparison_predicates_are_unsafe() -> None:
    schema = [
        {"Schema Name": "airports", "Attributes": [{"Attribute Name": "country"}]},
        {
            "Schema Name": "routes",
            "Attributes": [
                {"Attribute Name": "dst_ap_country"},
                {"Attribute Name": "src_ap_country"},
            ],
        },
    ]

    subquery = analyze_sql_predicates(
        "SELECT a.country FROM airports a "
        "WHERE a.country IN (SELECT al.country FROM airlines al WHERE al.active = 'Y');",
        schema,
        query_tables=["airports"],
    )
    column_compare = analyze_sql_predicates(
        "SELECT airline_name FROM routes WHERE dst_ap_country != src_ap_country;",
        schema,
        query_tables=["routes"],
    )

    assert subquery.safe_predicates_by_table == {}
    assert subquery.has_subquery_predicate is True
    assert "subquery_predicate" in subquery.unsafe_predicates[0]["reasons"]
    assert column_compare.safe_predicates_by_table == {}
    assert column_compare.has_column_comparison is True
    assert "column_comparison" in column_compare.unsafe_predicates[0]["reasons"]


def test_literal_predicate_remains_safe() -> None:
    schema = [
        {
            "Schema Name": "apartment_bookings",
            "Attributes": [{"Attribute Name": "booking_status_code"}],
        }
    ]

    analysis = analyze_sql_predicates(
        "SELECT * FROM apartment_bookings WHERE booking_status_code = 'Confirmed';",
        schema,
        query_tables=["apartment_bookings"],
    )

    assert list(analysis.safe_predicates_by_table) == ["apartment_bookings"]
    assert analysis.safe_predicates_by_table["apartment_bookings"][0].attribute == "booking_status_code"
    assert analysis.unsafe_predicates == []
