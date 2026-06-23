from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REMOVE_COLUMNS: dict[str, set[str]] = {
    # Mostly absent in the narrative and consistently missed by extraction.
    "spider.apartment_rentals.default_task": {
        "apartment_buildings.building_short_name",
        "apartments.building_short_name",
    },
    # Narrative documents do not expose these school administrative fields, or the
    # field is all-null in GT and encourages hallucinated values.
    "bird.california_schools.default_task": {
        "frpm.charter_school",
        "schools.doc",
        "schools.doc_type",
        "schools.ed_ops_code",
        "schools.eil_code",
        "schools.g_soffered",
        "schools.g_sserved",
        "schools.mail_city",
        "schools.mail_state",
        "schools.mail_str_abr",
        "schools.mail_street",
        "schools.mail_zip",
        "schools.open_date",
        "schools.phone",
        "schools.soc",
        "schools.soc_type",
        "schools.street_abr",
    },
    "bird.schools_demo": {
        "schools.adm_email2",
        "schools.adm_email3",
        "schools.adm_f_name2",
        "schools.adm_f_name3",
        "schools.adm_l_name2",
        "schools.adm_l_name3",
        "schools.charter_num",
        "schools.closed_date",
        "schools.doc",
        "schools.doc_type",
        "schools.ed_ops_code",
        "schools.eil_code",
        "schools.ext",
        "schools.funding_type",
        "schools.g_soffered",
        "schools.g_sserved",
        "schools.mail_city",
        "schools.mail_state",
        "schools.mail_str_abr",
        "schools.mail_street",
        "schools.mail_zip",
        "schools.open_date",
        "schools.phone",
        "schools.soc",
        "schools.soc_type",
        "schools.street_abr",
        "schools.website",
    },
    # These GT cells are all null; leaving them in rewards hallucinated prose.
    "spider.college_2.course_teaches_instructor": {
        "course_information.course_description",
    },
    "spider.flight_4.routes_airports_airlines": {
        "flight_routes.operation_focus",
        "flight_routes.route_type",
    },
    "spider.wine_1.wine_appellations": {
        "appellation.terroir_description",
        "wines.vineyard_name",
    },
    # These weather/GK micro-fields are not stated in the generated documents.
    "spider.bike_1.default_task": {
        "weather.cloud_cover",
        "weather.wind_dir_degrees",
    },
    "spider.soccer_1.default_task": {
        "player_attributes.gk_kicking",
        "player_attributes.gk_positioning",
        "player_attributes.gk_reflexes",
    },
}


DESCRIPTION_PATCHES: dict[str, dict[str, str]] = {
    "spider.apartment_rentals.default_task": {
        "apartment_bookings.booking_end_date": (
            "Exact booking end timestamp. Preserve the complete date and time exactly as "
            "stated, including hour, minute, and seconds when present; do not truncate to "
            "date-only or round seconds."
        ),
        "apartment_bookings.booking_start_date": (
            "Exact booking start timestamp. Preserve the complete date and time exactly as "
            "stated, including hour, minute, and seconds when present; do not truncate to "
            "date-only or round seconds."
        ),
        "apartment_buildings.building_address": (
            "Full physical address of the apartment building. Include street number, "
            "street name, city, state, and ZIP/postal code exactly as stated; do not omit "
            "the postal code or city/state."
        ),
        "guests.date_of_birth": (
            "Exact guest birth date or timestamp as stated. Preserve all provided date "
            "and time precision; do not invent or remove seconds."
        ),
        "view_unit_status.booking_end_date": (
            "Exact booking end timestamp associated with the unit status. Preserve all "
            "provided time precision, including seconds."
        ),
        "view_unit_status.booking_start_date": (
            "Exact booking start timestamp associated with the unit status. Preserve all "
            "provided time precision, including seconds."
        ),
        "view_unit_status.status_date": (
            "Exact status timestamp for the unit. Preserve the full date and time, "
            "including seconds when present; do not output date-only if time is stated."
        ),
    },
    "spider.bike_1.default_task": {
        "status.time": (
            "Exact status timestamp. Preserve seconds when present and do not round or "
            "default missing seconds to :00 unless explicitly stated."
        ),
        "trip.end_date": (
            "Exact trip end timestamp. Preserve the date and time exactly as stated, "
            "including minute/second precision; do not collapse to date-only."
        ),
        "trip.start_date": (
            "Exact trip start timestamp. Preserve the date and time exactly as stated, "
            "including minute/second precision; do not collapse to date-only."
        ),
        "weather.date": "Exact calendar date of the weather record.",
    },
    "bird.student_club.default_task": {
        "attendance.event_date": (
            "Exact event datetime in ISO 8601 format. Preserve the stated hour, minute, "
            "and seconds; do not default to midnight when a time is provided."
        ),
        "budget.event_date": (
            "Exact event datetime in ISO 8601 format. Preserve the stated hour, minute, "
            "and seconds; do not default to midnight when a time is provided."
        ),
        "event.event_date": (
            "Exact event datetime in ISO 8601 format. Preserve the stated hour, minute, "
            "and seconds; do not default to midnight when a time is provided."
        ),
    },
    "galois.fortune.default_task": {
        "fortune500_companies.best_companies_to_work_for": (
            "Closed-world boolean for Fortune Best Companies to Work For membership. "
            "Output false/0 when the document states the company is not on the list or "
            "does not recognize it as being on the list; do not leave blank for ordinary "
            "non-membership."
        ),
        "fortune500_companies.dropped_in_rank": (
            "Closed-world boolean indicating whether rank declined. Derive from the "
            "change in rank when stated: negative change means true; otherwise false."
        ),
        "fortune500_companies.founder_is_ceo": (
            "Closed-world boolean indicating whether the CEO is also the founder. Output "
            "false/0 when the document does not identify the CEO as the founder."
        ),
        "fortune500_companies.gained_in_rank": (
            "Closed-world boolean indicating whether rank improved. Derive from the "
            "change in rank when stated: positive change means true; otherwise false."
        ),
        "fortune500_companies.global500": (
            "Closed-world boolean for Fortune Global 500 membership. Output false/0 for "
            "ordinary non-membership rather than null."
        ),
        "fortune500_companies.growth_in_jobs": (
            "Closed-world boolean indicating whether the company shows growth in jobs or "
            "workforce expansion. Output true/1 when the document states job growth, job "
            "creation, employment growth, workforce expansion, or increased employment; "
            "otherwise output false/0 rather than null."
        ),
        "fortune500_companies.is_female_ceo": (
            "Closed-world boolean indicating whether the CEO is female. Output false/0 "
            "unless the CEO is identified as female."
        ),
        "fortune500_companies.is_profitable": (
            "Closed-world boolean indicating whether profits are positive. Derive from "
            "the profits value when stated: positive profits mean true, negative or zero "
            "means false."
        ),
        "fortune500_companies.newcomer_to_the_fortune500": (
            "Closed-world boolean indicating whether the company is a newcomer to the "
            "Fortune 500. Output false/0 when the document indicates it is not a newcomer "
            "or gives an ordinary prior ranking/history."
        ),
        "fortune500_companies.worlds_most_admired_companies": (
            "Closed-world boolean for World's Most Admired Companies membership. Output "
            "false/0 for ordinary non-membership rather than null."
        ),
    },
}


def full_column_id(table_id: str, column: dict[str, Any]) -> str:
    column_id = str(column.get("column_id") or "")
    if "." in column_id:
        return column_id
    return f"{table_id}.{column.get('name')}"


def patch_description(dataset_id: str, full_id: str, column: dict[str, Any]) -> bool:
    patches = DESCRIPTION_PATCHES.get(dataset_id, {})
    if full_id in patches:
        column["description"] = patches[full_id]
        return True

    name = str(column.get("name") or "")
    if dataset_id in {"bird.california_schools.default_task", "bird.schools_demo"}:
        if name.startswith("percent_eligible_"):
            column["description"] = (
                "Eligibility percentage stored as a decimal fraction between 0 and 1. "
                "If the document says a percent such as 87.4%, output 0.874, not 87.4."
            )
            return True
    return False


def collect_query_refs(query: Any) -> set[str]:
    refs: set[str] = set()

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                walk(child, key)
            return
        if not isinstance(value, str):
            return
        if key in {
            "required_columns",
            "output_columns",
            "columns",
            "predicate_column",
            "join_columns",
        }:
            refs.add(value)

    walk(query)
    sql = ""
    if isinstance(query, dict):
        sql = str(query.get("sql") or "")
    for name in re.findall(r'"([^"]+)"', sql):
        refs.add(name)
    return refs


def query_uses_removed_column(query: Any, remove_full: set[str]) -> bool:
    if not remove_full:
        return False
    remove_bare = {item.split(".", 1)[1] for item in remove_full if "." in item}
    refs = collect_query_refs(query)
    for ref in refs:
        if ref in remove_full:
            return True
        if "." not in ref and ref in remove_bare:
            # SQL aliases make table disambiguation unreliable; prefer removing the
            # query over leaving a dangling reference to a removed schema column.
            return True
    text = json.dumps(query, ensure_ascii=False)
    return any(f'"{item}"' in text for item in remove_full)


def filter_query_file(path: Path, remove_full: set[str]) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        return (0, 0)
    original = len(data["queries"])
    data["queries"] = [
        query for query in data["queries"] if not query_uses_removed_column(query, remove_full)
    ]
    removed = original - len(data["queries"])
    if removed:
        if "num_queries" in data:
            data["num_queries"] = len(data["queries"])
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return (original, removed)


def repair_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_id = dataset_dir.name
    schema_path = dataset_dir / "metadata/schema.json"
    gt_path = dataset_dir / "data/ground_truth.parquet"
    if not schema_path.exists() or not gt_path.exists():
        return {"dataset_id": dataset_id, "skipped": True}

    remove_full = REMOVE_COLUMNS.get(dataset_id, set())
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_removed: list[str] = []
    descriptions_updated: list[str] = []
    for table in schema.get("tables") or []:
        table_id = str(table.get("table_id") or table.get("name") or "")
        kept_columns = []
        for column in table.get("columns") or []:
            full_id = full_column_id(table_id, column)
            if full_id in remove_full:
                schema_removed.append(full_id)
                continue
            if patch_description(dataset_id, full_id, column):
                descriptions_updated.append(full_id)
            kept_columns.append(column)
        table["columns"] = kept_columns
    if schema_removed or descriptions_updated:
        schema_path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    gt = pd.read_parquet(gt_path)
    before_gt = len(gt)
    if remove_full:
        row_full_ids = gt["table_id"].astype(str) + "." + gt["column_name"].astype(str)
        gt = gt.loc[~row_full_ids.isin(remove_full)].copy()
        if len(gt) != before_gt:
            gt.to_parquet(gt_path, index=False)

    query_reports: dict[str, dict[str, int]] = {}
    for rel in [
        "metadata/queries.json",
        "metadata/query_sets/generated_queries.json",
        "metadata/query_sets/queries_proposed.json",
    ]:
        original, removed = filter_query_file(dataset_dir / rel, remove_full)
        if original or removed:
            query_reports[rel] = {"original": original, "removed": removed}

    return {
        "dataset_id": dataset_id,
        "schema_removed": schema_removed,
        "descriptions_updated": descriptions_updated,
        "ground_truth_rows_before": int(before_gt),
        "ground_truth_rows_after": int(len(gt)),
        "ground_truth_rows_removed": int(before_gt - len(gt)),
        "queries": query_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/derived")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--report", default="outputs/materialized_full_extraction_all/reports/attr_scope_repair_report.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    dataset_names = args.dataset or sorted(set(REMOVE_COLUMNS) | set(DESCRIPTION_PATCHES))
    reports = []
    for dataset_name in dataset_names:
        reports.append(repair_dataset(dataset_root / dataset_name))

    payload = {"datasets": reports}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
