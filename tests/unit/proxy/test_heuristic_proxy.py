from __future__ import annotations

from redd.core.utils.sql_filter_parser import AttributePredicate
from redd.proxy.predicate_proxy.heuristic_proxy import HeuristicPredicateProxy


def test_heuristic_proxy_pass_through_attributes_are_configurable() -> None:
    predicate = AttributePredicate("cname", "=", "Fresno")
    document = "The school is in Oakland."

    conservative = HeuristicPredicateProxy(predicate)
    _, conservative_passed = conservative.evaluate_documents([document])

    strict = HeuristicPredicateProxy(predicate, pass_through_attributes=[])
    _, strict_passed = strict.evaluate_documents([document])

    assert conservative_passed.tolist() == [True]
    assert strict_passed.tolist() == [False]


def test_average_score_proxy_uses_matching_score_field() -> None:
    document = (
        "The results include an average reading score of 445, "
        "a math average of 432, and an average writing score of 434. "
        "A total of 117 students took the SAT."
    )

    math_proxy = HeuristicPredicateProxy(
        AttributePredicate("avg_scr_math", ">=", 473),
        threshold=0.505,
        pass_through_attributes=[],
    )
    read_proxy = HeuristicPredicateProxy(
        AttributePredicate("avg_scr_read", ">=", 445),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, math_passed = math_proxy.evaluate_documents([document])
    _, read_passed = read_proxy.evaluate_documents([document])

    assert math_passed.tolist() == [False]
    assert read_passed.tolist() == [True]


def test_average_score_proxy_handles_loose_section_phrasing() -> None:
    document = (
        "The average reading score reached 432, while in math, students "
        "averaged slightly higher with a score of 448. Writing saw an average "
        "score of 433."
    )

    math_proxy = HeuristicPredicateProxy(
        AttributePredicate("avg_scr_math", ">=", 448),
        threshold=0.505,
        pass_through_attributes=[],
    )
    write_proxy = HeuristicPredicateProxy(
        AttributePredicate("avg_scr_write", ">=", 434),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, math_passed = math_proxy.evaluate_documents([document])
    _, write_passed = write_proxy.evaluate_documents([document])

    assert math_passed.tolist() == [True]
    assert write_passed.tolist() == [False]


def test_sat_count_and_average_score_proxy_handle_test_context() -> None:
    document = (
        "A total of 73 students took the test, demonstrating commitment. "
        "The average scores reveal capability, with reading averaging at 494, "
        "math at 506, and writing at 476. Among these scholars, 38 students "
        "achieved scores above 1500."
    )

    count_proxy = HeuristicPredicateProxy(
        AttributePredicate("num_tst_takr", ">=", 36),
        threshold=0.505,
        pass_through_attributes=[],
    )
    read_proxy = HeuristicPredicateProxy(
        AttributePredicate("avg_scr_read", ">=", 473),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, count_passed = count_proxy.evaluate_documents([document])
    _, read_passed = read_proxy.evaluate_documents([document])

    assert count_passed.tolist() == [True]
    assert read_passed.tolist() == [True]


def test_soccer_physical_attribute_parsers_use_units() -> None:
    document = (
        "Standing tall at an impressive height of 182.88 centimeters and "
        "weighing in at 181 pounds, the player brings a strong frame."
    )

    height_proxy = HeuristicPredicateProxy(
        AttributePredicate("height", ">=", 180),
        threshold=0.505,
        pass_through_attributes=[],
    )
    weight_proxy = HeuristicPredicateProxy(
        AttributePredicate("weight", "<", 180),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, height_passed = height_proxy.evaluate_documents([document])
    _, weight_passed = weight_proxy.evaluate_documents([document])

    assert height_passed.tolist() == [True]
    assert weight_passed.tolist() == [False]


def test_soccer_weight_parser_handles_weighs_phrasing() -> None:
    documents = [
        "The player stands tall at 182.88 centimeters and weighs in at 154 pounds.",
        "The player stands tall at 177.8 centimeters and weighs 176 pounds.",
    ]
    proxy = HeuristicPredicateProxy(
        AttributePredicate("weight", "<", 170),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, passed = proxy.evaluate_documents(documents)

    assert passed.tolist() == [True, False]


def test_soccer_rating_parsers_use_named_fields() -> None:
    document = (
        "The player received an overall performance rating of 69.0. "
        "With a potential rating soaring to 80.0, his future looks promising. "
        "He also displays a commendable level of aggression rated at 63.0."
    )

    overall_proxy = HeuristicPredicateProxy(
        AttributePredicate("overall_rating", ">", 70),
        threshold=0.505,
        pass_through_attributes=[],
    )
    potential_proxy = HeuristicPredicateProxy(
        AttributePredicate("potential", ">", 70),
        threshold=0.505,
        pass_through_attributes=[],
    )
    aggression_proxy = HeuristicPredicateProxy(
        AttributePredicate("aggression", ">", 60),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, overall_passed = overall_proxy.evaluate_documents([document])
    _, potential_passed = potential_proxy.evaluate_documents([document])
    _, aggression_passed = aggression_proxy.evaluate_documents([document])

    assert overall_passed.tolist() == [False]
    assert potential_passed.tolist() == [True]
    assert aggression_passed.tolist() == [True]


def test_row_id_proxy_uses_source_row_metadata() -> None:
    proxy = HeuristicPredicateProxy(
        AttributePredicate("row_id", "<=", 89),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, passed = proxy.evaluate_documents(
        ["The document text does not expose the row identifier."],
        metadata=[{"source_row_id": "55"}],
    )

    assert passed.tolist() == [True]


def test_row_id_proxy_ignores_missing_metadata_candidates() -> None:
    proxy = HeuristicPredicateProxy(
        AttributePredicate("row_id", "<=", 89),
        threshold=0.505,
        pass_through_attributes=[],
    )

    _, passed = proxy.evaluate_documents(
        ["The document text does not expose the row identifier."],
        metadata=[{"source_row_id": "111", "row_id": None, "rowid": None}],
    )

    assert passed.tolist() == [False]


def test_proxy_can_pass_through_configured_doc_ids() -> None:
    proxy = HeuristicPredicateProxy(
        AttributePredicate(attribute="event_name", operator="LIKE", value="%speaker%"),
        threshold=0.505,
        pass_through_attributes=[],
        pass_through_doc_ids=["d1"],
    )

    scores, passed = proxy.evaluate_documents(
        ["The event was a football game.", "The event was a football game."],
        doc_ids=["d1", "d2"],
    )

    assert scores.tolist() == [1.0, 0.0]
    assert passed.tolist() == [True, False]


def test_proxy_can_force_reject_configured_doc_ids() -> None:
    proxy = HeuristicPredicateProxy(
        AttributePredicate(attribute="event_name", operator="LIKE", value="%speaker%"),
        threshold=0.505,
        pass_through_attributes=[],
        force_reject_doc_ids=["d1"],
    )

    scores, passed = proxy.evaluate_documents(
        ["The event was the November Speaker.", "The event was the November Speaker."],
        doc_ids=["d1", "d2"],
    )

    assert scores.tolist() == [0.0, 1.0]
    assert passed.tolist() == [False, True]
