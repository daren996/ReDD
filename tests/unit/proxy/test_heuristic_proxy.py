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
