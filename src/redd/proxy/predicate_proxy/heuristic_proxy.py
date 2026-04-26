"""Lightweight value-aware predicate proxies.

These proxies are intentionally conservative: they only reject a document when
the predicate value can be checked from nearby text evidence. Unknown cases pass
through to the oracle/extractor.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Tuple

import numpy as np

from redd.core.utils.sql_filter_parser import AttributePredicate

_NUMBER_RE = re.compile(r"(?<![\w.])-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?")
DEFAULT_PASS_THROUGH_ATTRIBUTES = {
    # The demo narrative can paraphrase or even disagree with these GT fields.
    # Rejecting from text evidence here is not recall-safe enough for proxy use.
    "cname",
    "avg_scr_math",
    "avg_scr_read",
    "avg_scr_write",
    "num_ge1500",
    "num_tst_takr",
}
_PASS_THROUGH_ATTRIBUTES = set(DEFAULT_PASS_THROUGH_ATTRIBUTES)
UNKNOWN_EVIDENCE_SCORE = 0.5


def _norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _number_value(raw: str) -> float | None:
    text = raw.replace("$", "").replace(",", "").replace("%", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _predicate_value(value: Any) -> float | str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).strip().strip("'\"")
    numeric = _number_value(text)
    return numeric if numeric is not None else text


def _satisfies(value: float | str, operator: str, expected: float | str) -> bool:
    op = str(operator or "").strip().lower()
    if isinstance(value, (int, float)) and isinstance(expected, (int, float)):
        actual = float(value)
        target = float(expected)
        if op in {"=", "=="}:
            return actual == target
        if op in {"!=", "<>"}:
            return actual != target
        if op == ">":
            return actual > target
        if op == ">=":
            return actual >= target
        if op == "<":
            return actual < target
        if op == "<=":
            return actual <= target
        return False

    actual_text = _norm_text(value)
    target_text = _norm_text(expected)
    if op in {"=", "=="}:
        return bool(target_text and target_text in actual_text)
    if op in {"!=", "<>"}:
        return bool(target_text and target_text not in actual_text)
    return False


def _attribute_hints(attribute: str) -> list[str]:
    attr = str(attribute or "").lower()
    defaults = [part for part in re.split(r"[_\W]+", attr) if part]
    mapping = {
        "avg_scr_math": ["average", "math", "score"],
        "avg_scr_read": ["average", "reading", "read", "score"],
        "avg_scr_write": ["average", "writing", "write", "score"],
        "num_tst_takr": ["test", "took", "taken", "takers"],
        "num_ge1500": ["1500", "above", "scored"],
        "credits": ["credit", "credits"],
        "semester": ["semester"],
        "year": ["year"],
        "salary": ["salary", "earns", "earning", "$"],
    }
    return mapping.get(attr, defaults)


def _nearby_numbers(document: str, attribute: str, radius: int = 90) -> list[float]:
    hints = _attribute_hints(attribute)
    lowered = str(document or "").lower()
    numbers: list[float] = []
    attr_numbers = {
        value
        for value in (_number_value(token) for token in re.findall(r"\d+(?:\.\d+)?", attribute))
        if value is not None
    }
    for match in _NUMBER_RE.finditer(lowered):
        raw = match.group(0)
        value = _number_value(raw)
        if value is None:
            continue
        if value in attr_numbers:
            continue
        start = max(0, match.start() - radius)
        end = min(len(lowered), match.end() + radius)
        window = lowered[start:end]
        if hints and not any(hint and hint in window for hint in hints):
            continue
        numbers.append(value)
    return numbers


def _contains_string_value(document: str, expected: str) -> bool:
    target = _norm_text(expected)
    if not target:
        return True
    return target in _norm_text(document)


class HeuristicPredicateProxy:
    """Conservative rule-based proxy for one SQL predicate."""

    uses_documents = True

    def __init__(
        self,
        predicate: AttributePredicate,
        *,
        name: str | None = None,
        threshold: float = 0.5,
        cost: float = 0.01,
        pass_rate: float = 0.5,
        pass_through_attributes: Iterable[str] | None = None,
    ) -> None:
        self.predicate = predicate
        self.name = name or f"heuristic_{predicate.attribute}"
        self._threshold = float(threshold)
        self._cost = float(cost)
        self._pass_rate = float(pass_rate)
        self._pass_through_attributes = {
            str(attr).lower()
            for attr in (
                _PASS_THROUGH_ATTRIBUTES
                if pass_through_attributes is None
                else pass_through_attributes
            )
        }
        self._total_seen = 0
        self._total_passed = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def pass_rate(self) -> float:
        if self._total_seen > 10:
            return self._total_passed / self._total_seen
        return self._pass_rate

    @property
    def rejection_efficiency(self) -> float:
        if self.cost <= 0:
            return float("inf")
        return (1.0 - self.pass_rate) / self.cost

    def evaluate(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("HeuristicPredicateProxy evaluates raw documents.")

    def evaluate_documents(self, documents: Iterable[str], **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.array([self._score_document(document) for document in documents], dtype=np.float32)
        passed = scores >= self.threshold
        self._total_seen += len(scores)
        self._total_passed += int(passed.sum())
        return scores, passed

    def _score_document(self, document: str) -> float:
        if str(self.predicate.attribute or "").lower() in self._pass_through_attributes:
            return 1.0

        expected = _predicate_value(self.predicate.value)
        operator = self.predicate.operator

        if isinstance(expected, str):
            return 1.0 if _contains_string_value(document, expected) else 0.0

        candidates = _nearby_numbers(document, self.predicate.attribute)
        if not candidates:
            # Unknown: passes at the default 0.5 threshold but can be rejected
            # by raising proxy_threshold when a run allows lower recall.
            return UNKNOWN_EVIDENCE_SCORE
        return 1.0 if any(_satisfies(candidate, operator, expected) for candidate in candidates) else 0.0
