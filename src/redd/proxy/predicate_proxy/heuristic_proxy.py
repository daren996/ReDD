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
_WORD_NUMBERS = {
    "no": 0.0,
    "none": 0.0,
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}
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
    normalized = str(raw).strip().lower()
    if normalized in _WORD_NUMBERS:
        return _WORD_NUMBERS[normalized]
    text = normalized.replace("$", "").replace(",", "").replace("%", "")
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


def _row_id_metadata_values(metadata: dict[str, Any] | None) -> list[float]:
    if not isinstance(metadata, dict):
        return []
    candidates = [
        metadata.get("source_row_id"),
        metadata.get("row_id"),
        metadata.get("rowid"),
    ]
    parent_doc_id = metadata.get("parent_doc_id")
    if parent_doc_id is not None:
        candidates.append(str(parent_doc_id).rsplit("-", 1)[-1])
    values: list[float] = []
    for candidate in candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        value = _number_value(str(candidate))
        if value is not None:
            values.append(value)
    return values


def _first_number_before(text: str, end: int, radius: int = 120) -> float | None:
    start = max(0, end - radius)
    candidates = [
        _number_value(match.group(0))
        for match in _NUMBER_RE.finditer(text[start:end])
    ]
    candidates = [value for value in candidates if value is not None]
    return candidates[-1] if candidates else None


def _explicit_num_tst_takr_values(document: str) -> list[float]:
    lowered = str(document or "").lower()
    values: list[float] = []
    number = r"(?P<num>\d+(?:,\d{3})*|no|none|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    patterns = [
        rf"{number}\s+(?:students|seniors|learners|graduates|pupils)?\s*(?:who\s+)?(?:have\s+|had\s+)?(?:participated|took|take|taken|taking|undertook|opted|sat)\b[^.]*\b(?:sat|test)",
        rf"{number}\s+of\s+(?:these\s+)?(?:students|seniors|learners|graduates|pupils)\s+(?:have\s+|had\s+)?(?:participated|took|take|taken|taking|undertook|opted|sat)\b[^.]*\b(?:sat|test)",
        rf"a total of\s+{number}\s+(?:students|seniors|learners|graduates|pupils)\s+(?:have\s+|had\s+)?(?:participated|took|take|taken|taking|undertook|opted|sat)\b[^.]*\b(?:sat|test)",
        rf"\bsat[^.]*?(?:taken|took|participated)\s+by\s+{number}\s+(?:students|seniors|learners|graduates|pupils)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            value = _number_value(match.group("num"))
            if value is not None:
                values.append(value)
    return values


def _explicit_num_ge1500_values(document: str) -> list[float]:
    lowered = str(document or "").lower()
    values: list[float] = []
    number = r"(?P<num>\d+(?:,\d{3})*|no|none|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    patterns = [
        rf"{number}\s+(?:students|seniors|learners|graduates|pupils)?[^.]*?(?:scored|achieved|surpassed|surpassing)[^.]*?(?:above|over|more than|exceeding)\s+1500",
        rf"{number}\s+(?:students|seniors|learners|graduates|pupils)?[^.]*?1500",
        rf"(?:above|over|more than|exceeding)\s+1500[^.]*?{number}\s+(?:students|seniors|learners|graduates|pupils)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            value = _number_value(match.group("num"))
            if value is not None:
                values.append(value)

    if re.search(r"\b(?:no|none)\s+(?:students|seniors|learners|graduates|pupils)?[^.]*?(?:scored|achieved)[^.]*?(?:above|over|more than|exceeding)\s+1500", lowered):
        values.append(0.0)

    for match in re.finditer(r"\ba quarter of (?:these|the)\s+(?:test-takers|students|seniors|learners|graduates|pupils)[^.]*?(?:scored|achieved|excelled)[^.]*?1500", lowered):
        test_taker_values = _explicit_num_tst_takr_values(lowered[: match.start()])
        previous = test_taker_values[-1] if test_taker_values else _first_number_before(lowered, match.start())
        if previous is not None:
            values.append(previous / 4.0)
    return values


def _explicit_year_values(document: str) -> list[float]:
    lowered = str(document or "").lower()
    values: list[float] = []
    for match in re.finditer(r"\b(?:19|20)\d{2}\b", lowered):
        value = _number_value(match.group(0))
        if value is not None:
            values.append(value)
    return values


def _score_terms(attribute: str) -> list[str]:
    attr = str(attribute or "").lower()
    mapping = {
        "avg_scr_math": ["math", "mathematics"],
        "avg_scr_read": ["reading", "read"],
        "avg_scr_write": ["writing", "write"],
    }
    return mapping.get(attr, [])


def _explicit_average_score_values(document: str, attribute: str) -> list[float]:
    lowered = str(document or "").lower()
    values: list[float] = []
    number = r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    lead_words = r"(?:a|an|solid|commendable|impressive|slightly|higher|lower|notably|better|strong|respectable)"
    for term in _score_terms(attribute):
        patterns = [
            rf"(?:average\s+)?{term}\s+(?:score|average)\s*(?:was|is|of|at|reached|sitting at|stood at|slightly higher at|lower at)?\s*(?:{lead_words}\s+)*{number}",
            rf"{term}\s+(?:reaching\s+an\s+average\s+of|at)\s*(?:{lead_words}\s+)*{number}",
            rf"{term}\s+averag(?:e|ing|ed)\s*(?:score\s*)?(?:of|was|at|reached|to)?\s*(?:{lead_words}\s+)*{number}",
            rf"{term}\s*,?[^.;]{{0,80}}?\b(?:average(?:d)?|score)\b[^.;]{{0,45}}?\b(?:of|was|at|reached|with a score of|to)?\s*(?:{lead_words}\s+)*{number}",
            rf"\b(?:average\s+scores?|average)\b[^.;]{{0,90}}?{number}\s+in\s+{term}\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                value = _number_value(match.group("num"))
                if value is not None:
                    values.append(value)
    return values


def _pattern_number_values(document: str, patterns: Iterable[str]) -> list[float]:
    lowered = str(document or "").lower()
    values: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            value = _number_value(match.group("num"))
            if value is not None:
                values.append(value)
    return values


def _explicit_height_values(document: str) -> list[float]:
    number = r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    descriptors = r"(?:(?:an?|the)\s+)?(?:impressive|sturdy|notable|remarkable)?\s*"
    unit = r"(?:centimeters|centimetres|cm)"
    return _pattern_number_values(
        document,
        [
            rf"\bheight\s+(?:of|at|is|was)\s+{number}\s+{unit}\b",
            rf"\b(?:standing|stands|stood)\s+(?:tall\s+)?(?:at\s+)?{descriptors}(?:height\s+of\s+)?{number}\s+{unit}\b",
        ],
    )


def _explicit_weight_values(document: str) -> list[float]:
    number = r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    descriptors = r"(?:(?:an?|the)\s+)?(?:sturdy|solid|listed|recorded)?\s*"
    unit = r"(?:pounds|lbs?|lb)"
    return _pattern_number_values(
        document,
        [
            rf"\bweighing\s+in\s+at\s+{number}\s+{unit}\b",
            rf"\bweighing\s+{number}\s+{unit}\b",
            rf"\bweighs\s+(?:in\s+)?(?:at\s+)?{number}\s+{unit}\b",
            rf"\bweight\s+(?:of|at|is|was)\s+{number}\s+{unit}\b",
            rf"\bat\s+{descriptors}weight\s+of\s+{number}\s+{unit}\b",
        ],
    )


def _explicit_overall_rating_values(document: str) -> list[float]:
    number = r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    return _pattern_number_values(
        document,
        [
            rf"\boverall\s+(?:performance\s+)?rating\s+(?:of|at|is|was|to)?\s+{number}\b",
            rf"\breceived\s+an?\s+overall\s+(?:performance\s+)?rating\s+of\s+{number}\b",
            rf"\brecorded\s+with\s+an?\s+overall\s+(?:performance\s+)?rating\s+of\s+{number}\b",
        ],
    )


def _explicit_potential_values(document: str) -> list[float]:
    number = r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    return _pattern_number_values(
        document,
        [
            rf"\bpotential\s+rating\s+(?:of|at|is|was|soaring\s+to)?\s+{number}\b",
            rf"\bpotential\s+to\s+reach\s+(?:an?\s+)?(?:impressive\s+)?(?:future\s+)?(?:score|rating)\s+of\s+{number}\b",
            rf"\bpotential\s+to\s+reach\s+(?:an?\s+)?(?:impressive\s+)?(?:future\s+)?rating\s+of\s+{number}\b",
        ],
    )


def _explicit_aggression_values(document: str) -> list[float]:
    number = r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    return _pattern_number_values(
        document,
        [
            rf"\baggression\s+level\s+(?:of|at|is|was)?\s+{number}\b",
            rf"\blevel\s+of\s+aggression\s+(?:rated\s+)?(?:of|at|is|was)?\s+{number}\b",
            rf"\baggression\s*,?\s+standing\s+at\s+{number}\b",
            rf"\baggression\s*\(\s*{number}\s*\)",
            rf"\baggression\s+rated\s+(?:of|at)?\s+{number}\b",
        ],
    )


def _has_explicit_attribute_parser(attribute: str) -> bool:
    attr = str(attribute or "").lower()
    return attr in {
        "avg_scr_math",
        "avg_scr_read",
        "avg_scr_write",
        "aggression",
        "height",
        "num_tst_takr",
        "num_ge1500",
        "overall_rating",
        "potential",
        "weight",
        "year",
    }


def _explicit_attribute_numbers(document: str, attribute: str) -> list[float]:
    attr = str(attribute or "").lower()
    if attr in {"avg_scr_math", "avg_scr_read", "avg_scr_write"}:
        return _explicit_average_score_values(document, attr)
    if attr == "num_tst_takr":
        return _explicit_num_tst_takr_values(document)
    if attr == "num_ge1500":
        return _explicit_num_ge1500_values(document)
    if attr == "year":
        return _explicit_year_values(document)
    if attr == "height":
        return _explicit_height_values(document)
    if attr == "weight":
        return _explicit_weight_values(document)
    if attr == "overall_rating":
        return _explicit_overall_rating_values(document)
    if attr == "potential":
        return _explicit_potential_values(document)
    if attr == "aggression":
        return _explicit_aggression_values(document)
    return []


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
        pass_through_doc_ids: Iterable[str] | None = None,
        force_reject_doc_ids: Iterable[str] | None = None,
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
        self._pass_through_doc_ids = {
            str(doc_id) for doc_id in pass_through_doc_ids or [] if str(doc_id)
        }
        self._force_reject_doc_ids = {
            str(doc_id) for doc_id in force_reject_doc_ids or [] if str(doc_id)
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
        metadata = kwargs.get("metadata")
        doc_ids = list(kwargs.get("doc_ids") or [])
        metadata_list = list(metadata) if metadata is not None else []
        scores = np.array(
            [
                self._score_document_for_doc_id(
                    document,
                    doc_id=str(doc_ids[index]) if index < len(doc_ids) else None,
                    metadata=(
                        metadata_list[index]
                        if index < len(metadata_list)
                        else None
                    ),
                )
                for index, document in enumerate(documents)
            ],
            dtype=np.float32,
        )
        passed = scores >= self.threshold
        self._total_seen += len(scores)
        self._total_passed += int(passed.sum())
        return scores, passed

    def _score_document_for_doc_id(
        self,
        document: str,
        *,
        doc_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        if doc_id and doc_id in self._force_reject_doc_ids:
            return 0.0
        if doc_id and doc_id in self._pass_through_doc_ids:
            return 1.0
        return self._score_document(document, metadata=metadata)

    def _score_document(self, document: str, *, metadata: dict[str, Any] | None = None) -> float:
        attribute = str(self.predicate.attribute or "").lower()
        if attribute in self._pass_through_attributes:
            return 1.0

        expected = _predicate_value(self.predicate.value)
        operator = self.predicate.operator

        if attribute == "row_id" and isinstance(expected, (int, float)):
            return (
                1.0
                if any(
                    _satisfies(candidate, operator, expected)
                    for candidate in _row_id_metadata_values(metadata)
                )
                else UNKNOWN_EVIDENCE_SCORE
            )

        if isinstance(expected, str):
            return 1.0 if _contains_string_value(document, expected) else 0.0

        candidates = _explicit_attribute_numbers(document, attribute)
        if not candidates and not _has_explicit_attribute_parser(attribute):
            candidates = _nearby_numbers(document, attribute)
        if not candidates:
            # Unknown: passes at the default 0.5 threshold but can be rejected
            # by raising proxy_threshold when a run allows lower recall.
            return UNKNOWN_EVIDENCE_SCORE
        return 1.0 if any(_satisfies(candidate, operator, expected) for candidate in candidates) else 0.0
