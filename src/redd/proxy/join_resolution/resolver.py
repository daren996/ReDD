"""
Join-resolution proxy for the proxy runtime.

A proxy that filters documents by join-key membership: for a child table in a
join, it extracts the join attribute from the document and rejects values not
present in the parent table's extracted set.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = ["JoinResolver", "create_join_resolver"]


def _normalize_for_membership(v: Any, case_insensitive: bool = True) -> Any:
    """Normalize value for set membership check."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    s = str(v).strip()
    if not s:
        return None
    return s.lower() if case_insensitive else s


def _build_allowed_set(values: List[Any], case_insensitive: bool = True) -> Set[Any]:
    """Build a set of normalized values for membership testing."""
    result: Set[Any] = set()
    for v in values:
        n = _normalize_for_membership(v, case_insensitive)
        if n is not None:
            result.add(n)
    return result


class JoinResolver:
    """
    Proxy that extracts the join attribute from documents and rejects if
    the value is not in the allowed set (from parent table extractions).
    
    Uses document text; implements uses_documents=True for ProxyExecutor.
    """

    uses_documents = True

    def __init__(
        self,
        name: str,
        attr: str,
        allowed_set: Set[Any],
        extract_fn: Callable[..., Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        cost: float = 0.8,
        pass_rate: float = 0.5,
        case_insensitive: bool = True,
    ):
        """
        Initialize JoinResolver.
        
        Args:
            name: Proxy name (e.g., "join_instructor_name")
            attr: Join attribute to extract from document
            allowed_set: Set of allowed values (from parent table)
            extract_fn: Callable(document, schema, attributes) -> {attr: value}
            schema: Schema dict for extraction context
            threshold: Score threshold (1.0 = in set, 0.0 = not in set)
            cost: Relative cost (LLM extraction is expensive)
            pass_rate: Estimated fraction that pass
            case_insensitive: Normalize strings for membership check
        """
        self.name = name
        self.attr = attr
        self._allowed_set = _build_allowed_set(list(allowed_set), case_insensitive)
        self._extract_fn = extract_fn
        self._schema = schema or {}
        self._threshold = threshold
        self._cost = cost
        self._pass_rate = pass_rate
        self._case_insensitive = case_insensitive

        self._total_seen = 0
        self._total_passed = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

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
        """
        JoinResolver uses documents. Call evaluate_documents instead.
        """
        raise NotImplementedError(
            "JoinResolver uses documents. Call evaluate_documents() or ensure "
            "executor passes documents for proxies with uses_documents=True."
        )

    def evaluate_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract join attribute from each document and check membership.
        
        Args:
            documents: List of document texts
            doc_ids: Optional doc IDs (for oracle extraction)
            
        Returns:
            (scores, passed_mask) - score 1.0 if in set, 0.0 otherwise
        """
        if not documents:
            return np.array([]), np.array([], dtype=bool)

        scores = np.zeros(len(documents), dtype=np.float32)
        passed_mask = np.zeros(len(documents), dtype=bool)

        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else None
            try:
                extracted = self._extract_fn(
                    doc,
                    self._schema,
                    [self.attr],
                    doc_id=doc_id,
                )
                value = extracted.get(self.attr) if isinstance(extracted, dict) else None
                norm = _normalize_for_membership(value, self._case_insensitive)
                in_set = norm is not None and norm in self._allowed_set
                scores[i] = 1.0 if in_set else 0.0
                passed_mask[i] = in_set
            except Exception as e:
                logging.warning(f"[JoinResolver] Extraction failed for doc {i}: {e}")
                scores[i] = 0.0
                passed_mask[i] = False

            self._total_seen += 1
            if passed_mask[i]:
                self._total_passed += 1

        return scores, passed_mask

    def reset_stats(self):
        """Reset runtime statistics."""
        self._total_seen = 0
        self._total_passed = 0


def create_join_resolver(
    attr: str,
    allowed_set: Set[Any],
    oracle: Any,
    schema: Dict[str, Any],
    table_name: Optional[str] = None,
    cost: float = 0.8,
    pass_rate: float = 0.5,
) -> JoinResolver:
    """
    Create a JoinResolver using an oracle for extraction.
    
    Args:
        attr: Join attribute to extract
        allowed_set: Set of allowed values from parent table
        oracle: Oracle with extract(document, schema, attributes, doc_id)
        schema: Schema dict for the child table
        table_name: Optional table name for proxy naming
        cost: Relative cost
        pass_rate: Estimated pass rate
        
    Returns:
        Configured JoinResolver
    """
    def extract_fn(doc: str, sch: Dict, attrs: List[str], **kwargs):
        return oracle.extract(document=doc, schema=sch, attributes=attrs, doc_id=kwargs.get("doc_id"))

    name = f"join_{attr}"
    if table_name:
        name = f"join_{table_name}_{attr}"

    return JoinResolver(
        name=name,
        attr=attr,
        allowed_set=allowed_set,
        extract_fn=extract_fn,
        schema=schema,
        cost=cost,
        pass_rate=pass_rate,
    )
