"""
Types for alpha allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


STAGE_DOC_FILTERING = "doc_filtering"
STAGE_PREDICATE_PROXY = "predicate_proxy"

DEFAULT_ALPHA_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]


@dataclass
class AlphaAllocationConfig:
    """Config for joint alpha allocation (doc + predicate stages)."""

    enabled: bool = False
    target_recall: float = 0.95
    alpha_grid: List[float] = field(default_factory=lambda: list(DEFAULT_ALPHA_GRID))

    @classmethod
    def from_raw(cls, raw: Any) -> "AlphaAllocationConfig":
        """Build config from raw dict-like value."""
        if not isinstance(raw, dict):
            return cls()

        enabled = bool(raw.get("enabled", False))

        try:
            target_recall = float(raw.get("target_recall", 0.95))
        except (TypeError, ValueError):
            target_recall = 0.95
        target_recall = min(max(target_recall, 1e-6), 1.0 - 1e-6)

        raw_grid = raw.get("alpha_grid", DEFAULT_ALPHA_GRID)
        grid: List[float] = []
        if isinstance(raw_grid, list):
            for item in raw_grid:
                try:
                    value = float(item)
                except (TypeError, ValueError):
                    continue
                if 0.0 <= value < 1.0:
                    grid.append(value)
        if not grid:
            grid = list(DEFAULT_ALPHA_GRID)
        grid = sorted(set(grid))
        if grid[0] != 0.0:
            grid.insert(0, 0.0)

        return cls(
            enabled=enabled,
            target_recall=target_recall,
            alpha_grid=grid,
        )


@dataclass
class AllocationTraceStep:
    """One greedy update step in alpha allocation."""

    iteration: int
    stage: str
    alpha_before: float
    alpha_after: float
    cost_before: float
    cost_after: float
    budget_before: float
    budget_after: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step to plain dict."""
        return {
            "iteration": self.iteration,
            "stage": self.stage,
            "alpha_before": self.alpha_before,
            "alpha_after": self.alpha_after,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "score": self.score,
        }


@dataclass
class AlphaAllocationResult:
    """Output for alpha allocation."""

    enabled: bool
    alpha_doc_filtering: float
    alpha_predicate_proxy: float
    target_recall_doc_filtering: float
    target_recall_predicate_proxy: float
    estimated_cost: float
    target_recall_global: float
    budget_used: float
    budget_total: float
    trace: List[AllocationTraceStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to plain dict."""
        return {
            "enabled": self.enabled,
            "alpha_doc_filtering": self.alpha_doc_filtering,
            "alpha_predicate_proxy": self.alpha_predicate_proxy,
            "target_recall_doc_filtering": self.target_recall_doc_filtering,
            "target_recall_predicate_proxy": self.target_recall_predicate_proxy,
            "estimated_cost": self.estimated_cost,
            "target_recall_global": self.target_recall_global,
            "budget_used": self.budget_used,
            "budget_total": self.budget_total,
            "trace": [step.to_dict() for step in self.trace],
        }
