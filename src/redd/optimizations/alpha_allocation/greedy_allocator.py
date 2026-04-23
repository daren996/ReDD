"""
Greedy joint alpha allocator.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

from .types import AllocationTraceStep


class GreedyAlphaAllocator:
    """
    Greedy discrete alpha allocator based on the paper's Algorithm 4.
    """

    def __init__(
        self,
        stages: List[str],
        alpha_grid: List[float],
        target_recall: float,
        cost_fn: Callable[[Dict[str, float]], float],
    ):
        if not stages:
            raise ValueError("stages must not be empty")
        if not alpha_grid:
            raise ValueError("alpha_grid must not be empty")

        self.stages = list(stages)
        self.alpha_grid = sorted(set(float(v) for v in alpha_grid))
        self.target_recall = float(target_recall)
        self.cost_fn = cost_fn

    @staticmethod
    def _b(alpha: float) -> float:
        """Budget transform b(alpha) = -log(1 - alpha)."""
        alpha = min(max(alpha, 0.0), 1.0 - 1e-12)
        return -math.log(1.0 - alpha)

    def _budget_used(self, alphas: Dict[str, float]) -> float:
        return sum(self._b(alphas.get(stage, 0.0)) for stage in self.stages)

    def _next_alpha(self, current: float) -> float | None:
        for value in self.alpha_grid:
            if value > current + 1e-12:
                return value
        return None

    def allocate(
        self,
    ) -> Tuple[Dict[str, float], float, float, float, List[AllocationTraceStep]]:
        """
        Run greedy allocation.

        Returns:
            (alphas, final_cost, budget_used, budget_total, trace)
        """
        budget_total = -math.log(max(self.target_recall, 1e-12))
        alphas = {stage: self.alpha_grid[0] for stage in self.stages}
        trace: List[AllocationTraceStep] = []
        cache: Dict[Tuple[str, ...], float] = {}

        def cached_cost(cur: Dict[str, float]) -> float:
            key = tuple(f"{stage}:{cur.get(stage, 0.0):.12f}" for stage in self.stages)
            if key in cache:
                return cache[key]
            value = float(self.cost_fn(cur))
            cache[key] = value
            return value

        if self._budget_used(alphas) > budget_total + 1e-12:
            final_cost = cached_cost(alphas)
            return alphas, final_cost, self._budget_used(alphas), budget_total, trace

        iteration = 0
        while self._budget_used(alphas) <= budget_total + 1e-12:
            iteration += 1
            current_cost = cached_cost(alphas)
            current_budget = self._budget_used(alphas)

            best_stage: str | None = None
            best_alpha: float | None = None
            best_score = 0.0
            best_cost = current_cost
            best_budget = current_budget

            for stage in self.stages:
                current_alpha = alphas[stage]
                alpha_next = self._next_alpha(current_alpha)
                if alpha_next is None:
                    continue

                candidate = dict(alphas)
                candidate[stage] = alpha_next
                candidate_budget = self._budget_used(candidate)
                if candidate_budget > budget_total + 1e-12:
                    continue

                candidate_cost = cached_cost(candidate)
                delta_cost = candidate_cost - current_cost
                delta_budget = self._b(alpha_next) - self._b(current_alpha)

                if delta_cost >= 0 or delta_budget <= 0:
                    continue

                score = (-delta_cost) / delta_budget
                if score > best_score:
                    best_stage = stage
                    best_alpha = alpha_next
                    best_score = score
                    best_cost = candidate_cost
                    best_budget = candidate_budget

            if best_stage is None or best_alpha is None:
                break

            step = AllocationTraceStep(
                iteration=iteration,
                stage=best_stage,
                alpha_before=alphas[best_stage],
                alpha_after=best_alpha,
                cost_before=current_cost,
                cost_after=best_cost,
                budget_before=current_budget,
                budget_after=best_budget,
                score=best_score,
            )
            trace.append(step)
            alphas[best_stage] = best_alpha

        final_cost = cached_cost(alphas)
        return alphas, final_cost, self._budget_used(alphas), budget_total, trace
