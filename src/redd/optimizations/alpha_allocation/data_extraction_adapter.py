"""
Data-extraction adapter for query-level alpha allocation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from redd.core.utils.constants import RESULT_DATA_KEY, RESULT_TABLE_KEY, SCHEMA_NAME_KEY
from redd.core.utils.sql_filter_parser import group_predicates_by_table
from redd.embedding import (
    EmbeddingManager,
    embedding_manager_kwargs,
    resolve_embedding_storage_path,
)
from redd.exp.evaluation import EvalDataExtraction
from redd.optimizations.doc_filtering import create_doc_filter
from redd.optimizations.doc_filtering.runtime import normalize_doc_filter_config
from redd.proxy.proxy_runtime.config import (
    normalize_proxy_runtime_config,
    resolve_proxy_flag,
    resolve_proxy_threshold,
)
from redd.proxy.proxy_runtime.oracle import GoldenOracle
from redd.proxy.proxy_runtime.pipeline import ProxyPipeline
from redd.proxy.proxy_runtime.types import ProxyPipelineConfig

from .greedy_allocator import GreedyAlphaAllocator
from .types import (
    STAGE_DOC_FILTERING,
    STAGE_PREDICATE_PROXY,
    AlphaAllocationConfig,
    AlphaAllocationResult,
)


class DataExtractionAlphaAllocator:
    """Adapter that estimates per-query alpha split on training-prefix docs."""

    def __init__(
        self,
        extraction_config: Dict[str, Any],
        data_path: Path,
        loader: Any,
        api_key: Optional[str],
        train_doc_ids: List[str],
    ):
        self.extraction_config = extraction_config
        self.data_path = Path(data_path)
        self.loader = loader
        self.api_key = api_key or extraction_config.get("api_key")
        self.train_doc_ids = list(train_doc_ids)

        self.alloc_config = AlphaAllocationConfig.from_raw(
            extraction_config.get("alpha_allocation")
        )
        self.doc_filter_config = normalize_doc_filter_config(
            extraction_config.get("doc_filter")
        )
        self.proxy_runtime_config = normalize_proxy_runtime_config(extraction_config)
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._embedding_managers: Dict[Tuple[str, str], EmbeddingManager] = {}

    @staticmethod
    def _clip_target_recall(value: float) -> float:
        """
        Clamp target recall to strict open interval (0, 1).
        """
        return min(max(float(value), 1e-6), 1.0 - 1e-6)

    def allocate_for_query(
        self,
        query_id: str,
        schema_query: List[Dict[str, Any]],
    ) -> Optional[AlphaAllocationResult]:
        """Allocate alpha for one query."""
        if not self.alloc_config.enabled:
            return None
        if not self.train_doc_ids:
            logging.warning(
                "[DataExtractionAlphaAllocator:allocate_for_query] No training docs; skip allocation."
            )
            return None

        query_context = self._build_query_context(query_id=query_id, schema_query=schema_query)
        alpha_map, cost_value, budget_used, budget_total, trace = self._allocate_base_alpha_map(
            query_context=query_context,
        )

        answer_calibration: Dict[str, Any] = {}
        if self.alloc_config.answer_recall_calibration:
            alpha_map, answer_calibration = self._calibrate_predicate_alpha_for_answer_recall(
                query_context=query_context,
                alpha_map=alpha_map,
                budget_total=budget_total,
            )
            cost_value = self._evaluate_cost(
                query_context=query_context,
                alpha_doc=alpha_map.get(STAGE_DOC_FILTERING, 0.0),
                alpha_predicate=alpha_map.get(STAGE_PREDICATE_PROXY, 0.0),
            )

        result = self._build_result(
            query_context=query_context,
            alpha_map=alpha_map,
            cost_value=cost_value,
            budget_total=budget_total,
            trace=trace,
            answer_calibration=answer_calibration,
        )
        logging.info(
            "[DataExtractionAlphaAllocator:allocate_for_query] Query %s -> "
            "alpha_doc=%.4f alpha_predicate=%.4f cost=%.4f",
            query_id,
            result.alpha_doc_filtering,
            result.alpha_predicate_proxy,
            result.estimated_cost,
        )
        return result

    def allocate_for_queries(
        self,
        query_schemas: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, AlphaAllocationResult]:
        """
        Allocate alpha for a group of queries with dataset-level answer recall calibration.

        The final experiment recall is reported as a weighted answer recall over all
        answers, while per-query calibration optimizes each query independently. This
        group mode chooses one candidate alpha per query so the training split's
        aggregate answer recall is closest to the configured target.
        """
        if not self.alloc_config.enabled or not self.train_doc_ids:
            return {}
        if not (
            self.alloc_config.answer_recall_calibration
            and self.alloc_config.answer_recall_calibration_global
        ):
            return {}

        plans: Dict[str, Dict[str, Any]] = {}
        for query_id, schema_query in query_schemas.items():
            query_context = self._build_query_context(
                query_id=query_id,
                schema_query=schema_query,
            )
            alpha_map, cost_value, budget_used, budget_total, trace = (
                self._allocate_base_alpha_map(query_context=query_context)
            )
            del budget_used
            observations = self._answer_recall_candidate_observations(
                query_context=query_context,
                alpha_map=alpha_map,
                budget_total=budget_total,
            )
            plans[query_id] = {
                "query_context": query_context,
                "alpha_map": alpha_map,
                "cost_value": cost_value,
                "budget_total": budget_total,
                "trace": trace,
                "observations": observations,
            }

        selections = self._select_global_answer_recall_observations(plans)
        results: Dict[str, AlphaAllocationResult] = {}
        for query_id, plan in plans.items():
            selected_observation = selections.get(query_id)
            alpha_map = dict(plan["alpha_map"])
            answer_calibration: Dict[str, Any]
            if selected_observation is None:
                answer_calibration = {
                    "enabled": True,
                    "scope": "global",
                    "skipped": "answer_recall_not_executable",
                    "observations": plan["observations"],
                }
            else:
                chosen_alpha = float(selected_observation["alpha_predicate_proxy"])
                alpha_map[STAGE_PREDICATE_PROXY] = chosen_alpha
                answer_calibration = {
                    "enabled": True,
                    "scope": "global",
                    "target_recall": float(self.alloc_config.target_recall),
                    "selected_alpha_predicate_proxy": chosen_alpha,
                    "selected_target_recall_predicate_proxy": self._clip_target_recall(
                        1.0 - chosen_alpha
                    ),
                    "selected_answer_recall": float(selected_observation["recall"]),
                    "selected_answer_covered": int(selected_observation["covered"]),
                    "selected_answer_total": int(selected_observation["total"]),
                    "global_training_answer_recall": selections.get(
                        "__global_training_answer_recall__"
                    ),
                    "global_training_answer_covered": selections.get(
                        "__global_training_answer_covered__"
                    ),
                    "global_training_answer_total": selections.get(
                        "__global_training_answer_total__"
                    ),
                    "observations": plan["observations"],
                }
            cost_value = self._evaluate_cost(
                query_context=plan["query_context"],
                alpha_doc=alpha_map.get(STAGE_DOC_FILTERING, 0.0),
                alpha_predicate=alpha_map.get(STAGE_PREDICATE_PROXY, 0.0),
            )
            results[query_id] = self._build_result(
                query_context=plan["query_context"],
                alpha_map=alpha_map,
                cost_value=cost_value,
                budget_total=float(plan["budget_total"]),
                trace=plan["trace"],
                answer_calibration=answer_calibration,
            )
        return results

    def _allocate_base_alpha_map(
        self,
        *,
        query_context: Dict[str, Any],
    ) -> Tuple[Dict[str, float], float, float, float, List[Any]]:
        stages = [STAGE_DOC_FILTERING, STAGE_PREDICATE_PROXY]

        def evaluate(alphas: Dict[str, float]) -> float:
            return self._evaluate_cost(
                query_context=query_context,
                alpha_doc=alphas.get(STAGE_DOC_FILTERING, 0.0),
                alpha_predicate=alphas.get(STAGE_PREDICATE_PROXY, 0.0),
            )

        allocator = GreedyAlphaAllocator(
            stages=stages,
            alpha_grid=self.alloc_config.alpha_grid,
            target_recall=self.alloc_config.target_recall,
            cost_fn=evaluate,
        )
        alpha_map, cost_value, budget_used, budget_total, trace = allocator.allocate()
        alpha_map = self._use_remaining_budget_for_predicate_proxy(
            alpha_map=alpha_map,
            budget_total=budget_total,
        )
        budget_used = self._budget_used(alpha_map)
        cost_value = evaluate(alpha_map)
        return alpha_map, cost_value, budget_used, budget_total, trace

    def _build_result(
        self,
        *,
        query_context: Dict[str, Any],
        alpha_map: Dict[str, float],
        cost_value: float,
        budget_total: float,
        trace: List[Any],
        answer_calibration: Dict[str, Any],
    ) -> AlphaAllocationResult:
        del query_context
        alpha_doc = float(alpha_map.get(STAGE_DOC_FILTERING, 0.0))
        alpha_predicate = float(alpha_map.get(STAGE_PREDICATE_PROXY, 0.0))
        return AlphaAllocationResult(
            enabled=True,
            alpha_doc_filtering=alpha_doc,
            alpha_predicate_proxy=alpha_predicate,
            target_recall_doc_filtering=self._clip_target_recall(1.0 - alpha_doc),
            target_recall_predicate_proxy=self._clip_target_recall(1.0 - alpha_predicate),
            estimated_cost=float(cost_value),
            target_recall_global=self.alloc_config.target_recall,
            budget_used=float(self._budget_used(alpha_map)),
            budget_total=float(budget_total),
            trace=trace,
            answer_recall_calibration=answer_calibration,
        )

    @staticmethod
    def _alpha_budget(alpha: float) -> float:
        """Budget transform b(alpha) = -log(1 - alpha)."""
        alpha = min(max(float(alpha), 0.0), 1.0 - 1e-12)
        return -math.log(1.0 - alpha)

    def _budget_used(self, alphas: Dict[str, float]) -> float:
        return self._alpha_budget(alphas.get(STAGE_DOC_FILTERING, 0.0)) + self._alpha_budget(
            alphas.get(STAGE_PREDICATE_PROXY, 0.0)
        )

    def _use_remaining_budget_for_predicate_proxy(
        self,
        *,
        alpha_map: Dict[str, float],
        budget_total: float,
    ) -> Dict[str, float]:
        """
        Carry otherwise-unused global recall budget into predicate calibration.

        The greedy allocator only takes steps that reduce estimated training cost.
        With target-insensitive doc filters or conservative proxy estimates, it can
        leave all alphas at zero, which makes the configured global target invisible
        to the downstream runtime. Filling remaining budget on the predicate stage
        preserves the joint budget constraint while keeping target recall tunable.
        """
        filled = dict(alpha_map)
        current_predicate = float(filled.get(STAGE_PREDICATE_PROXY, 0.0))
        best_predicate = current_predicate
        for candidate in self.alloc_config.alpha_grid:
            candidate = float(candidate)
            if candidate <= current_predicate + 1e-12:
                continue
            trial = dict(filled)
            trial[STAGE_PREDICATE_PROXY] = candidate
            if self._budget_used(trial) <= budget_total + 1e-12:
                best_predicate = candidate
        filled[STAGE_PREDICATE_PROXY] = best_predicate
        return filled

    def _calibrate_predicate_alpha_for_answer_recall(
        self,
        *,
        query_context: Dict[str, Any],
        alpha_map: Dict[str, float],
        budget_total: float,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Empirically map the predicate alpha to training SQL-answer recall.

        Predicate-positive calibration can still keep final SQL-answer recall
        much higher than the requested target because many rejected rows do not
        participate in the answer. This optional pass probes the configured
        alpha grid on training docs and picks the predicate alpha whose estimated
        SQL-answer recall is closest to the global target.
        """
        if not str((query_context.get("query_info") or {}).get("sql") or "").strip():
            return dict(alpha_map), {"enabled": True, "skipped": "query_has_no_sql"}

        alpha_doc = float(alpha_map.get(STAGE_DOC_FILTERING, 0.0))
        target = float(self.alloc_config.target_recall)
        observations = self._answer_recall_candidate_observations(
            query_context=query_context,
            alpha_map=alpha_map,
            budget_total=budget_total,
        )
        if not observations:
            return dict(alpha_map), {
                "enabled": True,
                "skipped": "answer_recall_not_executable",
                "observations": observations,
            }
        best: Dict[str, Any] | None = None
        del alpha_doc
        for observation in observations:
            candidate = float(observation["alpha_predicate_proxy"])
            recall = float(observation["recall"])
            distance = abs(recall - target)
            over_target = recall >= target
            if best is None:
                best = {"observation": observation, "distance": distance, "over_target": over_target}
                continue
            best_obs = best["observation"]
            if over_target and not bool(best["over_target"]):
                best = {"observation": observation, "distance": distance, "over_target": over_target}
            elif over_target == bool(best["over_target"]) and distance < best["distance"] - 1e-12:
                best = {"observation": observation, "distance": distance, "over_target": over_target}
            elif over_target == bool(best["over_target"]) and abs(distance - best["distance"]) <= 1e-12:
                # Equal training recall should keep the held-out path conservative.
                if candidate < float(best_obs["alpha_predicate_proxy"]):
                    best = {"observation": observation, "distance": distance, "over_target": over_target}

        if not observations or best is None:
            return dict(alpha_map), {
                "enabled": True,
                "skipped": "answer_recall_not_executable",
                "observations": observations,
            }

        chosen = dict(alpha_map)
        chosen_alpha = float(best["observation"]["alpha_predicate_proxy"])
        chosen[STAGE_PREDICATE_PROXY] = chosen_alpha
        return chosen, {
            "enabled": True,
            "target_recall": target,
            "selected_alpha_predicate_proxy": chosen_alpha,
            "selected_target_recall_predicate_proxy": self._clip_target_recall(1.0 - chosen_alpha),
            "selected_answer_recall": float(best["observation"]["recall"]),
            "observations": observations,
        }

    def _answer_recall_candidate_observations(
        self,
        *,
        query_context: Dict[str, Any],
        alpha_map: Dict[str, float],
        budget_total: float,
    ) -> List[Dict[str, Any]]:
        alpha_doc = float(alpha_map.get(STAGE_DOC_FILTERING, 0.0))
        candidates: List[float] = []
        for value in self.alloc_config.alpha_grid:
            candidate = float(value)
            trial = dict(alpha_map)
            trial[STAGE_PREDICATE_PROXY] = candidate
            if (
                not self.alloc_config.answer_recall_calibration_allow_over_budget
                and self._budget_used(trial) > budget_total + 1e-12
            ):
                continue
            candidates.append(candidate)

        observations: List[Dict[str, Any]] = []
        for candidate in candidates:
            estimate = self._estimate_answer_recall_for_alphas(
                query_context=query_context,
                alpha_doc=alpha_doc,
                alpha_predicate=candidate,
            )
            if estimate is None:
                continue
            observations.append(
                {
                    "alpha_predicate_proxy": candidate,
                    "target_recall_predicate_proxy": self._clip_target_recall(
                        1.0 - candidate
                    ),
                    **estimate,
                }
            )
        return observations

    def _select_global_answer_recall_observations(
        self,
        plans: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        selectable: List[Tuple[str, List[Dict[str, Any]]]] = []
        selected: Dict[str, Any] = {}
        for query_id, plan in plans.items():
            observations = plan.get("observations") or []
            if not observations:
                continue
            zero_weight = all(int(obs.get("total", 0) or 0) <= 0 for obs in observations)
            if zero_weight:
                selected[query_id] = max(
                    observations,
                    key=lambda obs: float(obs.get("alpha_predicate_proxy", 0.0)),
                )
            else:
                dedup: Dict[int, Dict[str, Any]] = {}
                for obs in observations:
                    covered = int(obs.get("covered", 0) or 0)
                    prior = dedup.get(covered)
                    if prior is None or float(obs["alpha_predicate_proxy"]) < float(
                        prior["alpha_predicate_proxy"]
                    ):
                        dedup[covered] = obs
                selectable.append((query_id, list(dedup.values())))

        total = 0
        for _, observations in selectable:
            total += max(int(obs.get("total", 0) or 0) for obs in observations)
        if total <= 0:
            return selected

        target = float(self.alloc_config.target_recall)
        dp: Dict[int, Tuple[float, Dict[str, Dict[str, Any]]]] = {0: (0.0, {})}
        for query_id, observations in selectable:
            next_dp: Dict[int, Tuple[float, Dict[str, Dict[str, Any]]]] = {}
            for coverage, (alpha_score, chosen) in dp.items():
                for obs in observations:
                    new_coverage = coverage + int(obs.get("covered", 0) or 0)
                    new_alpha_score = alpha_score + float(obs["alpha_predicate_proxy"])
                    new_chosen = dict(chosen)
                    new_chosen[query_id] = obs
                    prior = next_dp.get(new_coverage)
                    if prior is None or new_alpha_score < prior[0] - 1e-12:
                        next_dp[new_coverage] = (new_alpha_score, new_chosen)
            dp = next_dp

        target_coverage = target * total
        over_target_items = [
            item for item in dp.items() if float(item[0]) + 1e-12 >= target_coverage
        ]
        candidates = over_target_items or list(dp.items())
        best_coverage, (best_alpha_score, best_selection) = min(
            candidates,
            key=lambda item: (
                abs(float(item[0]) - target_coverage),
                float(item[1][0]),
            ),
        )
        del best_alpha_score
        selected.update(best_selection)
        selected["__global_training_answer_covered__"] = int(best_coverage)
        selected["__global_training_answer_total__"] = int(total)
        selected["__global_training_answer_recall__"] = (
            float(best_coverage) / float(total) if total else None
        )
        return selected

    def _build_query_context(
        self,
        query_id: str,
        schema_query: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        query_info = self.loader.get_query_info(query_id) if hasattr(self.loader, "get_query_info") else {}
        if not isinstance(query_info, dict):
            query_info = {}
        all_tables = [s.get(SCHEMA_NAME_KEY, "") for s in schema_query if isinstance(s, dict)]
        all_tables = [t for t in all_tables if t]

        table_to_schema = {
            s.get(SCHEMA_NAME_KEY, ""): s for s in schema_query if isinstance(s, dict)
        }
        table_to_schema = {k: v for k, v in table_to_schema.items() if k}

        sql = query_info.get("sql", "")
        query_tables = query_info.get("tables", all_tables)
        if sql:
            predicates_by_table = group_predicates_by_table(
                sql,
                schema_query,
                query_tables=query_tables,
            )
        else:
            predicates_by_table = {t: [] for t in all_tables}
            logging.warning(
                "[DataExtractionAlphaAllocator:_build_query_context] Query %s has no SQL; predicates empty.",
                query_id,
            )

        gt_to_task_table: Dict[str, str] = {}
        if hasattr(self.loader, "load_name_map"):
            name_map = self.loader.load_name_map(query_id)
            if isinstance(name_map, dict):
                table_map = name_map.get("table", {})
                if isinstance(table_map, dict):
                    gt_to_task_table = {
                        str(gt): str(task) for task, gt in table_map.items()
                    }

        train_doc_to_table: Dict[str, str] = {}
        for doc_id in self.train_doc_ids:
            table_name = self._get_task_table_for_doc(
                doc_id=doc_id,
                all_tables=all_tables,
                gt_to_task_table=gt_to_task_table,
            )
            if table_name:
                train_doc_to_table[doc_id] = table_name

        return {
            "query_id": query_id,
            "query_info": query_info,
            "all_tables": all_tables,
            "table_to_schema": table_to_schema,
            "predicates_by_table": predicates_by_table,
            "gt_to_task_table": gt_to_task_table,
            "train_doc_to_table": train_doc_to_table,
        }

    def _evaluate_cost(
        self,
        query_context: Dict[str, Any],
        alpha_doc: float,
        alpha_predicate: float,
    ) -> float:
        key = f"{query_context['query_id']}|{alpha_doc:.12f}|{alpha_predicate:.12f}"
        if key in self._query_cache:
            return float(self._query_cache[key]["cost"])

        train_docs_after_doc_filter = self._apply_doc_filter(
            query_id=query_context["query_id"],
            schema_query=list(query_context["table_to_schema"].values()),
            alpha_doc=alpha_doc,
        )
        if not train_docs_after_doc_filter:
            self._query_cache[key] = {"cost": 0.0}
            return 0.0

        pred_target_recall = self._clip_target_recall(1.0 - alpha_predicate)
        passed_docs = self._estimate_predicate_passed_docs(
            query_context=query_context,
            doc_ids=train_docs_after_doc_filter,
            predicate_target_recall=pred_target_recall,
        )
        cost = float(passed_docs)
        self._query_cache[key] = {"cost": cost}
        return cost

    def _estimate_answer_recall_for_alphas(
        self,
        *,
        query_context: Dict[str, Any],
        alpha_doc: float,
        alpha_predicate: float,
    ) -> Optional[Dict[str, Any]]:
        key = (
            f"answer|{query_context['query_id']}|"
            f"{alpha_doc:.12f}|{alpha_predicate:.12f}"
        )
        if key in self._query_cache:
            cached = self._query_cache[key]
            return dict(cached) if cached.get("executable") else None

        train_docs_after_doc_filter = self._apply_doc_filter(
            query_id=query_context["query_id"],
            schema_query=list(query_context["table_to_schema"].values()),
            alpha_doc=alpha_doc,
        )
        if not train_docs_after_doc_filter:
            result = {"executable": True, "recall": 0.0, "covered": 0, "total": 0}
            self._query_cache[key] = result
            return result

        query_info = query_context.get("query_info", {})
        evaluator = EvalDataExtraction(
            {
                "training_data_count": 0,
                "training_data_split": "prefix",
            },
            data_loader=self.loader,
        )
        required_by_table = evaluator._required_attrs_by_table(
            self.loader,
            query_context["query_id"],
            query_info,
        )
        if not required_by_table:
            self._query_cache[key] = {"executable": False, "reason": "no_required_tables"}
            return None

        result_dict = self._run_proxy_for_answer_calibration(
            query_context=query_context,
            doc_ids=train_docs_after_doc_filter,
            predicate_target_recall=self._clip_target_recall(1.0 - alpha_predicate),
        )
        answer = evaluator._evaluate_answer_recall(
            loader=self.loader,
            result_dict=result_dict,
            query_info=query_info,
            eval_doc_ids=list(self.train_doc_ids),
            required_by_table=required_by_table,
        )
        if not answer.get("executable", False) or answer.get("recall") is None:
            self._query_cache[key] = {
                "executable": False,
                "reason": answer.get("reason", "not_executable"),
            }
            return None

        result = {
            "executable": True,
            "recall": float(answer.get("recall", 0.0)),
            "covered": int(answer.get("covered", 0)),
            "total": int(answer.get("total", 0)),
        }
        self._query_cache[key] = result
        return result

    def _run_proxy_for_answer_calibration(
        self,
        *,
        query_context: Dict[str, Any],
        doc_ids: List[str],
        predicate_target_recall: float,
    ) -> Dict[str, Dict[str, Any]]:
        proxy_pipeline_config = self._build_proxy_pipeline_config(
            query_id=query_context["query_id"],
            predicate_target_recall=predicate_target_recall,
        )
        pipeline = ProxyPipeline(proxy_pipeline_config)
        pipeline._data_loader = self.loader
        pipeline._query_info = query_context.get("query_info", {})
        pipeline._schema = list(query_context["table_to_schema"].values())
        pipeline._oracle = GoldenOracle(self.loader)

        all_tables = query_context["all_tables"]
        gt_to_task_table = query_context["gt_to_task_table"]
        train_doc_to_table = query_context["train_doc_to_table"]
        table_to_schema = query_context["table_to_schema"]
        predicates_by_table = query_context["predicates_by_table"]
        query_text = query_context.get("query_info", {}).get("query", "")

        table_to_doc_ids: Dict[str, List[str]] = {}
        for doc_id in doc_ids:
            table_name = train_doc_to_table.get(doc_id)
            if table_name is None:
                table_name = self._get_task_table_for_doc(
                    doc_id=doc_id,
                    all_tables=all_tables,
                    gt_to_task_table=gt_to_task_table,
                )
            if table_name:
                table_to_doc_ids.setdefault(table_name, []).append(doc_id)

        result_dict: Dict[str, Dict[str, Any]] = {}
        for table_name, table_doc_ids in table_to_doc_ids.items():
            if not table_doc_ids:
                continue
            table_schema = table_to_schema.get(table_name, {})
            predicates = predicates_by_table.get(table_name, [])
            train_ids_for_table = [
                doc_id for doc_id in self.train_doc_ids if train_doc_to_table.get(doc_id) == table_name
            ]
            results = pipeline.run_for_documents(
                doc_ids=table_doc_ids,
                train_doc_ids=train_ids_for_table,
                predicates=predicates,
                table_schema=table_schema,
                query_text=query_text,
                data_loader=self.loader,
                extra_proxies=None,
            )
            for doc_id, extracted in results.extractions.items():
                result_dict[doc_id] = {
                    RESULT_TABLE_KEY: table_name,
                    RESULT_DATA_KEY: extracted if isinstance(extracted, dict) else {},
                }

        return result_dict

    def _apply_doc_filter(
        self,
        query_id: str,
        schema_query: List[Dict[str, Any]],
        alpha_doc: float,
    ) -> List[str]:
        if not bool(self.doc_filter_config.get("enabled", False)):
            return list(self.train_doc_ids)

        cfg = dict(self.doc_filter_config)
        cfg["target_recall"] = self._clip_target_recall(1.0 - alpha_doc)
        doc_filter = create_doc_filter(cfg)

        storage_path = resolve_embedding_storage_path(
            config=cfg,
            loader=self.loader,
        )
        manager_kwargs = embedding_manager_kwargs(
            cfg,
            default_model="text-embedding-3-small",
            fallback_api_key=self.api_key,
        )
        manager_key = (
            str(manager_kwargs["model"]),
            str(manager_kwargs.get("api_key") or ""),
            str(manager_kwargs.get("provider") or ""),
            str(manager_kwargs.get("base_url") or ""),
            str(storage_path or ""),
        )
        if manager_key not in self._embedding_managers:
            self._embedding_managers[manager_key] = EmbeddingManager(
                storage_path=storage_path,
                loader=self.loader,
                **manager_kwargs,
            )
        embedding_manager = self._embedding_managers[manager_key]

        result = doc_filter.filter(
            query_id=query_id,
            doc_ids=list(self.train_doc_ids),
            data_loader=self.loader,
            schema_context=schema_query,
            embedding_manager=embedding_manager,
            enable_calibrate=bool(cfg.get("enable_calibrate", False)),
            train_doc_ids=list(self.train_doc_ids),
        )
        excluded = set(result.excluded_doc_ids)
        return [doc_id for doc_id in self.train_doc_ids if doc_id not in excluded]

    def _estimate_predicate_passed_docs(
        self,
        query_context: Dict[str, Any],
        doc_ids: List[str],
        predicate_target_recall: float,
    ) -> int:
        if not bool(self.proxy_runtime_config.get("enabled", False)):
            return len(doc_ids)

        proxy_pipeline_config = self._build_proxy_pipeline_config(
            query_id=query_context["query_id"],
            predicate_target_recall=predicate_target_recall,
        )
        pipeline = ProxyPipeline(proxy_pipeline_config)
        pipeline._data_loader = self.loader
        pipeline._query_info = query_context.get("query_info", {})
        pipeline._schema = list(query_context["table_to_schema"].values())
        pipeline._oracle = GoldenOracle(self.loader)

        all_tables = query_context["all_tables"]
        gt_to_task_table = query_context["gt_to_task_table"]
        train_doc_to_table = query_context["train_doc_to_table"]
        table_to_schema = query_context["table_to_schema"]
        predicates_by_table = query_context["predicates_by_table"]
        query_text = query_context.get("query_info", {}).get("query", "")

        table_to_doc_ids: Dict[str, List[str]] = {}
        for doc_id in doc_ids:
            table_name = train_doc_to_table.get(doc_id)
            if table_name is None:
                table_name = self._get_task_table_for_doc(
                    doc_id=doc_id,
                    all_tables=all_tables,
                    gt_to_task_table=gt_to_task_table,
                )
            if table_name:
                table_to_doc_ids.setdefault(table_name, []).append(doc_id)

        total_passed = 0
        for table_name, table_doc_ids in table_to_doc_ids.items():
            if not table_doc_ids:
                continue
            table_schema = table_to_schema.get(table_name, {})
            predicates = predicates_by_table.get(table_name, [])
            train_ids_for_table = [
                doc_id for doc_id in self.train_doc_ids if train_doc_to_table.get(doc_id) == table_name
            ]
            results = pipeline.run_for_documents(
                doc_ids=table_doc_ids,
                train_doc_ids=train_ids_for_table,
                predicates=predicates,
                table_schema=table_schema,
                query_text=query_text,
                data_loader=self.loader,
                extra_proxies=None,
            )
            total_passed += int(results.documents_passed_proxies)

        return total_passed

    def _build_proxy_pipeline_config(
        self,
        query_id: str,
        predicate_target_recall: float,
    ) -> ProxyPipelineConfig:
        proxy_cfg = self.proxy_runtime_config
        return ProxyPipelineConfig(
            dataset_path=str(self.data_path),
            query_id=query_id,
            data_main=str(self.extraction_config.get("data_main", "dataset/")),
            llm_mode=proxy_cfg.get("llm_mode", self.extraction_config.get("mode", "gemini")),
            llm_model=proxy_cfg.get("llm_model", self.extraction_config.get("llm_model", "gemini-2.5-flash-lite")),
            api_key=self.api_key,
            embedding_model=proxy_cfg.get("embedding_model", "gemini-embedding-001"),
            embedding_api_key=proxy_cfg.get("embedding_api_key") or self.api_key,
            embedding_provider=proxy_cfg.get("embedding_provider"),
            embedding_base_url=proxy_cfg.get("embedding_base_url"),
            embedding_storage_path=proxy_cfg.get("embedding_storage_path"),
            embedding_cache_dir=proxy_cfg.get("embedding_cache_dir"),
            embedding_cache_file=proxy_cfg.get("embedding_cache_file"),
            embeddings_cache_dir=proxy_cfg.get("embeddings_cache_dir"),
            use_embedding_proxies=resolve_proxy_flag(proxy_cfg, "use_embedding_proxies", True),
            use_learned_proxies=resolve_proxy_flag(proxy_cfg, "use_learned_proxies", True),
            use_finetuned_learned_proxies=resolve_proxy_flag(
                proxy_cfg,
                "use_finetuned_learned_proxies",
                True,
            ),
            predicate_proxy_mode=str(proxy_cfg.get("predicate_proxy_mode", "pretrained")),
            allow_embedding_fallback=resolve_proxy_flag(
                proxy_cfg,
                "allow_embedding_fallback",
                False,
            ),
            training_data_count=len(self.train_doc_ids),
            min_training_data=0,
            min_calibration_data=0,
            proxy_threshold=resolve_proxy_threshold(proxy_cfg, self.extraction_config),
            target_recall=self._clip_target_recall(predicate_target_recall),
            random_seed=int(proxy_cfg.get("random_seed", self.extraction_config.get("random_seed", 42))),
            save_hard_negatives=False,
            verbose=False,
            use_join_resolution=False,
            join_extractor="llm",
            allow_train_test_overlap=True,
            finetuned_model=proxy_cfg.get(
                "finetuned_model", "knowledgator/gliclass-small-v1.0"
            ),
            finetuned_epochs=int(proxy_cfg.get("finetuned_epochs", 3)),
            finetuned_learning_rate=float(proxy_cfg.get("finetuned_learning_rate", 2e-5)),
            use_gliclass_icl=proxy_cfg.get("use_gliclass_icl", False),
            gliclass_icl_examples_per_class=int(
                proxy_cfg.get("gliclass_icl_examples_per_class", 3)
            ),
        )

    def _get_task_table_for_doc(
        self,
        doc_id: str,
        all_tables: List[str],
        gt_to_task_table: Dict[str, str],
    ) -> Optional[str]:
        """Resolve task table for a doc from ground truth metadata."""
        if not hasattr(self.loader, "get_doc_info"):
            return None
        doc_info = self.loader.get_doc_info(doc_id)
        if not doc_info:
            return None

        data_records = doc_info.get("data_records") or []
        if not data_records:
            return None

        gt_table = data_records[0].get("table_name")
        if not gt_table:
            return None

        task_table = gt_to_task_table.get(gt_table)
        if task_table and task_table in all_tables:
            return task_table
        if gt_table in all_tables:
            return str(gt_table)
        return None
