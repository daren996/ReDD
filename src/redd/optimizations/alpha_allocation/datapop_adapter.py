"""
DataPop adapter for query-level alpha allocation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from redd.embedding import EmbeddingManager
from redd.core.utils.constants import SCHEMA_NAME_KEY
from redd.core.utils.sql_filter_parser import group_predicates_by_table
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
    AlphaAllocationConfig,
    AlphaAllocationResult,
    STAGE_DOC_FILTERING,
    STAGE_PREDICATE_PROXY,
)


class DataPopAlphaAllocator:
    """Adapter that estimates per-query alpha split on training-prefix docs."""

    def __init__(
        self,
        datapop_config: Dict[str, Any],
        data_path: Path,
        loader: Any,
        api_key: Optional[str],
        train_doc_ids: List[str],
    ):
        self.datapop_config = datapop_config
        self.data_path = Path(data_path)
        self.loader = loader
        self.api_key = api_key or datapop_config.get("api_key")
        self.train_doc_ids = list(train_doc_ids)

        self.alloc_config = AlphaAllocationConfig.from_raw(
            datapop_config.get("alpha_allocation")
        )
        self.doc_filter_config = normalize_doc_filter_config(
            datapop_config.get("doc_filter")
        )
        self.proxy_runtime_config = normalize_proxy_runtime_config(datapop_config)
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
                "[DataPopAlphaAllocator:allocate_for_query] No training docs; skip allocation."
            )
            return None

        stages = [STAGE_DOC_FILTERING, STAGE_PREDICATE_PROXY]
        query_context = self._build_query_context(query_id=query_id, schema_query=schema_query)

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

        alpha_doc = float(alpha_map.get(STAGE_DOC_FILTERING, 0.0))
        alpha_predicate = float(alpha_map.get(STAGE_PREDICATE_PROXY, 0.0))
        target_recall_doc = self._clip_target_recall(1.0 - alpha_doc)
        target_recall_predicate = self._clip_target_recall(1.0 - alpha_predicate)
        result = AlphaAllocationResult(
            enabled=True,
            alpha_doc_filtering=alpha_doc,
            alpha_predicate_proxy=alpha_predicate,
            target_recall_doc_filtering=target_recall_doc,
            target_recall_predicate_proxy=target_recall_predicate,
            estimated_cost=float(cost_value),
            target_recall_global=self.alloc_config.target_recall,
            budget_used=float(budget_used),
            budget_total=float(budget_total),
            trace=trace,
        )
        logging.info(
            "[DataPopAlphaAllocator:allocate_for_query] Query %s -> "
            "alpha_doc=%.4f alpha_predicate=%.4f cost=%.4f",
            query_id,
            result.alpha_doc_filtering,
            result.alpha_predicate_proxy,
            result.estimated_cost,
        )
        return result

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
                "[DataPopAlphaAllocator:_build_query_context] Query %s has no SQL; predicates empty.",
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

        emb_model = str(cfg.get("embedding_model", "text-embedding-3-small"))
        emb_api_key = str(cfg.get("embedding_api_key") or self.api_key or "")
        manager_key = (emb_model, emb_api_key)
        if manager_key not in self._embedding_managers:
            self._embedding_managers[manager_key] = EmbeddingManager(
                loader=self.loader,
                model=emb_model,
                api_key=(emb_api_key or None),
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
            data_main=str(self.datapop_config.get("data_main", "dataset/")),
            llm_mode=proxy_cfg.get("llm_mode", self.datapop_config.get("mode", "gemini")),
            llm_model=proxy_cfg.get("llm_model", self.datapop_config.get("llm_model", "gemini-2.5-flash-lite")),
            api_key=self.api_key,
            embedding_model=proxy_cfg.get("embedding_model", "gemini-embedding-001"),
            use_embedding_proxies=resolve_proxy_flag(proxy_cfg, "use_embedding_proxies", True),
            use_learned_proxies=resolve_proxy_flag(proxy_cfg, "use_learned_proxies", True),
            use_finetuned_learned_proxies=resolve_proxy_flag(
                proxy_cfg,
                "use_finetuned_learned_proxies",
                True,
            ),
            training_data_count=len(self.train_doc_ids),
            min_training_data=0,
            min_calibration_data=0,
            proxy_threshold=resolve_proxy_threshold(proxy_cfg, self.datapop_config),
            target_recall=self._clip_target_recall(predicate_target_recall),
            random_seed=int(proxy_cfg.get("random_seed", self.datapop_config.get("random_seed", 42))),
            save_hard_negatives=False,
            verbose=False,
            use_join_resolution=False,
            join_extractor="llm",
            allow_train_test_overlap=True,
            finetuned_model=proxy_cfg.get(
                "finetuned_model", "knowledgator/gliclass-instruct-large-v1.0"
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

        mappings = doc_info.get("mappings") or doc_info.get("data_records")
        if not mappings:
            return None

        gt_table = mappings[0].get("table_name")
        if not gt_table:
            return None

        task_table = gt_to_task_table.get(gt_table)
        if task_table and task_table in all_tables:
            return task_table
        if gt_table in all_tables:
            return str(gt_table)
        return None
