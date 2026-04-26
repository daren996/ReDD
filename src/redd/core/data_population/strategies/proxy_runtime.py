"""Proxy-runtime extraction strategy for unified data extraction."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from redd.proxy.join_resolution import create_join_resolver
from redd.proxy.proxy_runtime.config import (
    normalize_proxy_runtime_config,
    resolve_proxy_flag,
    resolve_proxy_threshold,
)
from redd.proxy.proxy_runtime.oracle import GoldenOracle
from redd.proxy.proxy_runtime.pipeline import ProxyPipeline
from redd.proxy.proxy_runtime.types import ProxyPipelineConfig

from ...utils.constants import (
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTES_KEY,
    NULL_VALUE,
    RESULT_DATA_KEY,
    RESULT_TABLE_KEY,
    SCHEMA_NAME_KEY,
)
from ...utils.data_split import resolve_training_data_count
from ...utils.progress import emit_progress_event
from ...utils.sql_filter_parser import (
    compute_table_processing_order,
    get_join_graph,
    group_predicates_by_table,
    predicates_to_filter_dict,
)
from ...utils.utils import is_none_value

__all__ = ["ProxyRuntimeExtractionStrategy"]


class ProxyRuntimeExtractionStrategy:
    """Orchestrator for proxy-runtime execution after table assignment."""

    def __init__(
        self,
        extraction_config: Dict[str, Any],
        data_path: Path,
        loader: Any,
        api_key: Optional[str] = None,
        train_doc_ids: Optional[List[str]] = None,
    ):
        """
        Initialize the proxy-runtime extraction strategy.
        
        Args:
            extraction_config: data-extraction configuration (mode, llm_model, etc.)
            data_path: Path to dataset directory
            loader: Data loader instance (from data extraction)
            api_key: Optional API key
        """
        self.extraction_config = extraction_config
        self.data_path = Path(data_path)
        self.loader = loader
        self.api_key = api_key or extraction_config.get("api_key")
        self.train_doc_ids = list(train_doc_ids or [])

        proxy_cfg = normalize_proxy_runtime_config(extraction_config)
        if "training_size" in proxy_cfg:
            raise ValueError(
                "proxy_runtime.training_size is deprecated. "
                "Use top-level training_data_count instead."
            )
        self.proxy_runtime_config = ProxyPipelineConfig(
            dataset_path=str(self.data_path),
            query_id="",  # Set per query
            data_main=str(extraction_config.get("data_main", "dataset/")),
            llm_mode=proxy_cfg.get("llm_mode", extraction_config.get("mode", "gemini")),
            llm_model=proxy_cfg.get("llm_model", extraction_config.get("llm_model", "gemini-2.5-flash-lite")),
            api_key=self.api_key,
            embedding_model=proxy_cfg.get("embedding_model", "gemini-embedding-001"),
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
            training_data_count=resolve_training_data_count(extraction_config),
            min_training_data=0,
            min_calibration_data=0,
            proxy_threshold=resolve_proxy_threshold(proxy_cfg, extraction_config),
            target_recall=float(proxy_cfg.get("target_recall", extraction_config.get("target_recall", 0.95))),
            random_seed=int(proxy_cfg.get("random_seed", extraction_config.get("random_seed", 42))),
            save_hard_negatives=proxy_cfg.get("save_hard_negatives", False),
            verbose=proxy_cfg.get("verbose", False),
            use_join_resolution=resolve_proxy_flag(proxy_cfg, "use_join_resolution", True),
            join_extractor=proxy_cfg.get("join_extractor", "llm"),
            allow_train_test_overlap=proxy_cfg.get("allow_train_test_overlap", False),
            finetuned_model=proxy_cfg.get(
                "finetuned_model", "knowledgator/gliclass-small-v1.0"
            ),
            finetuned_epochs=int(proxy_cfg.get("finetuned_epochs", 3)),
            finetuned_learning_rate=float(proxy_cfg.get("finetuned_learning_rate", 2e-5)),
            use_gliclass_icl=proxy_cfg.get("use_gliclass_icl", False),
            gliclass_icl_examples_per_class=int(
                proxy_cfg.get("gliclass_icl_examples_per_class", 3)
            ),
            heuristic_pass_through_attributes=(
                list(proxy_cfg["heuristic_pass_through_attributes"])
                if isinstance(proxy_cfg.get("heuristic_pass_through_attributes"), list)
                else None
            ),
        )

    def process_proxy_runtime_per_table(
        self,
        qid: str,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        save_results_fn: Optional[Callable[[Path, Dict], None]] = None,
    ) -> None:
        """
        Run the proxy runtime per table and merge results into res_data.
        
        Args:
            qid: Query ID
            schema_query: Query-specific schema (list of table schemas)
            res_data: Table assignment results {doc_id: {res: table, data: {}}}
            res_path: Path to save results
            save_results_fn: Optional callback to save (e.g., extractor.save_results)
        """
        # Set query_id for this run
        self.proxy_runtime_config = replace(self.proxy_runtime_config, query_id=qid)
        
        # 1. Group documents by table
        table_to_doc_ids: Dict[str, List[str]] = {}
        all_tables = [s[SCHEMA_NAME_KEY] for s in schema_query]
        
        for doc_id in self.loader.doc_ids:
            if doc_id not in res_data:
                continue
            entry = res_data[doc_id]
            table_assigned = entry.get(RESULT_TABLE_KEY)
            if is_none_value(table_assigned) or table_assigned == NULL_VALUE:
                continue
            if table_assigned not in all_tables:
                continue
            table_to_doc_ids.setdefault(table_assigned, []).append(doc_id)
        
        logging.info(
            f"[{self.__class__.__name__}] Grouped docs by table: "
            f"{ {t: len(docs) for t, docs in table_to_doc_ids.items()} }"
        )
        
        # 2. Load query info and group predicates by table
        query_info = self.loader.get_query_info(qid)
        if not query_info:
            query_info = {}
        sql = query_info.get("sql", "")
        query_tables = query_info.get("tables", all_tables)
        
        if sql:
            predicates_by_table = group_predicates_by_table(
                sql, schema_query, query_tables=query_tables
            )
            logging.info(
                f"[{self.__class__.__name__}] Predicates by table: "
                f"{ {t: [str(p) for p in preds] for t, preds in predicates_by_table.items()} }"
            )
        else:
            predicates_by_table = {t: [] for t in all_tables}
            logging.warning(f"[{self.__class__.__name__}] No golden SQL found; no predicates")
        
        # 3. Build table -> schema mapping
        table_to_schema = {s[SCHEMA_NAME_KEY]: s for s in schema_query}
        gt_to_task_table: Dict[str, str] = {}
        if hasattr(self.loader, "load_name_map"):
            name_map = self.loader.load_name_map(qid)
            if isinstance(name_map, dict):
                table_map = name_map.get("table", {})
                if isinstance(table_map, dict):
                    gt_to_task_table = {gt: task for task, gt in table_map.items()}
        
        # 4. Join-aware processing order
        join_graph = get_join_graph(sql, schema_query, query_tables=query_tables) if sql else None
        table_order = compute_table_processing_order(join_graph, all_tables)
        if join_graph and self.proxy_runtime_config.use_join_resolution:
            logging.info(f"[{self.__class__.__name__}] Join detected: processing order {table_order}")

        # 5. Run the proxy runtime per table. When use_gt_extraction is True, populate with GT data.
        use_gt_extraction = self.extraction_config.get("use_gt_extraction", False)
        if use_gt_extraction:
            self._populate_gt_extraction(
                table_to_doc_ids=table_to_doc_ids,
                table_to_schema=table_to_schema,
                res_data=res_data,
                save_results_fn=save_results_fn,
                res_path=res_path,
                qid=qid,
            )
            logging.info(f"[{self.__class__.__name__}] Done GT extraction for query {qid}")
            return

        # Clear extraction data for all docs before proxy execution.
        all_doc_ids_for_proxy_runtime = set()
        for doc_ids in table_to_doc_ids.values():
            all_doc_ids_for_proxy_runtime.update(doc_ids)
        for doc_id in all_doc_ids_for_proxy_runtime:
            if doc_id in res_data:
                res_data[doc_id][RESULT_DATA_KEY] = {}
        if all_doc_ids_for_proxy_runtime and save_results_fn:
            save_results_fn(res_path, res_data)

        # Accumulator for per-table proxy decisions (for recall analysis)
        all_proxy_decisions: Dict[str, Any] = {}

        # 6. Run proxy execution per table.
        use_oracle = self.extraction_config.get("use_oracle_extraction", False)
        pipeline = ProxyPipeline(self.proxy_runtime_config)
        pipeline._data_loader = self.loader
        pipeline._query_info = query_info
        pipeline._schema = schema_query
        if use_oracle:
            pipeline._oracle = GoldenOracle(self.loader)
            logging.info(
                f"[{self.__class__.__name__}] Oracle mode: using GoldenOracle "
                "(predicate proxy and join resolution run, extraction uses ground truth)"
            )
        
        query_text = query_info.get("query", "")
        
        # Join key values: (table, attr) -> set of extracted values
        join_key_values: Dict[tuple, set] = {}
        
        for table_name in table_order:
            doc_ids = table_to_doc_ids.get(table_name, [])
            if not doc_ids:
                logging.info(f"[{self.__class__.__name__}] Table {table_name}: no documents, skipping")
                continue

            # Proxy fit/calibration uses only global training-prefix docs.
            train_doc_ids_for_table = self._select_train_doc_ids_for_table(
                table_name=table_name,
                res_data=res_data,
                all_tables=all_tables,
                gt_to_task_table=gt_to_task_table,
            )
            
            predicates = predicates_by_table.get(table_name, [])
            table_schema = table_to_schema.get(table_name, {})
            
            # Build join-resolution proxies for child tables.
            extra_proxies: List[Any] = []
            if join_graph and self.proxy_runtime_config.use_join_resolution and table_name in join_graph.child_to_parent:
                for parent_table, attr_parent, attr_child in join_graph.child_to_parent[table_name]:
                    key = (parent_table, attr_parent)
                    allowed_set = join_key_values.get(key, set())
                    if allowed_set:
                        join_resolver = create_join_resolver(
                            attr=attr_child,
                            allowed_set=allowed_set,
                            oracle=pipeline.oracle,
                            schema=table_schema,
                            table_name=table_name,
                        )
                        extra_proxies.append(join_resolver)
                        logging.info(
                            f"[{self.__class__.__name__}] Join resolver for {table_name}.{attr_child}: "
                            f"allowed {len(allowed_set)} values from {parent_table}.{attr_parent}"
                        )
                    else:
                        logging.warning(
                            f"[{self.__class__.__name__}] No join key values from {parent_table}.{attr_parent} "
                            f"for child table {table_name}; skipping join resolution"
                        )
            
            logging.info(
                f"[{self.__class__.__name__}] Processing table {table_name}: "
                f"{len(doc_ids)} docs, {len(predicates)} predicates, {len(extra_proxies)} join resolvers"
            )
            results = pipeline.run_for_documents(
                doc_ids=doc_ids,
                train_doc_ids=train_doc_ids_for_table,
                predicates=predicates,
                table_schema=table_schema,
                query_text=query_text,
                data_loader=self.loader,
                extra_proxies=extra_proxies if extra_proxies else None,
            )

            # Collect per-proxy decisions and compute per-proxy recall
            if results.execution_stats is not None:
                es = results.execution_stats
                safe_stats = {
                    proxy_name: {
                        key: int(value) if hasattr(value, "item") else value
                        for key, value in stat.items()
                    }
                    for proxy_name, stat in es.proxy_stats.items()
                }
                proxy_recalls = self._compute_proxy_recalls(
                    doc_ids=doc_ids,
                    execution_stats=es,
                    predicates=predicates,
                    table_schema=table_schema,
                )
                all_proxy_decisions[table_name] = {
                    "proxy_stats": safe_stats,
                    "proxy_rejected_doc_ids": dict(es.proxy_rejected_doc_ids),
                    "passed_doc_ids": list(es.passed_doc_ids),
                    "all_doc_ids": list(es.all_doc_ids),
                    "proxy_recalls": proxy_recalls,
                }
                self._emit_proxy_optimization_update(
                    qid=qid,
                    table_name=table_name,
                    proxy_decision=all_proxy_decisions[table_name],
                )
                for proxy_name, rec in proxy_recalls.items():
                    logging.info(
                        f"[{self.__class__.__name__}] Proxy {proxy_name} recall: {rec['recall']:.4f} "
                        f"precision: {rec['precision']:.4f} "
                        f"(gt_relevant_passed={rec['gt_relevant_passed']}, "
                        f"total_passed={rec['total_passed']})"
                    )

            # Collect join key values from this table's extractions (for child tables)
            if join_graph and table_name in join_graph.parent_to_children:
                for jc in join_graph.conditions:
                    if jc.table_parent == table_name:
                        key = (jc.table_parent, jc.attr_parent)
                        values = []
                        for doc_id, ext in results.extractions.items():
                            v = ext.get(jc.attr_parent) if isinstance(ext, dict) else None
                            if v is not None and str(v).strip():
                                values.append(v)
                        join_key_values.setdefault(key, set()).update(values)
                        logging.info(
                            f"[{self.__class__.__name__}] Collected {len(values)} join key values for "
                            f"{jc.table_parent}.{jc.attr_parent}"
                        )
            
            # 6. Merge extractions into res_data
            for doc_id, extracted in results.extractions.items():
                if doc_id in res_data:
                    existing = res_data[doc_id].get(RESULT_DATA_KEY, {})
                    if not isinstance(existing, dict):
                        existing = {}
                    existing.update(extracted)
                    res_data[doc_id][RESULT_DATA_KEY] = existing
            
            # Save after each table
            if save_results_fn:
                save_results_fn(res_path, res_data)
        
        # Save proxy decisions sidecar for evaluation recall analysis.
        if all_proxy_decisions:
            proxy_decisions_path = res_path.parent / (
                res_path.stem + "_proxy_decisions.json"
            )
            try:
                import json as _json
                proxy_decisions_path.parent.mkdir(parents=True, exist_ok=True)
                with open(proxy_decisions_path, "w", encoding="utf-8") as _f:
                    _json.dump(all_proxy_decisions, _f, indent=2, ensure_ascii=False)
                logging.info(
                    f"[{self.__class__.__name__}] Saved proxy decisions to {proxy_decisions_path}"
                )
            except Exception as _e:
                logging.warning(
                    f"[{self.__class__.__name__}] Failed to save proxy decisions: {_e}"
                )
            if self.proxy_runtime_config.verbose:
                self._print_proxy_performance_summary(all_proxy_decisions)

        logging.info(f"[{self.__class__.__name__}] Done proxy runtime per table for query {qid}")

    def _emit_proxy_optimization_update(
        self,
        *,
        qid: str,
        table_name: str,
        proxy_decision: Dict[str, Any],
    ) -> None:
        all_doc_ids = {str(doc_id) for doc_id in proxy_decision.get("all_doc_ids", [])}
        passed_doc_ids = {str(doc_id) for doc_id in proxy_decision.get("passed_doc_ids", [])}
        rejected_doc_ids = sorted(all_doc_ids - passed_doc_ids)
        before = len(all_doc_ids)
        after = len(passed_doc_ids)
        saved = max(before - after, 0)
        pass_rate = after / before if before else None
        proxy_stats = proxy_decision.get("proxy_stats")
        proxy_count = len(proxy_stats) if isinstance(proxy_stats, dict) else 0
        message = (
            f"Proxy {table_name} {qid}: passed {after}/{before} docs, "
            f"saved {saved} LLM-doc calls"
        )
        emit_progress_event(
            {
                "type": "optimization_update",
                "step": "proxy_runtime",
                "message": message,
                "optimization": {
                    "id": "proxy_runtime",
                    "title": "Proxy Runtime",
                    "status": "running",
                    "message": message,
                    "partial": True,
                    "metrics": {
                        "tables": 1,
                        "evaluated": before,
                        "passed": after,
                        "rejected": saved,
                        "llm_doc_calls_before": before,
                        "llm_doc_calls_after": after,
                        "llm_doc_calls_saved": saved,
                        "llm_doc_call_reduction": saved / before if before else None,
                        "pass_rate": pass_rate,
                    },
                    "details": [
                        {
                            "kind": "proxy_runtime",
                            "dataset": self.data_path.name,
                            "query_id": qid,
                            "table": table_name,
                            "proxy_count": proxy_count,
                            "evaluated": before,
                            "passed": after,
                            "rejected": saved,
                            "llm_doc_calls_before": before,
                            "llm_doc_calls_after": after,
                            "llm_doc_calls_saved": saved,
                            "pass_rate": pass_rate,
                            "rejected_doc_ids_preview": rejected_doc_ids[:8],
                            "rejected_doc_ids_total": len(rejected_doc_ids),
                        }
                    ],
                },
            }
        )

    def _print_proxy_performance_summary(
        self, all_proxy_decisions: Dict[str, Any]
    ) -> None:
        """Print per-proxy recall summary at the end of proxy execution."""
        if not all_proxy_decisions:
            return
        width = 78
        print("\n" + "=" * width)
        print("Proxy Performance Summary")
        print("=" * width)
        print(
            "  Recall = GT-relevant passed / GT-relevant evaluated. "
            "Precision = GT-relevant passed / total passed."
        )
        print(
            "  GT-relevant = docs whose GT satisfies query predicates. "
            "Low precision = false positives (irrelevant docs passed)."
        )
        print("=" * width)
        for table_name, decisions in all_proxy_decisions.items():
            recalls = decisions.get("proxy_recalls", {})
            if not recalls:
                continue
            all_doc_ids = decisions.get("all_doc_ids", [])
            passed_doc_ids = decisions.get("passed_doc_ids", [])
            print(f"\n  Table: {table_name}")
            print(f"  Docs evaluated: {len(all_doc_ids)} -> passed: {len(passed_doc_ids)}")
            print("  " + "-" * (width - 2))
            print(
                f"  {'Proxy':<36} {'Recall':>8} {'Prec':>8} {'GT pass':>8} {'GT tot':>8} {'Passed':>8}"
            )
            print("  " + "-" * (width - 2))
            for gname, rec in recalls.items():
                r = rec.get("recall", 0.0)
                p = rec.get("precision", 0.0)
                gt_pass = rec.get("gt_relevant_passed", 0)
                gt_tot = rec.get("gt_relevant_total", 0)
                total_pass = rec.get("total_passed", 0)
                print(
                    f"  {gname:<36} {r:>8.4f} {p:>8.4f} {gt_pass:>8} {gt_tot:>8} {total_pass:>8}"
                )
        print("=" * width + "\n")

    def _compute_proxy_recalls(
        self,
        doc_ids: List[str],
        execution_stats: Any,
        predicates: List[Any],
        table_schema: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute per-proxy recall: of GT-relevant docs that each proxy evaluated,
        what fraction did it pass?

        GT-relevant = docs whose ground-truth extraction satisfies all predicates.
        """
        if not hasattr(self.loader, "get_doc_info"):
            return {}
        oracle = GoldenOracle(self.loader)
        predicate_fns = predicates_to_filter_dict(predicates)

        # Get attributes from schema
        attributes = []
        for a in table_schema.get(ATTRIBUTES_KEY, []):
            name = a.get(ATTRIBUTE_NAME_KEY) if isinstance(a, dict) else str(a) if a else None
            if name:
                attributes.append(name)
        if not attributes:
            attributes = [p.attribute for p in predicates]
        schema_dict = {"Schema Name": table_schema.get(SCHEMA_NAME_KEY, ""), "Attributes": table_schema.get(ATTRIBUTES_KEY, [])}

        # Build GT-relevant set
        gt_relevant: set = set()
        for doc_id in doc_ids:
            try:
                doc_text = self.loader.get_doc_text(doc_id) if hasattr(self.loader, "get_doc_text") else ""
                if not doc_text and hasattr(self.loader, "get_doc"):
                    tup = self.loader.get_doc(doc_id)
                    doc_text = tup[0] if tup else ""
                ext = oracle.extract(document=doc_text, schema=schema_dict, attributes=attributes, doc_id=doc_id)
                all_passed, _ = oracle.check_predicates(extracted_values=ext, predicates=predicate_fns)
                if all_passed:
                    gt_relevant.add(doc_id)
            except Exception:
                pass

        proxy_stats = getattr(execution_stats, "proxy_stats", {}) or {}
        proxy_rejected = getattr(execution_stats, "proxy_rejected_doc_ids", {}) or {}
        all_doc_ids = list(getattr(execution_stats, "all_doc_ids", []) or doc_ids)

        reached = set(all_doc_ids)
        recalls: Dict[str, Dict[str, Any]] = {}
        for proxy_name in proxy_stats:
            rejected_by_proxy = set(proxy_rejected.get(proxy_name, []))
            passed_by_proxy = reached - rejected_by_proxy
            gt_relevant_evaluated = reached & gt_relevant
            gt_relevant_passed = gt_relevant_evaluated - rejected_by_proxy
            n_gt_total = len(gt_relevant_evaluated)
            n_gt_passed = len(gt_relevant_passed)
            n_total_passed = len(passed_by_proxy)
            recall = n_gt_passed / n_gt_total if n_gt_total > 0 else None
            precision = n_gt_passed / n_total_passed if n_total_passed > 0 else None
            recalls[proxy_name] = {
                "recall": recall if recall is not None else 0.0,
                "precision": precision if precision is not None else 0.0,
                "gt_relevant_passed": n_gt_passed,
                "gt_relevant_total": n_gt_total,
                "total_passed": n_total_passed,
            }
            reached = reached - rejected_by_proxy
        return recalls

    def _select_train_doc_ids_for_table(
        self,
        table_name: str,
        res_data: Dict[str, Any],
        all_tables: List[str],
        gt_to_task_table: Dict[str, str],
    ) -> List[str]:
        """
        Select global training docs assigned to this table (for proxy training).

        Training docs are not in res_data (dropped before proxy execution). Use GT table
        from get_doc_info for train docs.
        """
        if not self.train_doc_ids:
            return []

        selected: List[str] = []
        for doc_id in self.train_doc_ids:
            if doc_id in res_data:
                table_assigned = res_data[doc_id].get(RESULT_TABLE_KEY)
            else:
                table_assigned = self._get_gt_table_for_doc(
                    doc_id, all_tables, gt_to_task_table
                )
            if is_none_value(table_assigned) or table_assigned == NULL_VALUE:
                continue
            if table_assigned != table_name:
                continue
            selected.append(doc_id)

        logging.info(
            f"[{self.__class__.__name__}] Training docs for table {table_name}: {len(selected)} "
            f"(from table assignment / GT)"
        )
        return selected

    def _get_gt_table_for_doc(
        self,
        doc_id: str,
        all_tables: List[str],
        gt_to_task_table: Dict[str, str],
    ) -> Optional[str]:
        """Get task table for doc from GT (get_doc_info). Used for train docs not in res_data."""
        if not hasattr(self.loader, "get_doc_info"):
            return None
        doc_info = self.loader.get_doc_info(doc_id)
        if not doc_info:
            return None
        data_records = doc_info.get("data_records") or []
        if not data_records:
            return None
        gt_table = data_records[0].get("table_name") if data_records else None
        if not gt_table:
            return None
        task_table = gt_to_task_table.get(gt_table)
        if task_table and task_table in all_tables:
            return task_table
        if gt_table in all_tables:
            return gt_table
        return None

    def _populate_gt_extraction(
        self,
        table_to_doc_ids: Dict[str, List[str]],
        table_to_schema: Dict[str, Dict],
        res_data: Dict[str, Any],
        save_results_fn: Optional[Callable[[Path, Dict], None]],
        res_path: Path,
        qid: str,
    ) -> None:
        """
        Populate res_data with ground truth tuple values for each document.
        Used when use_gt_extraction=True (no LLM extraction).
        """
        if not hasattr(self.loader, "load_name_map") or not hasattr(self.loader, "get_doc_info"):
            raise RuntimeError(
                "use_gt_extraction requires loader with load_name_map and get_doc_info"
            )
        name_map = self.loader.load_name_map(query_id=qid)
        table_map = name_map.get("table", {})  # task_table -> gt_table
        attr_map = name_map.get("attribute", {})  # gt_table -> {ts_attr -> gt_attr}

        for table_name, doc_ids in table_to_doc_ids.items():
            if not doc_ids:
                continue
            gt_table = table_map.get(table_name, table_name)
            table_schema = table_to_schema.get(table_name, {})
            schema_attrs = []
            for a in table_schema.get(ATTRIBUTES_KEY, []):
                if isinstance(a, dict):
                    name = a.get(ATTRIBUTE_NAME_KEY)
                else:
                    name = str(a) if a else None
                if name:
                    schema_attrs.append(name)
            ts_to_gt = attr_map.get(gt_table, {})

            for doc_id in doc_ids:
                if doc_id not in res_data:
                    continue
                doc_info = self.loader.get_doc_info(doc_id)
                if not doc_info:
                    continue
                data_records = doc_info.get("data_records") or []
                gt_data = {}
                for rec in data_records:
                    if rec.get("table_name") == gt_table:
                        gt_data = rec.get("data") or {}
                        break
                if not isinstance(gt_data, dict):
                    continue

                extracted = {}
                for ts_attr in schema_attrs:
                    gt_attr = ts_to_gt.get(ts_attr, ts_attr)
                    if isinstance(gt_attr, list):
                        vals = [gt_data.get(a) for a in gt_attr if a in gt_data]
                        extracted[ts_attr] = " | ".join(str(v) for v in vals if v is not None) if vals else None
                    else:
                        extracted[ts_attr] = gt_data.get(gt_attr)

                existing = res_data[doc_id].get(RESULT_DATA_KEY, {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(extracted)
                res_data[doc_id][RESULT_DATA_KEY] = existing

            if save_results_fn:
                save_results_fn(res_path, res_data)

        logging.info(
            f"[{self.__class__.__name__}] Populated GT extraction for "
            f"{sum(len(d) for d in table_to_doc_ids.values())} documents"
        )
