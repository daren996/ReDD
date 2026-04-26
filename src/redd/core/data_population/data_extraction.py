"""
Unified Data Population Implementation.

This module provides a single unified implementation of data population
that supports multiple LLM providers.

Output format: `<out_root>/res_tabular_data_{qid}_{param_str}.json`
Structure:
{
    <doc_id>: {
        "res": <table_name>,
        "data": {<attribute_name>: <value>, ...},
        "reason": "...",  # Optional reasoning
    },
    ...
}
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from redd.proxy.proxy_runtime.config import (
    is_proxy_runtime_enabled,
    normalize_proxy_runtime_config,
)

from ..data_loader import create_data_loader
from ..utils.constants import (
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTES_KEY,
    DEFAULT_MAX_ATTR_RETRIES,
    DEFAULT_MAX_TABLE_RETRIES,
    DOCUMENT_KEY,
    MAX_ATTRIBUTE_VALUE_LENGTH,
    MAX_CONSECUTIVE_UNKNOWN_SCHEMA,
    NULL_VALUE,
    PATH_TEMPLATES,
    RESULT_DATA_KEY,
    RESULT_TABLE_KEY,
    SCHEMA_KEY,
    SCHEMA_NAME_KEY,
    TARGET_ATTRIBUTE_KEY,
)
from ..utils.data_split import resolve_training_data_count, split_doc_ids
from ..utils.output_path import build_task_output_root
from ..utils.progress import emit_progress_event, tqdm
from ..utils.prompt_utils import create_prompt, get_api_key
from ..utils.sql_filter_parser import SQLFilterParser, create_predicate_function
from ..utils.structured_outputs import AttributeExtractionOutput, TableAssignmentOutput
from ..utils.utils import is_none_value
from .base import DataPopulator
from .strategies import (
    AlphaAllocationStrategy,
    DocFilteringStrategy,
    ProxyRuntimeExtractionStrategy,
)

__all__ = ["DataExtraction"]


class DataExtraction(DataPopulator):
    """
    Unified Data Population class supporting multiple LLM providers.

    Supported modes:
    - "openai": OpenAI GPT models
    - "deepseek": DeepSeek models
    - "together": Together AI models
    - "siliconflow": SiliconFlow models
    - "gemini": Google Gemini models
    - "local": Local transformer inference through the shared LLM backend
    - "ground_truth": No LLM calls; read table/attribute values from loader

    The mode is determined by config["mode"].

    TODO:
    - [ ] generate document attribute mappings for partially mapped documents
    """

    SUPPORTED_LLM_MODES = {
        "openai",
        "deepseek",
        "together",
        "siliconflow",
        "gemini",
        "local",
    }
    GROUND_TRUTH_MODES = {"ground_truth", "gt"}

    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize the data populator.

        Args:
            config: Configuration dictionary
            api_key: Optional API key (can also be provided in config or environment)
        """
        # Initialize base class
        super().__init__(config)
        config = self.config
        self.training_data_count = resolve_training_data_count(config)
        self.train_doc_ids: List[str] = []
        self.test_doc_ids: List[str] = []
        self._train_doc_ids_set: Set[str] = set()
        self._test_doc_ids_set: Set[str] = set()

        self.mode = str(config.get("mode", "deepseek")).strip().lower()
        self.disable_llm = bool(
            config.get("disable_llm", False)
            or config.get("use_ground_truth", False)
            or self.mode in self.GROUND_TRUTH_MODES
        )
        if self.mode in self.GROUND_TRUTH_MODES:
            self.mode = "ground_truth"

        # Validate mode
        if (not self.disable_llm) and self.mode not in self.SUPPORTED_LLM_MODES:
            logging.error(
                f"[{self.__class__.__name__}:__init__] Invalid mode: {self.mode}. "
                f"Supported modes: {sorted(self.SUPPORTED_LLM_MODES)}"
            )
            raise ValueError(f"Invalid mode: {self.mode}")

        # Validate required keys
        required_keys = ["res_param_str"]
        if not self.disable_llm:
            required_keys.extend(["llm_model", "prompts"])
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Missing required configuration keys: {missing_keys}")
            raise KeyError(f"Missing required configuration keys: {missing_keys}")

        try:
            self.param_str = config["res_param_str"]
            self.pause_controller = config.get("_pause_controller")

            # Data loader configuration
            self.loader_type = config.get("data_loader_type", "hf_manifest")
            self.loader_config = config.get("data_loader_config", {})

            # API rate limit retry configuration
            self.retry_params = {}
            if "max_retries" in config:
                self.retry_params["max_retries"] = config["max_retries"]
            if "wait_time" in config:
                self.retry_params["wait_time"] = config["wait_time"]
            if self.retry_params:
                logging.info(f"[{self.__class__.__name__}:__init__] Rate limit retry enabled: {self.retry_params}")

            if self.disable_llm:
                self.api_key = None
                self.config["api_key"] = None
                self.prompt_table = None
                self.prompt_attr = None
                # Keep consistent GT behavior for both table assignment and attr extraction.
                self.config.setdefault("use_gt_table_assignment", True)
                self.config.setdefault("use_gt_attr_extraction", True)
            else:
                # Get API key and store it in config
                resolved_api_key = get_api_key(config, self.mode, api_key)
                self.api_key = resolved_api_key
                self.config["api_key"] = resolved_api_key

                # Get prompt paths from config
                prompts_config = config.get("prompts", {})
                if not prompts_config:
                    logging.error(f"[{self.__class__.__name__}:__init__] "
                                 f"prompts config is required (must contain 'prompt_table' and 'prompt_attr')")
                    raise ValueError("prompts config is required (must contain 'prompt_table' and 'prompt_attr')")

                prompt_table_path = prompts_config.get("prompt_table")
                prompt_attr_path = prompts_config.get("prompt_attr")

                if not prompt_table_path or not prompt_attr_path:
                    logging.error(f"[{self.__class__.__name__}:__init__] "
                                 f"prompts.prompt_table and prompts.prompt_attr are required")
                    raise ValueError("prompts.prompt_table and prompts.prompt_attr are required")

                # Provider/backend selection is centralized behind create_prompt.
                self.prompt_table = create_prompt(
                    self.mode,
                    prompt_table_path,
                    llm_model=config.get("llm_model", "deepseek-chat"),
                    api_key=resolved_api_key,
                    config=config,
                )
                self.prompt_attr = create_prompt(
                    self.mode,
                    prompt_attr_path,
                    llm_model=config.get("llm_model", "deepseek-chat"),
                    api_key=resolved_api_key,
                    config=config,
                )

            # Doc filter (optional): filter irrelevant docs before table assignment
            self.proxy_runtime_config = normalize_proxy_runtime_config(config)
            self.use_proxy_runtime = is_proxy_runtime_enabled(config)
            self.doc_filter_strategy = DocFilteringStrategy(config)
            self.doc_filter_config = dict(self.doc_filter_strategy.config)
            self.doc_filter_enabled = self.doc_filter_strategy.enabled
            self.doc_filter_only = self.doc_filter_strategy.only

            logging.info(
                f"[{self.__class__.__name__}:__init__] Initialized with mode: {self.mode}, "
                + f"model: {config.get('llm_model', 'N/A')}, disable_llm={self.disable_llm}"
            )

        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:__init__] Error initializing: {e}")
            traceback.print_exc()
            raise

    def _wait_if_paused(self, stage: str) -> None:
        controller = self.pause_controller
        if controller is None:
            return
        wait_fn = getattr(controller, "wait_if_paused", None)
        if callable(wait_fn):
            wait_fn(stage)

    def __call__(self, dataset_task: Optional[str] = None) -> None:
        """
        Extract tabular data from documents in the specified dataset task.

        Args:
            dataset_task: Dataset/task to process. If None, uses config.
                Example: "wine_1/wine-appellations"
        """
        if dataset_task is None:
            dataset_task = self.config.get("exp_dataset_task")
        if not dataset_task:
            raise ValueError("Missing required config key: exp_dataset_task")

        logging.info(f"[{self.__class__.__name__}:__call__] Processing task: {dataset_task}")
        data_main = self.config.get("data_main", self.config["out_main"])
        doc_dir = Path(data_main) / dataset_task
        out_root = build_task_output_root(self.config, dataset_task, "data_pop")
        out_root.mkdir(parents=True, exist_ok=True)
        logging.info(f"[{self.__class__.__name__}:__call__] Start processing dataset: doc_dir={doc_dir}, out_root={out_root}")
        self._process_dataset(doc_dir, out_root)

    def _process_dataset(self, doc_dir: str | Path, out_root: str | Path):
        """Process all queries and documents in a dataset."""
        self.data_path = Path(doc_dir)
        self.out_root = Path(out_root)

        # Create data loader based on configuration
        self.loader = create_data_loader(
            self.data_path,
            loader_type=self.loader_type,
            loader_config=self.loader_config,
        )
        logging.info(f"[{self.__class__.__name__}:_process_dataset] Created {self.loader.__class__.__name__} for {self.data_path}")

        all_doc_ids = list(self.loader.doc_ids)
        self.train_doc_ids, self.test_doc_ids = split_doc_ids(
            all_doc_ids,
            self.training_data_count,
        )
        self._train_doc_ids_set = set(self.train_doc_ids)
        self._test_doc_ids_set = set(self.test_doc_ids)
        logging.info(
            f"[{self.__class__.__name__}:_process_dataset] Global split: "
            f"training={len(self.train_doc_ids)}, test={len(self.test_doc_ids)}, "
            f"training_data_count={self.training_data_count}, total={len(all_doc_ids)}"
        )

        query_dict = self.loader.load_query_dict()
        self.schema_general = self.loader.load_schema_general()

        if not query_dict:
            logging.warning(f"[{self.__class__.__name__}:_process_dataset] No queries found in dataset {self.data_path}")
            return

        logging.info(f"[{self.__class__.__name__}:_process_dataset] Processing {len(query_dict)} queries in dataset {self.data_path.name}")
        selected_query_ids = self._select_query_ids(query_dict)
        if not selected_query_ids:
            logging.warning(
                f"[{self.__class__.__name__}:_process_dataset] No query selected for "
                f"dataset {self.data_path.name}; skip dataset."
            )
            return
        logging.info(
            f"[{self.__class__.__name__}:_process_dataset] Running "
            f"{len(selected_query_ids)} selected queries: {selected_query_ids}"
        )

        alpha_strategy = AlphaAllocationStrategy(
            config=self.config,
            data_path=self.data_path,
            loader=self.loader,
            api_key=self.api_key,
            train_doc_ids=self.train_doc_ids,
            proxy_runtime_enabled=self.use_proxy_runtime,
        )

        for qid in selected_query_ids:
            self._wait_if_paused(f"query-{qid}")
            schema_query = self.loader.load_schema_query(qid)
            if not schema_query:
                schema_query = self.schema_general
            res_path = self.out_root / PATH_TEMPLATES.data_population_result(qid, self.param_str)
            if bool(self.config.get("force_rerun", False)):
                self._clear_query_outputs(qid=qid, res_path=res_path)
            res_data = self.load_processed_res(res_path)
            res_data = self._drop_training_results(res_data, res_path)
            pgbar_name = f"{self.data_path.name}-{qid}"

            doc_target_recall_override = None
            proxy_target_recall_override = None
            allocation_result = alpha_strategy.allocate_for_query(
                query_id=qid,
                schema_query=schema_query,
            )
            if allocation_result is not None:
                doc_target_recall_override = (
                    allocation_result.target_recall_doc_filtering
                )
                proxy_target_recall_override = (
                    allocation_result.target_recall_predicate_proxy
                )
                logging.info(
                    f"[{self.__class__.__name__}:_process_dataset] Query {qid} alpha allocation: "
                    f"doc_target_recall={doc_target_recall_override:.4f}, "
                    f"predicate_target_recall={proxy_target_recall_override:.4f}, "
                    f"estimated_cost={allocation_result.estimated_cost:.4f}"
                )

            # Phase 0: Doc filtering (optional) - exclude schema-irrelevant docs
            excluded_doc_ids = self._excluded_doc_ids_for_query(
                query_id=qid,
                schema_query=schema_query,
                target_recall_override=doc_target_recall_override,
            )
            self._emit_doc_filter_optimization_update(
                query_id=qid,
                excluded_doc_ids=excluded_doc_ids,
            )
            if excluded_doc_ids:
                logging.info(
                    f"[{self.__class__.__name__}:_process_dataset] Phase 0: Doc filter excluded "
                    f"{len(excluded_doc_ids)} docs for query-{qid}"
                )
                removed_stale = 0
                for doc_id in list(excluded_doc_ids):
                    if doc_id in res_data:
                        res_data.pop(doc_id, None)
                        removed_stale += 1
                if removed_stale:
                    self.save_results(res_path, res_data)
                    logging.info(
                        f"[{self.__class__.__name__}:_process_dataset] Removed "
                        f"{removed_stale} stale filtered docs from {res_path}"
                    )
            if self.doc_filter_only:
                logging.info(
                    f"[{self.__class__.__name__}:_process_dataset] "
                    f"doc_filter.only=True; skip Phase 1/2 for query-{qid}"
                )
                continue

            # Phase 1: Table assignment (save to same file after each doc)
            logging.info(f"[{self.__class__.__name__}:_process_dataset] Phase 1: Table assignment for query-{qid} -> {res_path}")
            self._process_table_assignment(
                schema_query=schema_query,
                res_data=res_data,
                res_path=res_path,
                pgbar_name=pgbar_name,
                excluded_doc_ids=excluded_doc_ids,
                query_id=qid,
                target_doc_ids=self.test_doc_ids,
            )

            # Reload results (may have been updated during phase 1)
            res_data = self.load_processed_res(res_path)

            # Phase 2: proxy runtime or standard attribute extraction
            if self.use_proxy_runtime:
                logging.info(f"[{self.__class__.__name__}:_process_dataset] Phase 2: proxy runtime per table for query-{qid} -> {res_path}")
                self._process_proxy_runtime_per_table(
                    qid=qid,
                    schema_query=schema_query,
                    res_data=res_data,
                    res_path=res_path,
                    predicate_target_recall=proxy_target_recall_override,
                )
            else:
                logging.info(f"[{self.__class__.__name__}:_process_dataset] Phase 2: Attribute extraction for query-{qid} -> {res_path}")
                self._process_attr_extraction(
                    schema_query=schema_query,
                    res_data=res_data,
                    res_path=res_path,
                    pgbar_name=pgbar_name,
                    query_id=qid,
                    target_doc_ids=self.test_doc_ids,
                    excluded_doc_ids=excluded_doc_ids,
                )
            # Data extraction persists raw per-document required-column values.
            # Query predicates/output projection are applied by later SQL execution.

    def _excluded_doc_ids_for_query(
        self,
        *,
        query_id: str,
        schema_query: List[Dict[str, Any]],
        target_recall_override: Optional[float],
    ) -> Set[str]:
        upstream_root = self.config.get("upstream_doc_filter_root")
        if upstream_root:
            reused = self.doc_filter_strategy.reused_excluded_doc_ids_for_query(
                query_id=query_id,
                upstream_root=Path(upstream_root),
                test_doc_ids=self.test_doc_ids,
                out_root=self.out_root,
                param_str=self.param_str,
                save_results_fn=lambda p, d: self.save_results(str(p), d),
            )
            if reused is not None:
                return reused
            logging.warning(
                f"[{self.__class__.__name__}:_excluded_doc_ids_for_query] "
                f"No reusable upstream doc filter artifact for query-{query_id} in {upstream_root}; "
                "falling back to local data_extraction filtering if enabled."
            )

        return self.doc_filter_strategy.excluded_doc_ids_for_query(
            query_id=query_id,
            schema_query=schema_query,
            loader=self.loader,
            test_doc_ids=self.test_doc_ids,
            train_doc_ids=self.train_doc_ids,
            api_key=self.api_key,
            out_root=self.out_root,
            param_str=self.param_str,
            save_results_fn=lambda p, d: self.save_results(str(p), d),
            target_recall_override=target_recall_override,
        )

    def _emit_doc_filter_optimization_update(
        self,
        *,
        query_id: str,
        excluded_doc_ids: Set[str],
    ) -> None:
        if not self.doc_filter_enabled:
            return
        input_docs = len(self.test_doc_ids)
        excluded_docs = len(excluded_doc_ids)
        kept_docs = max(input_docs - excluded_docs, 0)
        reduction = excluded_docs / input_docs if input_docs else None
        preview = sorted(str(doc_id) for doc_id in excluded_doc_ids)[:8]
        message = (
            f"Doc Filter {self.data_path.name} {query_id}: excluded "
            f"{excluded_docs}/{input_docs} docs"
        )
        if reduction is not None:
            message += f", saved {reduction * 100:.1f}%"
        emit_progress_event(
            {
                "type": "optimization_update",
                "step": "doc_filter",
                "message": message,
                "optimization": {
                    "id": "doc_filter",
                    "title": "Chunk / Document Filter",
                    "status": "running",
                    "message": message,
                    "partial": True,
                    "metrics": {
                        "queries": 1,
                        "input_docs": input_docs,
                        "kept_docs": kept_docs,
                        "excluded_docs": excluded_docs,
                        "llm_doc_calls_before": input_docs,
                        "llm_doc_calls_after": kept_docs,
                        "llm_doc_calls_saved": excluded_docs,
                        "llm_doc_call_reduction": reduction,
                    },
                    "details": [
                        {
                            "kind": "doc_filter",
                            "dataset": self.data_path.name,
                            "query_id": query_id,
                            "input_docs": input_docs,
                            "kept_docs": kept_docs,
                            "excluded_docs": excluded_docs,
                            "llm_doc_calls_saved": excluded_docs,
                            "reduction": reduction,
                            "excluded_doc_ids_preview": preview,
                            "excluded_doc_ids_total": excluded_docs,
                        }
                    ],
                },
            }
        )

    def _clear_query_outputs(self, *, qid: str, res_path: Path) -> None:
        """Remove persisted per-query artifacts so this query is recomputed."""
        candidates = [res_path, res_path.parent / f"{res_path.stem}_proxy_decisions.json"]
        for folder in (self.out_root / "doc_filter", self.out_root / "chunk_filter"):
            if folder.exists():
                candidates.extend(folder.glob(f"*{qid}*.json"))

        removed = 0
        seen: set[Path] = set()
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen or not path.exists() or not path.is_file():
                continue
            seen.add(resolved)
            path.unlink()
            removed += 1
        if removed:
            logging.info(
                f"[{self.__class__.__name__}:_clear_query_outputs] "
                f"force_rerun=True removed {removed} artifact(s) for query-{qid}"
            )

    def _drop_training_results(self, res_data: Dict[str, Any], res_path: Path) -> Dict[str, Any]:
        """
        Remove training documents from persisted result payload.

        This prevents historical runs (before split refactor) from leaking
        training outputs into test-only evaluation.
        """
        if not res_data or not self._train_doc_ids_set:
            return res_data

        removed_doc_ids = [doc_id for doc_id in list(res_data.keys()) if doc_id in self._train_doc_ids_set]
        if not removed_doc_ids:
            return res_data

        for doc_id in removed_doc_ids:
            res_data.pop(doc_id, None)
        self.save_results(res_path, res_data)
        logging.info(
            f"[{self.__class__.__name__}:_drop_training_results] "
            f"Removed {len(removed_doc_ids)} training docs from {res_path}"
        )
        return res_data

    def _select_query_ids(self, query_dict: Dict[str, Any]) -> List[str]:
        """
        Select query IDs to run from config.

        Supported config keys:
            - exp_query_id_list: ["Q1", "Q3"]
            - query_id_list: ["Q1", "Q3"] (backward-compatible alias)
            - query_id: "Q1" (single query)
        """
        requested = self.config.get("exp_query_id_list")
        if requested is None:
            requested = self.config.get("query_id_list")
        if requested is None and self.config.get("query_id"):
            requested = [self.config.get("query_id")]

        if requested is None:
            return list(query_dict.keys())

        if isinstance(requested, str):
            requested = [requested]
        elif not isinstance(requested, list):
            logging.warning(
                f"[{self.__class__.__name__}:_select_query_ids] Invalid query selector "
                f"type: {type(requested)}. Use all queries."
            )
            return list(query_dict.keys())

        selected = [qid for qid in requested if qid in query_dict]
        missing = [qid for qid in requested if qid not in query_dict]
        if missing:
            logging.warning(
                f"[{self.__class__.__name__}:_select_query_ids] Queries not found in "
                f"{self.data_path.name}: {missing}"
            )
        return selected

    def _process_table_assignment(
        self,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
        max_table_retries: int = DEFAULT_MAX_TABLE_RETRIES,
        excluded_doc_ids: Optional[set] = None,
        query_id: Optional[str] = None,
        target_doc_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Phase 1: Iterate over documents and assign each document to a table.

        When use_gt_table_assignment is True (evaluation mode), uses ground truth
        from loader.get_doc_info instead of calling the LLM.

        Saves to the same file as attribute extraction.
        Output format: {doc_id: {"res": table_name, "data": {}}, ...}
        """
        all_tables = [s[SCHEMA_NAME_KEY] for s in schema_query] if schema_query else [s[SCHEMA_NAME_KEY] for s in self.schema_general]
        excluded_doc_ids = excluded_doc_ids or set()
        target_doc_ids = list(self.loader.doc_ids) if target_doc_ids is None else target_doc_ids
        use_gt = bool(self.config.get("use_gt_table_assignment", False) or self.disable_llm)

        if excluded_doc_ids:
            logging.info(
                f"[{self.__class__.__name__}:_process_table_assignment] "
                f"Excluding {len(excluded_doc_ids)} documents"
            )

        if use_gt:
            logging.info(
                f"[{self.__class__.__name__}:_process_table_assignment] "
                "Using ground truth table assignment (no LLM calls)"
            )

        total_docs = len(target_doc_ids)
        # Count docs that already have table assignment
        already_processed = sum(
            1 for doc_id in target_doc_ids
            if doc_id in res_data
            and RESULT_TABLE_KEY in res_data[doc_id]
            and (not use_gt or not is_none_value(res_data[doc_id].get(RESULT_TABLE_KEY)))
        )
        progress_bar = tqdm(
            total=total_docs, initial=0,
            desc=f"Table Assignment {pgbar_name} ({already_processed}/{total_docs})"
        )

        # Build GT -> task schema name map when using ground truth
        gt_to_task_table: Dict[str, str] = {}
        if use_gt and hasattr(self.loader, "load_name_map"):
            name_map = self.loader.load_name_map(query_id)
            table_map = name_map.get("table", {})
            # table_map: task_table -> gt_table; we need gt_table -> task_table
            gt_to_task_table = {gt: ts for ts, gt in table_map.items()}

        excluded_count = 0
        unknown_schema_null_count = 0
        max_retries_null_count = 0
        for doc_id in target_doc_ids:
            self._wait_if_paused(f"table-assignment:{doc_id}")
            # Skip if already has table assignment
            if (
                doc_id in res_data
                and RESULT_TABLE_KEY in res_data[doc_id]
                and (not use_gt or not is_none_value(res_data[doc_id].get(RESULT_TABLE_KEY)))
            ):
                progress_bar.update(1)
                continue

            # Skip if document is excluded
            if doc_id in excluded_doc_ids:
                excluded_count += 1
                progress_bar.update(1)
                continue

            if use_gt:
                table_assigned = self._get_gt_table_assignment(
                    doc_id=doc_id,
                    all_tables=all_tables,
                    gt_to_task_table=gt_to_task_table,
                )
            else:
                if hasattr(self.loader, "get_doc_text"):
                    doc_text = self.loader.get_doc_text(doc_id)
                else:
                    doc_text = self.loader.get_doc(doc_id)[0]
                table_assigned, table_failed, skip_reason = self._assign_table_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    all_tables=all_tables,
                    prompt_schema=schema_query,
                    max_retries=max_table_retries,
                    **kwargs,
                )
                if table_failed:
                    if skip_reason == "unknown_schema":
                        unknown_schema_null_count += 1
                    elif skip_reason == "max_retries":
                        max_retries_null_count += 1
                    table_assigned = NULL_VALUE

            table_assigned = NULL_VALUE if is_none_value(table_assigned) else table_assigned
            # Initialize entry with table assignment and empty data
            res_data[doc_id] = {RESULT_TABLE_KEY: table_assigned, RESULT_DATA_KEY: {}}
            self.save_results(res_path, res_data)
            progress_bar.update(1)

        progress_bar.close()
        stats_parts = [f"excluded: {excluded_count}"]
        if unknown_schema_null_count > 0:
            stats_parts.append(f"unknown_schema_null: {unknown_schema_null_count}")
        if max_retries_null_count > 0:
            stats_parts.append(f"max_retries_null: {max_retries_null_count}")
        logging.info(
            f"[{self.__class__.__name__}:_process_table_assignment] Done table assignment -> {res_path} "
            f"({', '.join(stats_parts)})"
        )

    def _process_proxy_runtime_per_table(
        self,
        qid: str,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        predicate_target_recall: Optional[float] = None,
    ) -> None:
        """
        Phase 2 (proxy runtime): run the proxy runtime per table and merge results.

        Groups documents by table assignment, parses golden SQL predicates by table,
        runs the proxy runtime for each table, and merges extractions into res_data.
        """
        extraction_config = self.config
        if predicate_target_recall is not None:
            extraction_config = copy.deepcopy(self.config)
            proxy_cfg = normalize_proxy_runtime_config(extraction_config)
            proxy_cfg["target_recall"] = float(predicate_target_recall)
            extraction_config["proxy_runtime"] = proxy_cfg
            extraction_config["target_recall"] = float(predicate_target_recall)
            logging.info(
                f"[{self.__class__.__name__}:_process_proxy_runtime_per_table] Query {qid} override: "
                f"proxy_runtime.target_recall={predicate_target_recall:.4f}"
            )

        orchestrator = ProxyRuntimeExtractionStrategy(
            extraction_config=extraction_config,
            data_path=self.data_path,
            loader=self.loader,
            api_key=self.api_key,
            train_doc_ids=self.train_doc_ids,
        )
        orchestrator.process_proxy_runtime_per_table(
            qid=qid,
            schema_query=schema_query,
            res_data=res_data,
            res_path=res_path,
            save_results_fn=lambda p, d: self.save_results(str(p), d),
        )

    def _get_gt_table_assignment(
        self,
        doc_id: str,
        all_tables: List[str],
        gt_to_task_table: Dict[str, str],
    ) -> Optional[str]:
        """
        Get table assignment from ground truth (no LLM).

        Returns task schema table name, or None if doc has no GT record or
        GT table doesn't match any task table.
        """
        if not hasattr(self.loader, "get_doc_info"):
            raise RuntimeError(
                "use_gt_table_assignment requires loader with get_doc_info"
            )
        doc_info = self.loader.get_doc_info(doc_id)
        if not doc_info:
            return None
        data_records = doc_info.get("data_records") or []
        if not data_records:
            return None
        gt_table = data_records[0].get("table_name")
        if not gt_table:
            return None
        # Map GT table name to task schema name
        task_table = gt_to_task_table.get(gt_table)
        if task_table and task_table in all_tables:
            return task_table
        # No mapping: try direct match (GT name might equal task name)
        if gt_table in all_tables:
            return gt_table
        return None

    def _assign_table_single_doc(
        self,
        doc_id: str,
        doc_text: str,
        all_tables: List[str],
        prompt_schema: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = DEFAULT_MAX_TABLE_RETRIES,
        max_consecutive_unknown: int = MAX_CONSECUTIVE_UNKNOWN_SCHEMA,
        **kwargs,
    ) -> tuple:
        """
        Assign a single document to a table.

        If the LLM assigns solely to unknown schema (table not in all_tables) more
        than max_consecutive_unknown times in a row, report a failed assignment.
        If it assigns to both unknown and valid schema across retries, accept the
        valid assignment.

        Returns:
            (table_assigned, table_failed, skip_reason): skip_reason is "unknown_schema",
            "max_retries", or None.
        """
        table_attempt = 0
        table_assigned = None
        last_raw_text = ""
        consecutive_unknown_schema = 0
        while True:
            if table_attempt > max_retries:
                if last_raw_text:
                    logging.debug(
                        f"[{self.__class__.__name__}:_assign_table_single_doc] Last raw response (truncated): "
                        f"{last_raw_text[:500]}..."
                    )
                return None, True, "max_retries"
            tbl_input = {
                DOCUMENT_KEY: doc_text,
                SCHEMA_KEY: prompt_schema or self.schema_general,
            }
            tbl_input = json.dumps(tbl_input, ensure_ascii=False)
            try:
                res_tbl = self.prompt_table.complete_model(
                    tbl_input,
                    TableAssignmentOutput,
                    **self.retry_params,
                )
            except Exception as error:
                last_raw_text = str(error)
                consecutive_unknown_schema = 0  # Reset: not an unknown-schema failure
                table_attempt += 1
                continue
            table_assigned = res_tbl.table_assignment
            if table_assigned not in all_tables and not is_none_value(table_assigned):
                consecutive_unknown_schema += 1
                if consecutive_unknown_schema >= max_consecutive_unknown:
                    return None, True, "unknown_schema"
                table_attempt += 1
                continue
            # Valid assignment: accept it (ignore any prior unknown-schema responses)
            break
        return table_assigned, False, None

    def _process_attr_extraction(
        self,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
        max_attr_retries: int = DEFAULT_MAX_ATTR_RETRIES,
        excluded_doc_ids: Optional[set] = None,
        query_id: Optional[str] = None,
        target_doc_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Phase 2: Iterate over documents and extract attribute values based on assigned tables.

        Uses table assignment from Phase 1 (already in res_data).
        Saves to the same file as table assignment after EACH attribute extraction.
        Output format: {doc_id: {"res": table_name, "data": {attr: value, ...}}, ...}
        """
        all_tables = [s[SCHEMA_NAME_KEY] for s in self.schema_general]
        table2schema = {s[SCHEMA_NAME_KEY]: s for s in self.schema_general}
        table2attr: Dict[str, List[str]] = {
            s[SCHEMA_NAME_KEY]: [a[ATTRIBUTE_NAME_KEY] for a in s[ATTRIBUTES_KEY]]
            for s in schema_query
            if s[SCHEMA_NAME_KEY] in all_tables
        }
        use_gt_attr = bool(self.config.get("use_gt_attr_extraction", False) or self.disable_llm)
        task_to_gt_table: Dict[str, str] = {}
        attr_name_map: Dict[str, Dict[str, Any]] = {}
        if use_gt_attr and hasattr(self.loader, "load_name_map"):
            name_map = self.loader.load_name_map(query_id)
            if isinstance(name_map, dict):
                table_map = name_map.get("table", {})
                if isinstance(table_map, dict):
                    task_to_gt_table = dict(table_map)
                attribute_map = name_map.get("attribute", {})
                if isinstance(attribute_map, dict):
                    attr_name_map = dict(attribute_map)
        if use_gt_attr:
            logging.info(
                f"[{self.__class__.__name__}:_process_attr_extraction] "
                "Using ground truth attribute extraction (no LLM calls)"
            )

        excluded_doc_ids = excluded_doc_ids or set()
        target_doc_ids = list(self.loader.doc_ids) if target_doc_ids is None else target_doc_ids

        if excluded_doc_ids:
            logging.info(
                f"[{self.__class__.__name__}:_process_attr_extraction] "
                f"Excluding {len(excluded_doc_ids)} documents"
            )

        # Build pending tasks: list of (doc_id, attr) pairs that need processing
        # Also count already processed attrs for progress tracking
        pending_tasks: List[tuple] = []
        total_attrs = 0
        already_processed_attrs = 0
        excluded_count = 0

        for doc_id in target_doc_ids:
            # Skip if document is excluded
            if doc_id in excluded_doc_ids:
                excluded_count += 1
                continue

            # Skip if no table assignment for this document
            if doc_id not in res_data:
                continue

            table_assigned = res_data[doc_id].get(RESULT_TABLE_KEY)

            # Skip if table is None/NULL
            if is_none_value(table_assigned) or table_assigned == NULL_VALUE:
                continue

            # Get attributes needed for this table
            attrs_needed = table2attr.get(table_assigned, [])
            total_attrs += len(attrs_needed)

            # Check which attrs are already extracted for this doc
            existing_data = res_data[doc_id].get(RESULT_DATA_KEY, {})
            if not isinstance(existing_data, dict):
                existing_data = {}
                res_data[doc_id][RESULT_DATA_KEY] = existing_data

            for attr in attrs_needed:
                if attr in existing_data:
                    already_processed_attrs += 1
                else:
                    pending_tasks.append((doc_id, attr))

        logging.info(f"[{self.__class__.__name__}:_process_attr_extraction] "
                     f"Total attrs: {total_attrs}, already processed: {already_processed_attrs}, "
                     f"pending: {len(pending_tasks)}, excluded docs: {excluded_count}")

        # Create progress bar based on total attrs (show even if no pending tasks)
        progress_bar = tqdm(
            total=total_attrs, initial=already_processed_attrs,
            desc=f"Attr Extraction {pgbar_name}"
        )

        if not pending_tasks:
            logging.info(f"[{self.__class__.__name__}:_process_attr_extraction] "
                         f"No pending attrs to process.")
            progress_bar.close()
            return

        # Process each (doc_id, attr) pair
        for doc_id, attr in pending_tasks:
            self._wait_if_paused(f"attr-extraction:{doc_id}:{attr}")
            table_assigned = res_data[doc_id].get(RESULT_TABLE_KEY)
            if use_gt_attr:
                attr_val = self._get_gt_attribute_value(
                    doc_id=doc_id,
                    task_table=str(table_assigned),
                    task_attribute=attr,
                    task_to_gt_table=task_to_gt_table,
                    attr_name_map=attr_name_map,
                )
            else:
                doc_text = self.loader.get_doc_text(doc_id)
                attr_val = self._extract_attr_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    attr=attr,
                    table_schema=table2schema[table_assigned],
                    max_retries=max_attr_retries,
                    **kwargs,
                )

            attr_val = NULL_VALUE if is_none_value(attr_val) else attr_val
            if isinstance(attr_val, str) and len(attr_val) > MAX_ATTRIBUTE_VALUE_LENGTH:
                logging.info(f"[{self.__class__.__name__}:_process_attr_extraction] Attr too long "
                             f"(>{len(attr_val)}): doc {doc_id} attr {attr}")
            res_data[doc_id][RESULT_DATA_KEY][attr] = attr_val

            # Save after each attr extraction
            self.save_results(res_path, res_data)
            progress_bar.update(1)

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}:_process_attr_extraction] Done attr extraction -> {res_path}")

    def _materialize_query_output(self, query_id: str, res_path: Path) -> None:
        """
        Apply simple query predicates and project extraction results to output columns.

        Data extraction must first collect predicate columns from ``required_columns``.
        The persisted query result, however, should expose the query's
        ``output_columns``. This method handles simple per-document predicates from
        the query SQL and then trims ``data`` to the selected attributes.
        """
        if not hasattr(self.loader, "get_query_info"):
            return

        query_info = self.loader.get_query_info(query_id)
        if not isinstance(query_info, dict):
            return

        output_columns = query_info.get("output_columns") or []
        sql = str(query_info.get("sql") or "")
        if not output_columns and not sql:
            return

        res_data = self.load_processed_res(res_path)
        if not res_data:
            return

        output_attrs = self._column_ids_to_attr_names(output_columns)
        predicate_fns = []
        if sql:
            parser = SQLFilterParser()
            predicate_fns = [
                (predicate.attribute, create_predicate_function(predicate))
                for predicate in parser.parse(sql)
            ]

        changed = False
        for doc_result in res_data.values():
            if not isinstance(doc_result, dict):
                continue
            table_assigned = doc_result.get(RESULT_TABLE_KEY)
            data = doc_result.get(RESULT_DATA_KEY, {})
            if is_none_value(table_assigned) or table_assigned == NULL_VALUE:
                continue
            if not isinstance(data, dict):
                data = {}

            passed = True
            for attr, predicate_fn in predicate_fns:
                if not predicate_fn(data.get(attr)):
                    passed = False
                    break
            if not passed:
                doc_result[RESULT_TABLE_KEY] = NULL_VALUE
                doc_result[RESULT_DATA_KEY] = {}
                changed = True
                continue

            if output_attrs:
                projected = {
                    attr: data[attr]
                    for attr in output_attrs
                    if attr in data and not is_none_value(data.get(attr))
                }
                if projected != data:
                    doc_result[RESULT_DATA_KEY] = projected
                    changed = True

        if changed:
            self.save_results(res_path, res_data)

    @staticmethod
    def _column_ids_to_attr_names(column_ids: List[Any]) -> List[str]:
        attrs: List[str] = []
        for column_id in column_ids:
            text = str(column_id or "")
            if not text:
                continue
            attr = text.split(".", 1)[1] if "." in text else text
            if attr not in attrs:
                attrs.append(attr)
        return attrs

    def _get_gt_attribute_value(
        self,
        doc_id: str,
        task_table: str,
        task_attribute: str,
        task_to_gt_table: Dict[str, str],
        attr_name_map: Dict[str, Dict[str, Any]],
    ) -> Any:
        """
        Get attribute value from ground truth records (no LLM call).

        Attribute names in task schema may be mapped to one or more GT attributes.
        """
        if not hasattr(self.loader, "get_doc_info"):
            raise RuntimeError(
                "use_gt_attr_extraction requires loader with get_doc_info"
            )

        doc_info = self.loader.get_doc_info(doc_id)
        if not doc_info:
            return None

        data_records = doc_info.get("data_records") or []
        if not data_records:
            return None

        gt_table = task_to_gt_table.get(task_table, task_table)
        target_records = [r for r in data_records if r.get("table_name") == gt_table]
        if not target_records and task_table != gt_table:
            target_records = [r for r in data_records if r.get("table_name") == task_table]
        if not target_records:
            return None

        mapped_attrs = attr_name_map.get(gt_table, {}).get(task_attribute, task_attribute)
        if isinstance(mapped_attrs, list):
            merged_values: List[str] = []
            for gt_attr in mapped_attrs:
                for rec in target_records:
                    data = rec.get("data", {})
                    if isinstance(data, dict):
                        value = data.get(gt_attr)
                        if not is_none_value(value):
                            merged_values.append(str(value))
                            break
            if merged_values:
                return " ".join(merged_values)
            return None

        candidate_attrs = [mapped_attrs, task_attribute]
        for candidate_attr in candidate_attrs:
            for rec in target_records:
                data = rec.get("data", {})
                if not isinstance(data, dict):
                    continue
                if candidate_attr in data:
                    value = data.get(candidate_attr)
                    if not is_none_value(value):
                        return value
        return None

    def _extract_attr_single_doc(
        self,
        doc_id: str,
        doc_text: str,
        attr: str,
        table_schema: Dict[str, Any],
        max_retries: int = DEFAULT_MAX_ATTR_RETRIES,
        **kwargs,
    ) -> Any:
        """
        Extract a single attribute from a document.

        Returns:
            The extracted attribute value, or None if failed.
        """
        attr_attempt = 0
        attr_val = None
        while True:
            if attr_attempt > max_retries:
                logging.info(f"[{self.__class__.__name__}:_extract_attr_single_doc] Attr fail "
                             f">{max_retries}x for doc {doc_id} attr {attr}. Skipping attr.")
                break
            attr_input = {
                DOCUMENT_KEY: doc_text,
                SCHEMA_KEY: table_schema,
                TARGET_ATTRIBUTE_KEY: attr,
            }
            attr_input = json.dumps(attr_input, ensure_ascii=False)
            try:
                res_attr = self.prompt_attr.complete_model(
                    attr_input,
                    AttributeExtractionOutput,
                    **self.retry_params,
                ).root
            except Exception:
                attr_attempt += 1
                continue
            if attr not in res_attr:
                attr_attempt += 1
                continue
            attr_val = res_attr[attr]
            break
        return attr_val

    def get_res_schema(self, res_schema_path: str) -> Dict[str, Any]:
        """Load Result Schema from <res_schema_path>."""
        if not os.path.exists(res_schema_path):
            logging.error(f"[{self.__class__.__name__}:get_res_schema] Result Schema not found: {res_schema_path}")
            raise FileNotFoundError(f"Result schema not found: {res_schema_path}")
        return self.load_json(res_schema_path)

    @staticmethod
    def _extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from raw text.

        Args:
            raw_text: Raw text that may contain JSON content

        Returns:
            Parsed JSON object as dictionary, or None if no valid JSON found
        """
        if not raw_text or not raw_text.strip():
            return None

        def _try_parse_json_str(text: str) -> Optional[Dict[str, Any]]:
            """Try to parse JSON using multiple parsers."""
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(text)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
            return None

        # Strategy 1: Look for JSON code blocks
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ```
            r'```\s*(\{.*?\})\s*```',  # ``` {...} ```
        ]

        for pattern in json_patterns:
            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                json_candidate = match.group(1).strip()
                result = _try_parse_json_str(json_candidate)
                if result:
                    return result

        # Strategy 2: Look for JSON-like objects in the text
        # Use non-greedy matching to find the first complete JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, raw_text, re.DOTALL):
            json_candidate = match.group(0)
            result = _try_parse_json_str(json_candidate)
            if result:
                return result

        # Strategy 3: Try parsing the entire text as JSON
        result = _try_parse_json_str(raw_text.strip())
        if result:
            return result

        return None

    def __str__(self) -> str:
        """String representation of the data populator."""
        return f"{self.__class__.__name__} (mode={self.mode}): \n{self.param_str}"
