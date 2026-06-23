"""
Unified data extraction implementation.

This module provides a single unified implementation of data extraction
that supports multiple LLM providers.

Output format: `<out_root>/res_tabular_data_{qid}_{param_str}.json`
Structure:
{
    <doc_id>: {
        "res": <table_name>,
        "data": {<attribute_name>: <value>, ...},
        "records": [
            {"table": <table_name>, "record_id": "...", "data": {...}},
            ...
        ],  # Optional; present when one document yields multiple rows.
        "reason": "...",  # Optional reasoning
    },
    ...
}
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from redd.proxy.proxy_runtime.config import (
    is_proxy_runtime_enabled,
    normalize_proxy_runtime_config,
)

from ..data_loader import create_data_loader
from ..utils.constants import (
    ATTRIBUTE_DESC_KEY,
    ATTRIBUTE_NAME_KEY,
    ATTRIBUTES_KEY,
    DATA_EXTRACTED_KEY,
    DEFAULT_MAX_ATTR_RETRIES,
    DEFAULT_MAX_TABLE_RETRIES,
    DOCUMENT_KEY,
    MAX_ATTRIBUTE_VALUE_LENGTH,
    MAX_CONSECUTIVE_UNKNOWN_SCHEMA,
    NULL_VALUE,
    PATH_TEMPLATES,
    RESULT_DATA_KEY,
    RESULT_RECORD_ID_KEY,
    RESULT_TABLE_KEY,
    SCHEMA_KEY,
    SCHEMA_NAME_KEY,
    TABLE_ASSIGNMENT_KEY,
    TARGET_ATTRIBUTE_KEY,
)
from ..utils.data_split import (
    resolve_training_data_count,
    resolve_training_data_split,
    resolve_training_data_split_seed,
    split_doc_ids,
)
from ..utils.extraction_records import (
    active_result_records,
    make_legacy_result_entry,
    make_result_entry,
    make_result_record,
    update_result_record_data,
)
from ..utils.output_path import build_task_output_root
from ..utils.progress import emit_progress_event, tqdm
from ..utils.prompt_utils import create_prompt, get_api_key
from ..utils.sql_filter_parser import SQLFilterParser, create_predicate_function
from ..utils.structured_outputs import (
    AttributeExtractionOutput,
    FullDocumentExtractionBatchOutput,
    FullDocumentExtractionOutput,
    TableAssignmentOutput,
)
from ..utils.utils import is_none_value
from .base import DataExtractor

__all__ = ["DataExtraction"]


class DataExtraction(DataExtractor):
    """
    Unified data extraction class supporting multiple LLM providers.

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
        Initialize the data extractor.

        Args:
            config: Configuration dictionary
            api_key: Optional API key (can also be provided in config or environment)
        """
        # Initialize base class
        super().__init__(config)
        config = self.config
        self.multi_record_extraction = bool(
            config.get("multi_record_extraction", False)
            or config.get("one_to_many_extraction", False)
            or config.get("multi_record_output", False)
        )
        self.config["multi_record_extraction"] = self.multi_record_extraction
        self.training_data_count = resolve_training_data_count(config)
        self.training_data_split = resolve_training_data_split(config)
        self.training_data_split_seed = resolve_training_data_split_seed(config)
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
            self.doc_filter_strategy = None
            self.doc_filter_config = {}
            self.doc_filter_enabled = False
            self.doc_filter_only = False
            doc_filter_config = config.get("doc_filter")
            doc_filter_requested = False
            if isinstance(doc_filter_config, dict) and doc_filter_config:
                raw_enabled = doc_filter_config.get("enabled")
                enabled = True if raw_enabled is None else bool(raw_enabled)
                doc_filter_requested = enabled or bool(doc_filter_config.get("only", False))
            should_load_doc_filter = bool(config.get("upstream_doc_filter_root")) or doc_filter_requested
            if should_load_doc_filter:
                from .strategies.doc_filtering import DocFilteringStrategy

                self.doc_filter_strategy = DocFilteringStrategy(config)
                self.doc_filter_config = dict(self.doc_filter_strategy.config)
                self.doc_filter_enabled = self.doc_filter_strategy.enabled
                self.doc_filter_only = self.doc_filter_strategy.only
            self.result_save_interval = max(
                1,
                int(config.get("result_save_interval", 1) or 1),
            )
            self.materialized_full_extraction_only = bool(
                config.get("materialized_full_extraction_only", False)
            )
            self.materialized_full_extraction_enabled = bool(
                config.get("materialized_full_extraction", False)
                or self.materialized_full_extraction_only
            )
            self.materialized_full_extraction_batch_size = max(
                1,
                int(config.get("materialized_full_extraction_batch_size", 16) or 16),
            )
            self.materialized_full_extraction_batch_max_chars = max(
                1,
                int(config.get("materialized_full_extraction_batch_max_chars", 24000) or 24000),
            )
            self.materialized_full_extraction_concurrency = max(
                1,
                int(config.get("materialized_full_extraction_concurrency", 1) or 1),
            )
            self._materialized_full_extraction_data: Dict[str, Any] | None = None
            self._materialized_full_extraction_path: Path | None = None
            self._materialized_full_extraction_lookup_source: Dict[str, Any] | None = None
            self._materialized_full_extraction_lookup_index: Dict[str, Any] | None = None
            table_cache_config = config.get("table_assignment_cache", {})
            self.table_assignment_cache_general_schema = bool(
                isinstance(table_cache_config, dict)
                and table_cache_config.get("general_schema", False)
            )
            self.table_assignment_source_table_metadata = bool(
                isinstance(table_cache_config, dict)
                and table_cache_config.get("source_table_metadata", False)
            )
            self.table_assignment_cache_enabled = bool(
                config.get("enable_table_assignment_cache", False)
                or (
                    isinstance(table_cache_config, dict)
                    and table_cache_config.get("enabled", False)
                )
            )

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

    def _save_results_progress(
        self,
        res_path: Path | str,
        res_data: Dict[str, Any],
        update_count: int,
        *,
        force: bool = False,
    ) -> None:
        save_interval = max(1, int(getattr(self, "result_save_interval", 1) or 1))
        if (
            force
            or save_interval <= 1
            or update_count % save_interval == 0
        ):
            self.save_results(res_path, res_data)

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
        out_root = build_task_output_root(self.config, dataset_task, "data_extraction")
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
            strategy=self.training_data_split,
            seed=self.training_data_split_seed,
        )
        self._proxy_extraction_cache: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        self._table_assignment_cache: Dict[str, str] = {}
        self._table_assignment_null_cache: set[tuple[tuple[str, ...], str]] = set()
        self._table_assignment_cache_events: List[Dict[str, Any]] = []
        self._train_doc_ids_set = set(self.train_doc_ids)
        self._test_doc_ids_set = set(self.test_doc_ids)
        logging.info(
            f"[{self.__class__.__name__}:_process_dataset] Global split: "
            f"training={len(self.train_doc_ids)}, test={len(self.test_doc_ids)}, "
            f"training_data_count={self.training_data_count}, "
            f"training_data_split={self.training_data_split}, "
            f"total={len(all_doc_ids)}"
        )

        query_dict = self.loader.load_query_dict()
        self.schema_general = self.loader.load_schema_general()

        if self.materialized_full_extraction_enabled:
            materialized_full_data = self._ensure_materialized_full_extraction()
            self._materialized_full_extraction_data = materialized_full_data
            if self.materialized_full_extraction_only:
                logging.info(
                    f"[{self.__class__.__name__}:_process_dataset] "
                    "materialized_full_extraction_only=True; skip per-query execution."
                )
                return

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

        schema_by_query: Dict[str, List[Dict[str, Any]]] = {}
        for qid in selected_query_ids:
            schema_query = self.loader.load_schema_query(qid)
            if not schema_query:
                schema_query = self.schema_general
            schema_by_query[qid] = schema_query

        alpha_strategy = None
        allocation_results_by_query = {}
        alpha_config = self.config.get("alpha_allocation", {})
        if isinstance(alpha_config, dict) and alpha_config.get("enabled", False):
            from .strategies.alpha_allocation import AlphaAllocationStrategy

            alpha_strategy = AlphaAllocationStrategy(
                config=self.config,
                data_path=self.data_path,
                loader=self.loader,
                api_key=self.api_key,
                train_doc_ids=self.train_doc_ids,
                proxy_runtime_enabled=self.use_proxy_runtime,
            )
            allocation_results_by_query = alpha_strategy.allocate_for_queries(
                query_schemas=schema_by_query,
            )

        for qid in selected_query_ids:
            self._wait_if_paused(f"query-{qid}")
            schema_query = schema_by_query[qid]
            res_path = self.out_root / PATH_TEMPLATES.data_extraction_result(qid, self.param_str)
            if bool(self.config.get("force_rerun", False)):
                self._clear_query_outputs(qid=qid, res_path=res_path)
            res_data = self.load_processed_res(res_path)
            res_data = self._drop_training_results(res_data, res_path)
            pgbar_name = f"{self.data_path.name}-{qid}"

            doc_target_recall_override = None
            proxy_target_recall_override = None
            allocation_result = allocation_results_by_query.get(qid)
            if allocation_result is None and alpha_strategy is not None:
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
                allocation_path = (
                    self.out_root / f"alpha_allocation_{qid}_{self.param_str}.json"
                )
                allocation_path.write_text(
                    json.dumps(allocation_result.to_dict(), indent=2),
                    encoding="utf-8",
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
            if self.materialized_full_extraction_enabled:
                self._process_materialized_table_assignment(
                    schema_query=schema_query,
                    res_data=res_data,
                    res_path=res_path,
                    pgbar_name=pgbar_name,
                    excluded_doc_ids=excluded_doc_ids,
                    target_doc_ids=self.test_doc_ids,
                )
            else:
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
        if upstream_root and self.doc_filter_strategy is not None:
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

        if self.doc_filter_strategy is None:
            return set()

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
                    "title": "Document Filter",
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
        folder = self.out_root / "doc_filter"
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

    def _ensure_materialized_full_extraction(self) -> Dict[str, Any]:
        """
        Ensure the query-independent full extraction artifact exists.

        This is an experiment-run mode, not an algorithmic cache: the artifact is
        materialized before per-query execution and then used as the offline
        oracle for query runs.
        """
        res_path = self.out_root / PATH_TEMPLATES.materialized_full_extraction(self.param_str)
        meta_path = self.out_root / PATH_TEMPLATES.materialized_full_extraction_meta(self.param_str)
        self._materialized_full_extraction_path = res_path
        self.config["materialized_full_extraction_path"] = str(res_path)

        if bool(self.config.get("force_rerun", False)):
            for path in (res_path, meta_path):
                if path.exists() and path.is_file():
                    path.unlink()

        metadata = self._materialized_full_extraction_metadata()
        if res_path.exists() and meta_path.exists():
            try:
                existing_meta = self.load_json(str(meta_path))
            except json.JSONDecodeError:
                existing_meta = {}
            if not self._materialized_full_extraction_metadata_matches(existing_meta, metadata):
                logging.warning(
                    f"[{self.__class__.__name__}:_ensure_materialized_full_extraction] "
                    f"Existing materialized full extraction metadata does not match current "
                    f"config/schema; recomputing {res_path}"
                )
                res_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)

        target_doc_ids = list(self.loader.doc_ids)
        res_data = self.load_processed_res(res_path)
        if self._materialized_full_extraction_complete(res_data, target_doc_ids):
            logging.info(
                f"[{self.__class__.__name__}:_ensure_materialized_full_extraction] "
                f"Using existing materialized full extraction: {res_path}"
            )
            postprocessed_count = self._postprocess_materialized_full_extraction(
                schema_query=self.schema_general,
                res_data=res_data,
                target_doc_ids=target_doc_ids,
            )
            if postprocessed_count:
                self.save_results(str(res_path), res_data)
                logging.info(
                    f"[{self.__class__.__name__}:_ensure_materialized_full_extraction] "
                    f"Postprocessed {postprocessed_count} materialized full extraction cells."
                )
            self._write_materialized_full_extraction_metadata(
                meta_path=meta_path,
                metadata=metadata,
                res_data=res_data,
                target_doc_ids=target_doc_ids,
            )
            return res_data

        logging.info(
            f"[{self.__class__.__name__}:_ensure_materialized_full_extraction] "
            f"Materializing query-independent full extraction for {len(target_doc_ids)} docs -> {res_path}"
        )

        previous_materialized_data = self._materialized_full_extraction_data
        self._materialized_full_extraction_data = None
        try:
            self._process_materialized_full_doc_extraction(
                schema_query=self.schema_general,
                res_data=res_data,
                res_path=res_path,
                pgbar_name=f"{self.data_path.name}-materialized-full",
                query_id=None,
                target_doc_ids=target_doc_ids,
            )
            res_data = self.load_processed_res(res_path)
        finally:
            self._materialized_full_extraction_data = previous_materialized_data

        postprocessed_count = self._postprocess_materialized_full_extraction(
            schema_query=self.schema_general,
            res_data=res_data,
            target_doc_ids=target_doc_ids,
        )
        if postprocessed_count:
            self.save_results(str(res_path), res_data)
            logging.info(
                f"[{self.__class__.__name__}:_ensure_materialized_full_extraction] "
                f"Postprocessed {postprocessed_count} materialized full extraction cells."
            )

        self._write_materialized_full_extraction_metadata(
            meta_path=meta_path,
            metadata=metadata,
            res_data=res_data,
            target_doc_ids=target_doc_ids,
        )
        return res_data

    def _materialized_full_extraction_metadata(self) -> Dict[str, Any]:
        schema_hash = hashlib.sha256(
            json.dumps(self.schema_general, sort_keys=True, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        runtime_payload = {
            "mode": self.mode,
            "llm_model": self.config.get("llm_model"),
            "disable_llm": self.disable_llm,
            "multi_record_extraction": self.multi_record_extraction,
            "prompts": self.config.get("prompts"),
            "structured_backend": self.config.get("structured_backend"),
            "schema_hash": schema_hash,
        }
        runtime_hash = hashlib.sha256(
            json.dumps(runtime_payload, sort_keys=True, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        return {
            "artifact_type": "materialized_full_extraction",
            "schema_hash": schema_hash,
            "runtime_hash": runtime_hash,
            "mode": self.mode,
            "llm_model": self.config.get("llm_model"),
            "disable_llm": self.disable_llm,
            "multi_record_extraction": self.multi_record_extraction,
        }

    @staticmethod
    def _materialized_full_extraction_metadata_matches(
        existing: Dict[str, Any],
        current: Dict[str, Any],
    ) -> bool:
        if not existing:
            return True
        return (
            existing.get("artifact_type") == current.get("artifact_type")
            and existing.get("schema_hash") == current.get("schema_hash")
            and existing.get("runtime_hash") == current.get("runtime_hash")
        )

    def _write_materialized_full_extraction_metadata(
        self,
        *,
        meta_path: Path,
        metadata: Dict[str, Any],
        res_data: Dict[str, Any],
        target_doc_ids: List[str],
    ) -> None:
        assigned_doc_count = 0
        record_count = 0
        attr_count = 0
        for doc_id in target_doc_ids:
            entry = res_data.get(doc_id)
            if not isinstance(entry, dict):
                continue
            records = active_result_records(entry)
            if not records:
                continue
            assigned_doc_count += 1
            record_count += len(records)
            for record in records:
                data = record.get(RESULT_DATA_KEY, {})
                if isinstance(data, dict):
                    attr_count += len(data)
        self.save_results(
            str(meta_path),
            {
                **metadata,
                "doc_count": len(target_doc_ids),
                "assigned_doc_count": assigned_doc_count,
                "assigned_record_count": record_count,
                "materialized_attr_count": attr_count,
            },
        )

    def _postprocess_materialized_full_extraction(
        self,
        *,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        target_doc_ids: List[str],
    ) -> int:
        table2attrs = self._materialized_full_postprocess_attrs(schema_query)
        if not table2attrs:
            return 0

        updated_count = 0
        doc_text_cache: Dict[str, str] = {}
        for doc_id in target_doc_ids:
            entry = res_data.get(doc_id)
            if not isinstance(entry, dict):
                continue

            for record in active_result_records(entry):
                table_name = str(record.get(RESULT_TABLE_KEY, record.get("table")) or "")
                attrs = table2attrs.get(table_name)
                if not attrs:
                    continue
                data = record.get(RESULT_DATA_KEY, {})
                if not isinstance(data, dict):
                    continue
                record_index = int(record.get("_record_index", 0) or 0)

                doc_text = doc_text_cache.get(str(doc_id))
                if doc_text is None:
                    try:
                        doc_text = self.loader.get_doc_text(doc_id)
                    except Exception:
                        doc_text = ""
                    doc_text_cache[str(doc_id)] = doc_text

                for attr in attrs:
                    if not is_none_value(data.get(attr)):
                        continue
                    imputed_value = self._impute_materialized_full_attr(
                        attr=attr,
                        data=data,
                        doc_text=doc_text,
                    )
                    if imputed_value is None:
                        continue
                    update_result_record_data(entry, record_index, attr, imputed_value)
                    data[attr] = imputed_value
                    updated_count += 1

        if updated_count:
            self._materialized_full_extraction_lookup_source = None
            self._materialized_full_extraction_lookup_index = None
        return updated_count

    @classmethod
    def _materialized_full_postprocess_attrs(
        cls,
        schema_query: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        table2attrs: Dict[str, List[str]] = {}
        for table_schema in schema_query or []:
            if not isinstance(table_schema, dict):
                continue
            table_name = table_schema.get(SCHEMA_NAME_KEY) or table_schema.get("name")
            if not table_name:
                continue
            attrs = []
            for attr_schema in table_schema.get(ATTRIBUTES_KEY, []) or []:
                if not isinstance(attr_schema, dict):
                    continue
                attr_name = attr_schema.get(ATTRIBUTE_NAME_KEY) or attr_schema.get("name")
                if not attr_name:
                    continue
                desc = str(
                    attr_schema.get(ATTRIBUTE_DESC_KEY)
                    or attr_schema.get("description")
                    or ""
                )
                attr_name_str = str(attr_name)
                if cls._is_closed_world_boolean_description(desc) or attr_name_str in {
                    "market_cap_updated_m",
                    "ticker",
                }:
                    attrs.append(attr_name_str)
            if attrs:
                table2attrs[str(table_name)] = attrs
        return table2attrs

    @staticmethod
    def _is_closed_world_boolean_description(description: str) -> bool:
        desc = description.lower()
        return (
            "closed-world boolean" in desc
            or "closed world boolean" in desc
            or "closed-world bool" in desc
            or "closed world bool" in desc
        )

    def _impute_materialized_full_attr(
        self,
        *,
        attr: str,
        data: Dict[str, Any],
        doc_text: str,
    ) -> Optional[str]:
        attr_key = str(attr)
        if attr_key == "market_cap_updated_m":
            return self._impute_market_cap_updated_m(doc_text)
        if attr_key == "ticker":
            return self._impute_ticker(doc_text)
        return self._impute_closed_world_boolean(attr_key, data, doc_text)

    @classmethod
    def _impute_closed_world_boolean(
        cls,
        attr: str,
        data: Dict[str, Any],
        doc_text: str,
    ) -> str:
        attr_key = attr.lower()
        if attr_key == "dropped_in_rank":
            change = cls._coerce_number(data.get("change_in_rank"))
            if change is not None:
                return "1" if change < 0 else "0"
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bdropped in rank\b",
                    r"\bdrop in rank\b",
                    r"\bdecline(?:d)? (?:of \d+ places? )?in (?:its )?rank\b",
                    r"\bmoved down\b",
                ],
            ) else "0"
        if attr_key == "gained_in_rank":
            change = cls._coerce_number(data.get("change_in_rank"))
            if change is not None:
                return "1" if change > 0 else "0"
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bgain(?:ed)? in rank\b",
                    r"\bimproved (?:its )?rank\b",
                    r"\bmoved up\b",
                    r"\brise in (?:the )?rank(?:ing)?\b",
                    r"\bupward trajectory\b",
                ],
            ) else "0"
        if attr_key == "is_profitable":
            profits = cls._coerce_number(data.get("profits_m"))
            if profits is not None:
                return "1" if profits > 0 else "0"
            return "1" if cls._has_positive_phrase(
                doc_text,
                [r"\bprofitable\b", r"\bprofitability\b", r"\bposted (?:a )?profit\b"],
            ) else "0"
        if attr_key == "growth_in_jobs":
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bgrowth in jobs\b",
                    r"\bjob growth\b",
                    r"\bemployment growth\b",
                    r"\bworkforce growth\b",
                    r"\bjob creation\b",
                    r"\bcreated jobs?\b",
                    r"\badding jobs?\b",
                    r"\bincrease[sd]? in (?:employment|jobs|headcount|workforce)\b",
                    r"\bexpanded (?:its )?(?:workforce|employee base|headcount)\b",
                    r"\bexpanding (?:its )?(?:workforce|employee base|headcount)\b",
                    r"\bgrowing workforce\b",
                    r"\bworkforce expansion\b",
                    r"\bemployment opportunities\b",
                    r"\binvesting in human capital\b",
                ],
            ) else "0"
        if attr_key == "best_companies_to_work_for":
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bbest companies to work for\b",
                    r"\bbest workplaces?\b",
                ],
            ) else "0"
        if attr_key == "global500":
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bfortune global 500\b",
                    r"\bglobal fortune 500\b",
                    r"\bglobal 500\b",
                ],
            ) else "0"
        if attr_key == "founder_is_ceo":
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bfounder and (?:the )?ceo\b",
                    r"\bco-founder and (?:the )?ceo\b",
                    r"\bceo and (?:co-)?founder\b",
                    r"\bfounder[^.]{0,120}\b(?:serves as|is|as|currently serves as)[^.]{0,60}\bceo\b",
                    r"\bfounded by [^.]{0,120}\b(?:serves as|is|as|currently serves as)[^.]{0,60}\bceo\b",
                    r"\bchief executive[^.]{0,80}\b(?:founder|co-founder)\b",
                ],
            ) else "0"
        if attr_key == "is_female_ceo":
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bfemale ceo\b",
                    r"\bwoman ceo\b",
                    r"\bwoman-led\b",
                    r"\bled by [^.]{0,80}\bfemale\b",
                ],
            ) else "0"
        if attr_key == "newcomer_to_the_fortune500":
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bnewcomer to (?:the )?fortune\s*500\b",
                    r"\bnew entrant (?:to|in) (?:the )?fortune\s*500\b",
                    r"\bnewly entered (?:the )?fortune\s*500\b",
                    r"\bmade (?:its )?debut (?:on|in) (?:the )?fortune\s*500\b",
                    r"\bdebuted (?:on|in) (?:the )?fortune\s*500\b",
                ],
            ) else "0"
        if attr_key == "worlds_most_admired_companies":
            return "1" if cls._has_positive_phrase(
                doc_text,
                [
                    r"\bworld'?s most admired companies\b",
                    r"\bworlds most admired companies\b",
                    r"\bmost admired companies\b",
                ],
            ) else "0"
        return "0"

    @staticmethod
    def _coerce_number(value: Any) -> Optional[float]:
        if is_none_value(value):
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        text = re.sub(r"^\$", "", text)
        try:
            return float(text)
        except ValueError:
            return None

    @classmethod
    def _has_positive_phrase(cls, doc_text: str, patterns: List[str]) -> bool:
        text = str(doc_text or "")
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if not cls._is_negated_match(text, match.start(), match.end()):
                    return True
        return False

    @staticmethod
    def _is_negated_match(text: str, start: int, end: int) -> bool:
        prefix = text[max(0, start - 80):start].lower()
        suffix = text[end:min(len(text), end + 50)].lower()
        negation = (
            r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\black of\b|"
            r"\bdoes not\b|\bdid not\b|\bis not\b|\bwas not\b|"
            r"\bnot one of\b|\bmay not\b|\bnot a\b|\bnot an\b)"
        )
        return bool(re.search(negation, prefix) or re.search(r"^\s*(?:status|membership)?\s*[:=-]?\s*(?:no|false|0)\b", suffix))

    @classmethod
    def _impute_market_cap_updated_m(cls, doc_text: str) -> Optional[str]:
        text = str(doc_text or "")
        patterns = [
            r"(?:updated figures? of|updated (?:market )?(?:cap(?:italization)?|value) of)\s*(?:approximately\s*)?\$?([\d,]+(?:\.\d+)?)\s*(billion|million|m|bn)?",
            r"(?:market (?:cap(?:italization)?|value)[^.]{0,120}?as of June 4, 2024[^$0-9]{0,40})\$?([\d,]+(?:\.\d+)?)\s*(billion|million|m|bn)?",
            r"\$?([\d,]+(?:\.\d+)?)\s*(billion|million|m|bn)?[^.]{0,80}as of June 4, 2024",
            r"(?:minor increase|slight adjustment|adjustment|increase|update)\s+to\s+\$?([\d,]+(?:\.\d+)?)\s*(billion|million|m|bn)?[^.]{0,80}\bupdated figures?\b",
            r"\$?([\d,]+(?:\.\d+)?)\s*(billion|million|m|bn)?[^.]{0,80}\bupdated figures?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = match.group(1)
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            amount = cls._coerce_number(value)
            if amount is None:
                continue
            if unit and unit.lower() in {"billion", "bn"}:
                amount *= 1000
            if amount.is_integer():
                return str(int(amount))
            return str(amount)
        return None

    @staticmethod
    def _impute_ticker(doc_text: str) -> Optional[str]:
        text = str(doc_text or "")
        patterns = [
            r"\bticker symbol\s*(?:of|is|:)?\s*[\"']?([A-Z][A-Z0-9.\-]{0,9})[\"']?\b",
            r"\b(?:NYSE|Nasdaq|NASDAQ)\s+ticker\s*[:=-]?\s*([A-Z][A-Z0-9.\-]{0,9})\b",
            r"\b(?:listed|operates|traded) under (?:the )?ticker(?: symbol)?\s*[\"']?([A-Z][A-Z0-9.\-]{0,9})[\"']?\b",
            r"\btraded under (?:the )?symbol\s*[\"']?([A-Z][A-Z0-9.\-]{0,9})[\"']?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(".,;:)")
        return None

    def _materialized_full_extraction_complete(
        self,
        res_data: Dict[str, Any],
        target_doc_ids: List[str],
    ) -> bool:
        if not res_data:
            return False
        table2attr: Dict[str, List[str]] = {
            s[SCHEMA_NAME_KEY]: [a[ATTRIBUTE_NAME_KEY] for a in s.get(ATTRIBUTES_KEY, [])]
            for s in self.schema_general
            if s.get(SCHEMA_NAME_KEY)
        }
        for doc_id in target_doc_ids:
            entry = res_data.get(doc_id)
            if not isinstance(entry, dict):
                return False
            records = active_result_records(entry)
            if not records:
                if RESULT_TABLE_KEY in entry or "records" in entry:
                    continue
                return False
            for record in records:
                table_assigned = record.get("table")
                if is_none_value(table_assigned) or table_assigned == NULL_VALUE:
                    continue
                attrs_needed = table2attr.get(str(table_assigned))
                if attrs_needed is None:
                    return False
                data = record.get(RESULT_DATA_KEY, {})
                if not isinstance(data, dict):
                    return False
                for attr in attrs_needed:
                    if attr not in data:
                        return False
                continue
        return True

    def _process_materialized_table_assignment(
        self,
        *,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
        excluded_doc_ids: Optional[set] = None,
        target_doc_ids: Optional[List[str]] = None,
    ) -> None:
        full_data = self._materialized_full_extraction_data or {}
        all_tables = [s[SCHEMA_NAME_KEY] for s in schema_query] if schema_query else [
            s[SCHEMA_NAME_KEY] for s in self.schema_general
        ]
        excluded_doc_ids = excluded_doc_ids or set()
        target_doc_ids = list(self.loader.doc_ids) if target_doc_ids is None else target_doc_ids
        updated_count = 0
        progress_bar = tqdm(
            total=len(target_doc_ids),
            initial=0,
            desc=f"Materialized Table Assignment {pgbar_name}",
        )
        try:
            for doc_id in target_doc_ids:
                self._wait_if_paused(f"materialized-table-assignment:{doc_id}")
                if doc_id in excluded_doc_ids:
                    progress_bar.update(1)
                    continue
                full_entry = full_data.get(doc_id, {})
                existing_entry = res_data.get(doc_id, {})

                existing_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
                if isinstance(existing_entry, dict):
                    for record in active_result_records(existing_entry):
                        table = str(record.get("table"))
                        record_id = str(record.get(RESULT_RECORD_ID_KEY) or record.get("_record_index") or "")
                        data = record.get(RESULT_DATA_KEY, {})
                        existing_by_key[(table, record_id)] = data if isinstance(data, dict) else {}

                records = []
                full_records = active_result_records(full_entry)
                for index, record in enumerate(full_records):
                    table = record.get("table")
                    if table not in all_tables:
                        continue
                    record_id = record.get(RESULT_RECORD_ID_KEY)
                    if record_id is None and len(full_records) > 1:
                        record_id = f"{doc_id}#{index}"
                    key = (str(table), str(record_id or index if len(full_records) > 1 else ""))
                    existing_data = existing_by_key.get(key, {})
                    records.append(
                        make_result_record(
                            table,
                            existing_data,
                            record_id=record_id,
                        )
                    )
                if not self.multi_record_extraction:
                    records = records[:1]

                new_entry = make_result_entry(
                    records,
                    include_records_for_single=True,
                )
                if not isinstance(existing_entry, dict) or existing_entry != new_entry:
                    res_data[doc_id] = new_entry
                    updated_count += 1
                    self._save_results_progress(res_path, res_data, updated_count)
                progress_bar.update(1)
        finally:
            progress_bar.close()
        if updated_count or not res_path.exists():
            self._save_results_progress(res_path, res_data, updated_count, force=True)
        logging.info(
            f"[{self.__class__.__name__}:_process_materialized_table_assignment] "
            f"Done materialized table assignment -> {res_path}"
        )

    def _lookup_materialized_attribute(
        self,
        doc_id: str,
        attr: str,
        *,
        record_index: int = 0,
        table: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> Any:
        doc_index = self._materialized_full_lookup_index().get(str(doc_id))
        if not isinstance(doc_index, dict):
            return None

        if "records" in doc_index:
            candidates = doc_index.get("records", [])
            if table is not None:
                candidates = doc_index.get("by_table", {}).get(str(table), [])
            if record_id is not None:
                by_id_key = (str(table), str(record_id)) if table is not None else str(record_id)
                by_id = doc_index.get("by_id", {}).get(by_id_key)
                if by_id is not None:
                    candidates = [by_id]
            record_payload = (
                candidates[record_index]
                if 0 <= record_index < len(candidates)
                else (candidates[0] if candidates else {})
            )
            data = record_payload.get(RESULT_DATA_KEY, {})
            if not isinstance(data, dict):
                return None
            return data.get(attr)

        data = doc_index.get(RESULT_DATA_KEY, {})
        if not isinstance(data, dict):
            return None
        return data.get(attr)

    def _materialized_full_lookup_index(self) -> Dict[str, Any]:
        full_data = self._materialized_full_extraction_data or {}
        if (
            getattr(self, "_materialized_full_extraction_lookup_source", None) is full_data
            and getattr(self, "_materialized_full_extraction_lookup_index", None) is not None
        ):
            return self._materialized_full_extraction_lookup_index

        lookup_index: Dict[str, Any] = {}
        for doc_id, entry in full_data.items():
            if not isinstance(entry, dict):
                continue
            records = active_result_records(entry)
            if records:
                doc_payload: Dict[str, Any] = {
                    "records": [],
                    "by_table": {},
                    "by_id": {},
                }
                for record in records:
                    data = record.get(RESULT_DATA_KEY, {})
                    record_payload = {
                        RESULT_DATA_KEY: data if isinstance(data, dict) else {},
                    }
                    table = record.get("table")
                    table_key = str(table)
                    doc_payload["records"].append(record_payload)
                    doc_payload["by_table"].setdefault(table_key, []).append(record_payload)
                    record_id = record.get(RESULT_RECORD_ID_KEY)
                    if record_id is not None:
                        doc_payload["by_id"][(table_key, str(record_id))] = record_payload
                        doc_payload["by_id"].setdefault(str(record_id), record_payload)
                lookup_index[str(doc_id)] = doc_payload
                continue

            data = entry.get(RESULT_DATA_KEY, {})
            lookup_index[str(doc_id)] = {
                RESULT_DATA_KEY: data if isinstance(data, dict) else {},
            }

        self._materialized_full_extraction_lookup_source = full_data
        self._materialized_full_extraction_lookup_index = lookup_index
        return lookup_index

    def _process_materialized_full_doc_extraction(
        self,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
        excluded_doc_ids: Optional[set] = None,
        query_id: Optional[str] = None,
        target_doc_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Materialize full doc/table/attr extraction with one LLM call per document.
        """
        del query_id, kwargs
        runtime = getattr(self.prompt_attr, "runtime", None) or getattr(self.prompt_table, "runtime", None)
        if runtime is None or self.disable_llm or self.config.get("use_gt_attr_extraction"):
            self._process_table_assignment(
                schema_query=schema_query,
                res_data=res_data,
                res_path=res_path,
                pgbar_name=pgbar_name,
                query_id=None,
                target_doc_ids=target_doc_ids,
                excluded_doc_ids=excluded_doc_ids,
            )
            reloaded = self.load_processed_res(res_path)
            res_data.clear()
            res_data.update(reloaded)
            self._process_materialized_full_attr_extraction(
                schema_query=schema_query,
                res_data=res_data,
                res_path=res_path,
                pgbar_name=pgbar_name,
                query_id=None,
                target_doc_ids=target_doc_ids,
                excluded_doc_ids=excluded_doc_ids,
            )
            return

        table2attr: Dict[str, List[str]] = {
            s[SCHEMA_NAME_KEY]: [a[ATTRIBUTE_NAME_KEY] for a in s.get(ATTRIBUTES_KEY, [])]
            for s in schema_query
            if s.get(SCHEMA_NAME_KEY)
        }
        all_tables = set(table2attr)
        excluded_doc_ids = excluded_doc_ids or set()
        target_doc_ids = list(self.loader.doc_ids) if target_doc_ids is None else target_doc_ids

        pending_doc_ids: List[str] = []
        for doc_id in target_doc_ids:
            if doc_id in excluded_doc_ids:
                continue
            entry = res_data.get(doc_id)
            if not isinstance(entry, dict):
                pending_doc_ids.append(doc_id)
                continue
            records = active_result_records(entry)
            if not records:
                if RESULT_TABLE_KEY in entry or "records" in entry:
                    continue
                pending_doc_ids.append(doc_id)
                continue
            for record in records:
                table = record.get("table")
                attrs_needed = table2attr.get(str(table))
                data = record.get(RESULT_DATA_KEY, {})
                if attrs_needed is None or not isinstance(data, dict):
                    pending_doc_ids.append(doc_id)
                    break
                if any(attr not in data for attr in attrs_needed):
                    pending_doc_ids.append(doc_id)
                    break

        progress_bar = tqdm(
            total=len(target_doc_ids),
            initial=len(target_doc_ids) - len(pending_doc_ids),
            desc=f"Materialized Full Extraction {pgbar_name}",
        )
        if not pending_doc_ids:
            progress_bar.close()
            return

        batch_size = max(
            1,
            int(
                getattr(
                    self,
                    "materialized_full_extraction_batch_size",
                    self.config.get("materialized_full_extraction_batch_size", 16),
                )
                or 16
            ),
        )
        batch_max_chars = max(
            1,
            int(
                getattr(
                    self,
                    "materialized_full_extraction_batch_max_chars",
                    self.config.get("materialized_full_extraction_batch_max_chars", 24000),
                )
                or 24000
            ),
        )
        concurrency = max(
            1,
            int(
                getattr(
                    self,
                    "materialized_full_extraction_concurrency",
                    self.config.get("materialized_full_extraction_concurrency", 1),
                )
                or 1
            ),
        )

        updated_count = 0
        batches = list(self._iter_materialized_doc_batches(
            pending_doc_ids,
            batch_size=batch_size,
            batch_max_chars=batch_max_chars,
        ))

        if concurrency <= 1:
            for batch in batches:
                for doc_id, _doc_text in batch:
                    self._wait_if_paused(f"materialized-full-extraction:{doc_id}")

                batch_results = self._extract_full_doc_batch(
                    batch_docs=batch,
                    schema=schema_query,
                    runtime=runtime,
                )
                self._record_materialized_full_doc_batch(
                    batch=batch,
                    batch_results=batch_results,
                    res_data=res_data,
                    table2attr=table2attr,
                    all_tables=all_tables,
                )
                updated_count += len(batch)
                self._save_results_progress(res_path, res_data, updated_count)
                progress_bar.update(len(batch))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_batch = {}
                for batch in batches:
                    for doc_id, _doc_text in batch:
                        self._wait_if_paused(f"materialized-full-extraction:{doc_id}")
                    future = executor.submit(
                        self._extract_full_doc_batch,
                        batch_docs=batch,
                        schema=schema_query,
                        runtime=runtime,
                    )
                    future_to_batch[future] = batch

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                    except Exception as error:
                        logging.warning(
                            f"[{self.__class__.__name__}:_process_materialized_full_doc_extraction] "
                            f"Concurrent batch failed for {len(batch)} docs: {error}; "
                            "retrying this batch synchronously."
                        )
                        batch_results = self._extract_full_doc_batch(
                            batch_docs=batch,
                            schema=schema_query,
                            runtime=runtime,
                        )
                    self._record_materialized_full_doc_batch(
                        batch=batch,
                        batch_results=batch_results,
                        res_data=res_data,
                        table2attr=table2attr,
                        all_tables=all_tables,
                    )
                    updated_count += len(batch)
                    self._save_results_progress(res_path, res_data, updated_count)
                    progress_bar.update(len(batch))

        progress_bar.close()
        self._save_results_progress(res_path, res_data, updated_count, force=True)

    def _record_materialized_full_doc_batch(
        self,
        *,
        batch: List[tuple[str, str]],
        batch_results: Dict[str, List[Dict[str, Any]]],
        res_data: Dict[str, Any],
        table2attr: Dict[str, List[str]],
        all_tables: set,
    ) -> None:
        for doc_id, _doc_text in batch:
            extracted_records = []
            raw_records = batch_results.get(doc_id, [])
            if not self.multi_record_extraction:
                raw_records = raw_records[:1]
            for index, record in enumerate(raw_records):
                table_assigned = record.get("table")
                extracted = record.get(RESULT_DATA_KEY, {})
                if table_assigned not in all_tables or is_none_value(table_assigned):
                    continue
                if not isinstance(extracted, dict):
                    extracted = {}
                attrs_needed = table2attr[str(table_assigned)]
                data = {}
                for attr in attrs_needed:
                    value = extracted.get(attr)
                    data[attr] = NULL_VALUE if is_none_value(value) else value
                record_id = record.get(RESULT_RECORD_ID_KEY)
                if record_id is None and len(raw_records) > 1:
                    record_id = f"{doc_id}#{index}"
                extracted_records.append(
                    make_result_record(table_assigned, data, record_id=record_id)
                )
            res_data[doc_id] = make_result_entry(
                extracted_records,
                include_records_for_single=True,
            )

    def _iter_materialized_doc_batches(
        self,
        doc_ids: List[str],
        *,
        batch_size: int,
        batch_max_chars: int,
    ):
        batch: List[tuple[str, str]] = []
        batch_chars = 0
        for doc_id in doc_ids:
            doc_text = self.loader.get_doc_text(doc_id)
            doc_chars = len(doc_text)
            if batch and (
                len(batch) >= batch_size
                or batch_chars + doc_chars > batch_max_chars
            ):
                yield batch
                batch = []
                batch_chars = 0
            batch.append((doc_id, doc_text))
            batch_chars += doc_chars
        if batch:
            yield batch

    def _extract_full_doc_batch(
        self,
        *,
        batch_docs: List[tuple[str, str]],
        schema: List[Dict[str, Any]],
        runtime: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not batch_docs:
            return {}
        if len(batch_docs) == 1:
            doc_id, doc_text = batch_docs[0]
            return {
                doc_id: self._extract_full_doc_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    schema=schema,
                    runtime=runtime,
                )
            }

        from redd.llm import CompletionRequest

        if self.multi_record_extraction:
            output_instruction = (
                "For each document, identify every table record expressed by the document. "
                "A document may contain zero, one, or multiple records, possibly across different tables.\n"
                "For each record, choose the matching table and extract values for every attribute in that table.\n"
                "Return only one JSON object keyed by doc_id. Each value must have exactly these keys:\n"
                "- \"Records\": a list of record objects\n"
                "Each record object must have exactly these keys:\n"
                "- \"Table Assignment\": the selected table name\n"
                "- \"Data Extracted\": an object mapping attribute names to values for that table\n"
                "- \"Record ID\": a short stable identifier if the document contains multiple records; otherwise null\n"
            )
        else:
            output_instruction = (
                "For each document, choose the single best matching table record expressed by the document.\n"
                "Extract values for every attribute in that table.\n"
                "Return only one JSON object keyed by doc_id. Each value must have exactly these keys:\n"
                "- \"Table Assignment\": the selected table name\n"
                "- \"Data Extracted\": an object mapping attribute names to values for that table\n"
            )

        prompt = (
            "You are a database expert.\n"
            "Analyze each document against the provided table schemas.\n"
            f"{output_instruction}"
            "Use null for missing attribute values. Include every input doc_id. Do not include explanations.\n\n"
            "Input:\n"
            + json.dumps(
                {
                    "Documents": [
                        {"doc_id": doc_id, DOCUMENT_KEY: doc_text}
                        for doc_id, doc_text in batch_docs
                    ],
                    SCHEMA_KEY: schema,
                },
                ensure_ascii=False,
            )
        )
        try:
            result = runtime.complete_model(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    response_format="json_object",
                    context={
                        "stage": "materialized_full_extraction",
                        "doc_ids": [doc_id for doc_id, _doc_text in batch_docs],
                    },
                ),
                FullDocumentExtractionBatchOutput,
            )
        except Exception as error:
            logging.warning(
                f"[{self.__class__.__name__}:_extract_full_doc_batch] "
                f"Batch full extraction failed for {len(batch_docs)} docs: {error}; "
                "splitting the batch."
            )
            midpoint = len(batch_docs) // 2
            return {
                **self._extract_full_doc_batch(
                    batch_docs=batch_docs[:midpoint],
                    schema=schema,
                    runtime=runtime,
                ),
                **self._extract_full_doc_batch(
                    batch_docs=batch_docs[midpoint:],
                    schema=schema,
                    runtime=runtime,
                ),
            }

        raw_results = getattr(result, "root", result)
        if not isinstance(raw_results, dict):
            raw_results = {}

        coerced: Dict[str, List[Dict[str, Any]]] = {}
        for doc_id, doc_text in batch_docs:
            value = raw_results.get(doc_id)
            if value is None:
                logging.warning(
                    f"[{self.__class__.__name__}:_extract_full_doc_batch] "
                    f"Batch result missing doc {doc_id}; falling back to single-doc extraction."
                )
                coerced[doc_id] = self._extract_full_doc_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    schema=schema,
                    runtime=runtime,
                )
            else:
                coerced[doc_id] = self._coerce_full_doc_extraction_result(
                    value,
                    allow_multiple=self.multi_record_extraction,
                )
        return coerced

    @staticmethod
    def _coerce_full_doc_extraction_result(
        value: Any,
        *,
        allow_multiple: bool = True,
    ) -> List[Dict[str, Any]]:
        if isinstance(value, FullDocumentExtractionOutput):
            if value.records:
                records = [
                    make_result_record(
                        record.table_assignment,
                        record.data_extracted or {},
                        record_id=record.record_id,
                    )
                    for record in value.records
                ]
                return records if allow_multiple else records[:1]
            return [make_result_record(value.table_assignment, value.data_extracted or {})]
        if not isinstance(value, dict):
            return []
        raw_records = value.get("Records")
        if raw_records is None:
            raw_records = value.get("records")
        if isinstance(raw_records, list):
            records = []
            for raw_record in raw_records:
                if not isinstance(raw_record, dict):
                    continue
                table = raw_record.get(TABLE_ASSIGNMENT_KEY)
                if table is None:
                    table = raw_record.get("table_assignment")
                if table is None:
                    table = raw_record.get("table")
                if table is None:
                    table = raw_record.get(RESULT_TABLE_KEY)
                data = raw_record.get(DATA_EXTRACTED_KEY)
                if data is None:
                    data = raw_record.get("data_extracted")
                if data is None:
                    data = raw_record.get(RESULT_DATA_KEY)
                records.append(
                    make_result_record(
                        table,
                        data if isinstance(data, dict) else {},
                        record_id=raw_record.get("Record ID", raw_record.get(RESULT_RECORD_ID_KEY)),
                    )
                )
            return records if allow_multiple else records[:1]
        table_assignment = value.get(TABLE_ASSIGNMENT_KEY)
        if table_assignment is None:
            table_assignment = value.get("table_assignment")
        if table_assignment is None:
            table_assignment = value.get("table")
        if table_assignment is None:
            table_assignment = value.get(RESULT_TABLE_KEY)
        data_extracted = value.get(DATA_EXTRACTED_KEY)
        if data_extracted is None:
            data_extracted = value.get("data_extracted")
        if data_extracted is None:
            data_extracted = value.get(RESULT_DATA_KEY)
        if not isinstance(data_extracted, dict):
            data_extracted = {}
        return [make_result_record(table_assignment, data_extracted)]

    def _extract_full_doc_single_doc(
        self,
        *,
        doc_id: str,
        doc_text: str,
        schema: List[Dict[str, Any]],
        runtime: Any,
    ) -> List[Dict[str, Any]]:
        from redd.llm import CompletionRequest

        if self.multi_record_extraction:
            output_instruction = (
                "Identify every table record expressed by the document. "
                "A document may contain zero, one, or multiple records, possibly across different tables.\n"
                "For each record, choose the matching table and extract values for every attribute in that table.\n"
                "Return only JSON with exactly these keys:\n"
                "- \"Records\": a list of record objects\n"
                "Each record object must have exactly these keys:\n"
                "- \"Table Assignment\": the selected table name\n"
                "- \"Data Extracted\": an object mapping attribute names to values for that table\n"
                "- \"Record ID\": a short stable identifier if the document contains multiple records; otherwise null\n"
            )
        else:
            output_instruction = (
                "Choose the single best matching table record expressed by the document.\n"
                "Extract values for every attribute in that table.\n"
                "Return only JSON with exactly these keys:\n"
                "- \"Table Assignment\": the selected table name\n"
                "- \"Data Extracted\": an object mapping attribute names to values for that table\n"
            )
        prompt = (
            "You are a database expert.\n"
            "Analyze the document against the provided table schemas.\n"
            f"{output_instruction}"
            "Use null for missing attribute values. Do not include explanations.\n\n"
            "Input:\n"
            + json.dumps(
                {
                    DOCUMENT_KEY: doc_text,
                    SCHEMA_KEY: schema,
                },
                ensure_ascii=False,
            )
        )
        try:
            result = runtime.complete_model(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    response_format="json_object",
                    context={
                        "stage": "materialized_full_extraction",
                        "doc_id": doc_id,
                    },
                ),
                FullDocumentExtractionOutput,
            )
        except Exception as error:
            logging.warning(
                f"[{self.__class__.__name__}:_extract_full_doc_single_doc] "
                f"Full doc extraction failed for doc {doc_id}: {error}; "
                "falling back to table-then-attr extraction."
            )
            if hasattr(self.loader, "get_doc_text"):
                doc_text = self.loader.get_doc_text(doc_id)
            all_tables = [s[SCHEMA_NAME_KEY] for s in schema]
            table, failed, _reason = self._assign_table_single_doc(
                doc_id=doc_id,
                doc_text=doc_text,
                all_tables=all_tables,
                prompt_schema=schema,
                usage_stage="materialized_full_fallback_table_assignment",
            )
            if failed or table not in all_tables:
                return []
            table_schema = next(s for s in schema if s[SCHEMA_NAME_KEY] == table)
            attrs = [a[ATTRIBUTE_NAME_KEY] for a in table_schema.get(ATTRIBUTES_KEY, [])]
            data = self._extract_all_attrs_single_doc(
                doc_id=doc_id,
                doc_text=doc_text,
                attrs=attrs,
                table_schema=table_schema,
                usage_stage="materialized_full_fallback_attr_extraction",
            )
            return [make_result_record(table, data)]
        return self._coerce_full_doc_extraction_result(
            result,
            allow_multiple=self.multi_record_extraction,
        )

    def _process_materialized_full_attr_extraction(
        self,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
        excluded_doc_ids: Optional[set] = None,
        query_id: Optional[str] = None,
        target_doc_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Extract all attributes for the query-independent materialized artifact.

        Unlike query execution, the materialized artifact needs every attribute
        for the assigned table. Running one LLM call per document/table keeps the
        artifact semantics while avoiding one request per cell.
        """
        del query_id, kwargs
        all_tables = [s[SCHEMA_NAME_KEY] for s in self.schema_general]
        table2schema = {s[SCHEMA_NAME_KEY]: s for s in self.schema_general}
        table2attr: Dict[str, List[str]] = {
            s[SCHEMA_NAME_KEY]: [a[ATTRIBUTE_NAME_KEY] for a in s.get(ATTRIBUTES_KEY, [])]
            for s in schema_query
            if s.get(SCHEMA_NAME_KEY) in all_tables
        }
        excluded_doc_ids = excluded_doc_ids or set()
        target_doc_ids = list(self.loader.doc_ids) if target_doc_ids is None else target_doc_ids
        use_gt_attr = bool(self.config.get("use_gt_attr_extraction", False) or self.disable_llm)

        pending_docs: List[tuple[str, int, str, Optional[str], List[str]]] = []
        total_attrs = 0
        already_processed_attrs = 0
        for doc_id in target_doc_ids:
            if doc_id in excluded_doc_ids or doc_id not in res_data:
                continue
            for record in active_result_records(res_data[doc_id]):
                table_assigned = record.get("table")
                if is_none_value(table_assigned) or table_assigned == NULL_VALUE:
                    continue
                attrs_needed = table2attr.get(str(table_assigned), [])
                total_attrs += len(attrs_needed)
                existing_data = record.get(RESULT_DATA_KEY, {})
                if not isinstance(existing_data, dict):
                    existing_data = {}
                missing_attrs = [attr for attr in attrs_needed if attr not in existing_data]
                already_processed_attrs += len(attrs_needed) - len(missing_attrs)
                if missing_attrs:
                    pending_docs.append(
                        (
                            doc_id,
                            int(record.get("_record_index") or 0),
                            str(table_assigned),
                            record.get(RESULT_RECORD_ID_KEY),
                            missing_attrs,
                        )
                    )

        logging.info(
            f"[{self.__class__.__name__}:_process_materialized_full_attr_extraction] "
            f"Total attrs: {total_attrs}, already processed: {already_processed_attrs}, "
            f"pending docs: {len(pending_docs)}"
        )
        progress_bar = tqdm(
            total=total_attrs,
            initial=already_processed_attrs,
            desc=f"Materialized Attr Extraction {pgbar_name}",
        )
        if not pending_docs:
            progress_bar.close()
            return

        task_to_gt_table: Dict[str, str] = {}
        attr_name_map: Dict[str, Dict[str, Any]] = {}
        if use_gt_attr and hasattr(self.loader, "load_name_map"):
            name_map = self.loader.load_name_map(None)
            if isinstance(name_map, dict):
                table_map = name_map.get("table", {})
                if isinstance(table_map, dict):
                    task_to_gt_table = dict(table_map)
                attribute_map = name_map.get("attribute", {})
                if isinstance(attribute_map, dict):
                    attr_name_map = dict(attribute_map)

        updated_doc_count = 0
        for doc_id, record_index, table_assigned, record_id, missing_attrs in pending_docs:
            self._wait_if_paused(f"materialized-attr-extraction:{doc_id}")
            if use_gt_attr:
                extracted = {
                    attr: self._get_gt_attribute_value(
                        doc_id=doc_id,
                        task_table=table_assigned,
                        task_attribute=attr,
                        task_to_gt_table=task_to_gt_table,
                        attr_name_map=attr_name_map,
                        record_index=record_index,
                        record_id=record_id,
                    )
                    for attr in missing_attrs
                }
            else:
                doc_text = self.loader.get_doc_text(doc_id)
                extracted = self._extract_all_attrs_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    attrs=missing_attrs,
                    table_schema=table2schema[table_assigned],
                    query_id=None,
                    usage_stage="materialized_full_attr_extraction",
                )

            for attr in missing_attrs:
                attr_val = extracted.get(attr)
                attr_val = NULL_VALUE if is_none_value(attr_val) else attr_val
                update_result_record_data(res_data[doc_id], record_index, attr, attr_val)
            updated_doc_count += 1
            self._save_results_progress(res_path, res_data, updated_doc_count)
            progress_bar.update(len(missing_attrs))

        progress_bar.close()
        self._save_results_progress(res_path, res_data, updated_doc_count, force=True)

    def _extract_all_attrs_single_doc(
        self,
        *,
        doc_id: str,
        doc_text: str,
        attrs: List[str],
        table_schema: Dict[str, Any],
        query_id: Optional[str] = None,
        usage_stage: str = "attribute_extraction",
    ) -> Dict[str, Any]:
        if not attrs:
            return {}

        runtime = getattr(self.prompt_attr, "runtime", None)
        if runtime is None:
            return {
                attr: self._extract_attr_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    attr=attr,
                    table_schema=table_schema,
                    query_id=query_id,
                    usage_stage=usage_stage,
                )
                for attr in attrs
            }

        from redd.llm import CompletionRequest

        prompt = (
            "You are a database expert.\n"
            "Extract values for every requested attribute from the document under the provided schema.\n"
            "Return one JSON object whose keys are exactly the requested attribute names.\n"
            "If an attribute value is not present in the document, return null for that key.\n"
            "Do not include explanations or extra keys.\n\n"
            "Input:\n"
            + json.dumps(
                {
                    DOCUMENT_KEY: doc_text,
                    SCHEMA_KEY: table_schema,
                    "Target Attributes": attrs,
                },
                ensure_ascii=False,
            )
        )
        try:
            result = runtime.complete_model(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    response_format="json_object",
                    context={
                        "stage": usage_stage,
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "table": table_schema.get(SCHEMA_NAME_KEY),
                        "attributes": list(attrs),
                    },
                ),
                AttributeExtractionOutput,
            ).root
        except Exception as error:
            logging.warning(
                f"[{self.__class__.__name__}:_extract_all_attrs_single_doc] "
                f"Batch extraction failed for doc {doc_id}: {error}; falling back to per-attr extraction."
            )
            return {
                attr: self._extract_attr_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    attr=attr,
                    table_schema=table_schema,
                    query_id=query_id,
                    usage_stage=usage_stage,
                )
                for attr in attrs
            }
        if not isinstance(result, dict):
            return {attr: None for attr in attrs}
        return {attr: result.get(attr) for attr in attrs}

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
        cache_enabled = bool(getattr(self, "table_assignment_cache_enabled", False))
        cache_general_schema = bool(getattr(self, "table_assignment_cache_general_schema", False))
        cache_schema = self.schema_general if cache_general_schema else schema_query
        cache_tables = (
            [s[SCHEMA_NAME_KEY] for s in cache_schema]
            if cache_schema
            else all_tables
        )
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
        schema_signature = tuple(sorted(str(table) for table in all_tables))
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
        cache_hit_count = 0
        cache_miss_count = 0
        source_table_metadata_hit_count = 0
        source_table_metadata_miss_count = 0
        unknown_schema_null_count = 0
        max_retries_null_count = 0
        updated_result_count = 0
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

            cached_table = (
                self._table_assignment_cache.get(doc_id)
                if cache_enabled
                else None
            )
            null_cache_key = (schema_signature, doc_id)
            if cache_enabled and cached_table and cached_table in all_tables:
                res_data[doc_id] = make_legacy_result_entry(cached_table, {})
                updated_result_count += 1
                self._save_results_progress(
                    res_path,
                    res_data,
                    updated_result_count,
                )
                cache_hit_count += 1
                progress_bar.update(1)
                continue
            if cache_enabled and cached_table:
                res_data[doc_id] = make_legacy_result_entry(NULL_VALUE, {})
                updated_result_count += 1
                self._save_results_progress(
                    res_path,
                    res_data,
                    updated_result_count,
                )
                cache_hit_count += 1
                progress_bar.update(1)
                continue
            if cache_enabled and null_cache_key in self._table_assignment_null_cache:
                res_data[doc_id] = make_legacy_result_entry(NULL_VALUE, {})
                updated_result_count += 1
                self._save_results_progress(
                    res_path,
                    res_data,
                    updated_result_count,
                )
                cache_hit_count += 1
                progress_bar.update(1)
                continue

            table_assigned = None
            gt_result_records: Optional[List[Dict[str, Any]]] = None
            metadata_found = False
            metadata_lookup_enabled = bool(
                getattr(self, "table_assignment_source_table_metadata", False)
            )
            if metadata_lookup_enabled:
                metadata_found, table_assigned = self._get_source_table_metadata_assignment(
                    doc_id=doc_id,
                    cache_tables=cache_tables,
                )

            if metadata_found:
                source_table_metadata_hit_count += 1
            else:
                source_table_metadata_miss_count += int(metadata_lookup_enabled)
                cache_miss_count += int(cache_enabled)
            if metadata_found:
                pass
            elif use_gt:
                gt_result_records = self._get_gt_result_records(
                    doc_id=doc_id,
                    all_tables=cache_tables,
                    gt_to_task_table=gt_to_task_table,
                )
                table_assigned = (
                    gt_result_records[0].get("table") if gt_result_records else None
                )
            else:
                if hasattr(self.loader, "get_doc_text"):
                    doc_text = self.loader.get_doc_text(doc_id)
                else:
                    doc_text = self.loader.get_doc(doc_id)[0]
                table_assigned, table_failed, skip_reason = self._assign_table_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    all_tables=cache_tables,
                    prompt_schema=cache_schema,
                    max_retries=max_table_retries,
                    query_id=query_id,
                    **kwargs,
                )
                if table_failed:
                    if skip_reason == "unknown_schema":
                        unknown_schema_null_count += 1
                    elif skip_reason == "max_retries":
                        max_retries_null_count += 1
                    table_assigned = NULL_VALUE

            table_assigned = NULL_VALUE if is_none_value(table_assigned) else table_assigned
            if cache_enabled:
                if table_assigned in cache_tables:
                    self._table_assignment_cache[doc_id] = str(table_assigned)
                else:
                    self._table_assignment_null_cache.add(null_cache_key)
            if table_assigned not in all_tables:
                table_assigned = NULL_VALUE
            # Initialize entry with table assignment and empty data
            if gt_result_records is not None:
                records_for_output = (
                    gt_result_records
                    if self.multi_record_extraction
                    else gt_result_records[:1]
                )
                res_data[doc_id] = make_result_entry(
                    [
                        record
                        for record in records_for_output
                        if record.get("table") in all_tables
                    ],
                    include_records_for_single=True,
                )
            else:
                res_data[doc_id] = make_legacy_result_entry(table_assigned, {})
            updated_result_count += 1
            self._save_results_progress(
                res_path,
                res_data,
                updated_result_count,
            )
            progress_bar.update(1)

        progress_bar.close()
        if updated_result_count or not res_path.exists():
            self._save_results_progress(res_path, res_data, updated_result_count, force=True)
        stats_parts = [f"excluded: {excluded_count}"]
        if cache_enabled:
            stats_parts.append(f"cache_hits: {cache_hit_count}")
            stats_parts.append(f"cache_misses: {cache_miss_count}")
            self._record_table_assignment_cache_event(
                query_id=query_id or "",
                input_docs=total_docs,
                cache_hits=cache_hit_count,
                cache_misses=cache_miss_count,
                excluded=excluded_count,
                source_table_metadata_hits=source_table_metadata_hit_count,
                source_table_metadata_misses=source_table_metadata_miss_count,
            )
        if unknown_schema_null_count > 0:
            stats_parts.append(f"unknown_schema_null: {unknown_schema_null_count}")
        if max_retries_null_count > 0:
            stats_parts.append(f"max_retries_null: {max_retries_null_count}")
        logging.info(
            f"[{self.__class__.__name__}:_process_table_assignment] Done table assignment -> {res_path} "
            f"({', '.join(stats_parts)})"
        )

    def _record_table_assignment_cache_event(
        self,
        *,
        query_id: str,
        input_docs: int,
        cache_hits: int,
        cache_misses: int,
        excluded: int,
        source_table_metadata_hits: int = 0,
        source_table_metadata_misses: int = 0,
    ) -> None:
        if not self.table_assignment_cache_enabled:
            return
        event = {
            "dataset": self.data_path.name,
            "query_id": query_id,
            "input_docs": input_docs,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "excluded": excluded,
            "source_table_metadata_hits": source_table_metadata_hits,
            "source_table_metadata_misses": source_table_metadata_misses,
        }
        self._table_assignment_cache_events.append(event)
        out_path = self.out_root / "table_assignment_cache.json"
        payload = {
            "enabled": True,
            "events": self._table_assignment_cache_events,
            "totals": {
                "input_docs": sum(item["input_docs"] for item in self._table_assignment_cache_events),
                "cache_hits": sum(item["cache_hits"] for item in self._table_assignment_cache_events),
                "cache_misses": sum(item["cache_misses"] for item in self._table_assignment_cache_events),
                "excluded": sum(item["excluded"] for item in self._table_assignment_cache_events),
                "source_table_metadata_hits": sum(
                    item.get("source_table_metadata_hits", 0)
                    for item in self._table_assignment_cache_events
                ),
                "source_table_metadata_misses": sum(
                    item.get("source_table_metadata_misses", 0)
                    for item in self._table_assignment_cache_events
                ),
            },
        }
        self.save_results(str(out_path), payload)

    def _get_source_table_metadata_assignment(
        self,
        *,
        doc_id: str,
        cache_tables: List[str],
    ) -> tuple[bool, Optional[str]]:
        if not hasattr(self.loader, "get_doc"):
            return False, None
        try:
            _doc_text, _resolved_doc_id, metadata = self.loader.get_doc(doc_id)
        except Exception:
            return False, None
        if not isinstance(metadata, dict):
            return False, None

        schema_tables = metadata.get("schema_tables")
        if isinstance(schema_tables, list):
            for table_name in schema_tables:
                table = str(table_name or "").strip()
                if table in cache_tables:
                    return True, table

        for key in ("schema_table", "table_id"):
            table = str(metadata.get(key) or "").strip()
            if table:
                return True, table if table in cache_tables else NULL_VALUE

        source_table = metadata.get("table_name") or metadata.get("source_table")
        if is_none_value(source_table):
            return False, None
        table = str(source_table).strip()
        if not table or table not in cache_tables:
            return False, None
        return True, table

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

        from .strategies.proxy_runtime import ProxyRuntimeExtractionStrategy

        orchestrator = ProxyRuntimeExtractionStrategy(
            extraction_config=extraction_config,
            data_path=self.data_path,
            loader=self.loader,
            api_key=self.api_key,
            train_doc_ids=self.train_doc_ids,
            extraction_cache=self._proxy_extraction_cache,
        )
        orchestrator.process_proxy_runtime_per_table(
            qid=qid,
            schema_query=schema_query,
            res_data=res_data,
            res_path=res_path,
            save_results_fn=lambda p, d: self.save_results(str(p), d),
        )

    def _get_gt_result_records(
        self,
        doc_id: str,
        all_tables: List[str],
        gt_to_task_table: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Get all task-schema records for a document from ground truth."""
        if not hasattr(self.loader, "get_doc_info"):
            raise RuntimeError(
                "use_gt_table_assignment requires loader with get_doc_info"
            )
        doc_info = self.loader.get_doc_info(doc_id)
        if not doc_info:
            return []
        data_records = doc_info.get("data_records") or []
        if not data_records and doc_info.get("table"):
            data_records = [{"table_name": doc_info.get("table"), "data": doc_info.get("data", {})}]

        records = []
        for index, gt_record in enumerate(data_records):
            if not isinstance(gt_record, dict):
                continue
            gt_table = gt_record.get("table_name") or gt_record.get("table")
            if not gt_table:
                continue
            task_table = gt_to_task_table.get(gt_table, gt_table)
            if task_table not in all_tables:
                continue
            record_id = (
                gt_record.get(RESULT_RECORD_ID_KEY)
                or gt_record.get("row_id")
                or gt_record.get("source_row_id")
            )
            if record_id is None and len(data_records) > 1:
                record_id = f"{doc_id}#{index}"
            records.append(make_result_record(task_table, {}, record_id=record_id))
        return records

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
        records = self._get_gt_result_records(doc_id, all_tables, gt_to_task_table)
        return records[0].get("table") if records else None

    def _assign_table_single_doc(
        self,
        doc_id: str,
        doc_text: str,
        all_tables: List[str],
        prompt_schema: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = DEFAULT_MAX_TABLE_RETRIES,
        max_consecutive_unknown: int = MAX_CONSECUTIVE_UNKNOWN_SCHEMA,
        query_id: Optional[str] = None,
        usage_stage: str = "table_assignment",
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
                call_kwargs = {
                    **self.retry_params,
                    "usage_context": {
                        "stage": usage_stage,
                        "query_id": query_id,
                        "doc_id": doc_id,
                    },
                }
                res_tbl = self.prompt_table.complete_model(
                    tbl_input,
                    TableAssignmentOutput,
                    **call_kwargs,
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

        # Build pending tasks: list of (doc_id, record_index, table, record_id, attr).
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

            for record in active_result_records(res_data[doc_id]):
                table_assigned = record.get("table")
                if is_none_value(table_assigned) or table_assigned == NULL_VALUE:
                    continue

                attrs_needed = table2attr.get(str(table_assigned), [])
                total_attrs += len(attrs_needed)

                existing_data = record.get(RESULT_DATA_KEY, {})
                if not isinstance(existing_data, dict):
                    existing_data = {}

                for attr in attrs_needed:
                    if attr in existing_data:
                        already_processed_attrs += 1
                    else:
                        pending_tasks.append(
                            (
                                doc_id,
                                int(record.get("_record_index") or 0),
                                str(table_assigned),
                                record.get(RESULT_RECORD_ID_KEY),
                                attr,
                            )
                        )

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

        # Process each (doc_id, record_index, table, record_id, attr) tuple.
        updated_attr_count = 0
        for doc_id, record_index, table_assigned, record_id, attr in pending_tasks:
            self._wait_if_paused(f"attr-extraction:{doc_id}:{attr}")
            if use_gt_attr:
                attr_val = self._get_gt_attribute_value(
                    doc_id=doc_id,
                    task_table=table_assigned,
                    task_attribute=attr,
                    task_to_gt_table=task_to_gt_table,
                    attr_name_map=attr_name_map,
                    record_index=record_index,
                    record_id=record_id,
                )
            elif self._materialized_full_extraction_data is not None:
                attr_val = self._lookup_materialized_attribute(
                    doc_id,
                    attr,
                    record_index=record_index,
                    table=table_assigned,
                    record_id=record_id,
                )
            else:
                doc_text = self.loader.get_doc_text(doc_id)
                attr_val = self._extract_attr_single_doc(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    attr=attr,
                    table_schema=table2schema[table_assigned],
                    max_retries=max_attr_retries,
                    query_id=query_id,
                    **kwargs,
                )

            attr_val = NULL_VALUE if is_none_value(attr_val) else attr_val
            if isinstance(attr_val, str) and len(attr_val) > MAX_ATTRIBUTE_VALUE_LENGTH:
                logging.info(f"[{self.__class__.__name__}:_process_attr_extraction] Attr too long "
                             f"(>{len(attr_val)}): doc {doc_id} attr {attr}")
            update_result_record_data(res_data[doc_id], record_index, attr, attr_val)

            # Save after each attr extraction
            updated_attr_count += 1
            self._save_results_progress(res_path, res_data, updated_attr_count)
            progress_bar.update(1)

        progress_bar.close()
        self._save_results_progress(res_path, res_data, updated_attr_count, force=True)
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

            records = active_result_records(doc_result)
            if not records:
                continue

            projected_records = []
            for record in records:
                data = record.get(RESULT_DATA_KEY, {})
                if not isinstance(data, dict):
                    data = {}

                passed = True
                for attr, predicate_fn in predicate_fns:
                    if not predicate_fn(data.get(attr)):
                        passed = False
                        break
                if not passed:
                    changed = True
                    continue

                if output_attrs:
                    projected = {
                        attr: data[attr]
                        for attr in output_attrs
                        if attr in data and not is_none_value(data.get(attr))
                    }
                    if projected != data:
                        data = projected
                        changed = True
                projected_records.append(
                    make_result_record(
                        record.get("table"),
                        data,
                        record_id=record.get(RESULT_RECORD_ID_KEY),
                    )
                )

            new_entry = make_result_entry(
                projected_records,
                include_records_for_single=True,
            )
            if new_entry != doc_result:
                doc_result.clear()
                doc_result.update(new_entry)
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
        record_index: int = 0,
        record_id: Optional[str] = None,
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
        if record_id is not None:
            matching_records = [
                record
                for record in target_records
                if str(
                    record.get(RESULT_RECORD_ID_KEY)
                    or record.get("row_id")
                    or record.get("source_row_id")
                    or ""
                )
                == str(record_id)
            ]
            if matching_records:
                target_records = matching_records
            elif 0 <= record_index < len(target_records):
                target_records = [target_records[record_index]]
        elif 0 <= record_index < len(target_records):
            target_records = [target_records[record_index]]

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
        query_id: Optional[str] = None,
        usage_stage: str = "attribute_extraction",
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
                call_kwargs = {
                    **self.retry_params,
                    "usage_context": {
                        "stage": usage_stage,
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "table": table_schema.get(SCHEMA_NAME_KEY),
                        "attribute": attr,
                    },
                }
                res_attr = self.prompt_attr.complete_model(
                    attr_input,
                    AttributeExtractionOutput,
                    **call_kwargs,
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
        """String representation of the data extractor."""
        return f"{self.__class__.__name__} (mode={self.mode}): \n{self.param_str}"
