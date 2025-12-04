# data_population/datapop_api.py
"""
Base LLM model for data population using API providers (DeepSeek, SiliconFlow, TogetherAI, etc.).

This module handles data population by:
1. Assigning documents to appropriate tables
2. Extracting attribute values from documents
3. Saving results in structured JSON format

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

import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import re
import ast

from ..utils import constants
from ..utils.constants import (
    NULL_VALUE,
    SCHEMA_NAME_KEY, ATTRIBUTES_KEY, ATTRIBUTE_NAME_KEY,
    DOCUMENT_KEY, SCHEMA_KEY, TARGET_ATTRIBUTE_KEY,
    TABLE_ASSIGNMENT_KEY,
    RESULT_TABLE_KEY, RESULT_DATA_KEY,
    DEFAULT_MAX_TABLE_RETRIES, DEFAULT_MAX_ATTR_RETRIES,
    MAX_ATTRIBUTE_VALUE_LENGTH,
    PATH_TEMPLATES,
)
from ..data_loader import create_data_loader, DataLoaderBase
from .datapop_basic import DataPopBasic
from ..utils.utils import is_none_value

__all__ = ["DataPopAPI"]


class DataPopAPI(DataPopBasic):
    """
    Base class for API-based data population implementations.
    Provides common logic for table assignment and attribute extraction.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        required_keys = ["mode", "res_param_str", "llm_model", "prompts"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {missing_keys}")
        
        self.mode = config["mode"]
        self.param_str = config["res_param_str"]
        
        # Data loader configuration
        self.loader_type = config.get("data_loader_type", "sqlite")
        self.loader_config = config.get("data_loader_config", {})
        
        # API rate limit retry configuration
        self.retry_params = {}
        if "max_retries" in config:
            self.retry_params["max_retries"] = config["max_retries"]
        if "wait_time" in config:
            self.retry_params["wait_time"] = config["wait_time"]
        if self.retry_params:
            logging.info(f"[{self.__class__.__name__}:__init__] Rate limit retry enabled: {self.retry_params}")
        
        # These will be set by subclasses
        self.prompt_table = None
        self.prompt_attr = None

    def __call__(self, dn_fn_list: Optional[List[str]] = None) -> None:
        """
        Extract tabular data from documents in the specified datasets.
        
        Args: dn_fn_list: Datasets to process. If None, uses default SPIDER_DN_FN_LIST.
            Examples: ["store_1/customers-invoices", "wine_1/wine-appellations"]
        """
        if dn_fn_list is None:
            if "exp_dn_fn_list" in self.config:
                dn_fn_list = self.config["exp_dn_fn_list"]
            else:
                logging.warning(f"[{self.__class__.__name__}:__call__] No datasets specified, using default SPIDER_DN_FN_LIST")
                dn_fn_list = constants.SPIDER_DN_FN_LIST
        total_datasets = len(dn_fn_list)
        logging.info(f"[{self.__class__.__name__}:__call__] Processing {total_datasets} datasets: {dn_fn_list}")
        
        for dn_fn in dn_fn_list:
            # Separate data path and output path
            data_root = Path(self.config.get("data_main", self.config["out_main"])) / dn_fn
            out_root = Path(self.config["out_main"]) / dn_fn
            out_root.mkdir(parents=True, exist_ok=True)
            logging.info(f"[{self.__class__.__name__}:__call__] Start processing dataset: data_root={data_root}, out_root={out_root}")
            self._process_dataset(data_root, out_root)

    def _process_dataset(self, data_root: str | Path, out_root: str | Path):
        """Process all queries and documents in a dataset."""
        self.data_root = Path(data_root)
        self.out_root = Path(out_root)
        
        # Create data loader based on configuration
        self.loader = create_data_loader(
            data_root=self.data_root,
            loader_type=self.loader_type,
            loader_config=self.loader_config
        )
        logging.info(f"[{self.__class__.__name__}:_process_dataset] Created {self.loader.__class__.__name__} for {self.data_root}")
        
        query_dict = self.loader.load_query_dict()
        self.schema_general = self.loader.load_schema_general()

        if not query_dict:
            logging.warning(f"[{self.__class__.__name__}:_process_dataset] No queries found in dataset {self.data_root}")
            return
            
        logging.info(f"[{self.__class__.__name__}:_process_dataset] Processing {len(query_dict)} queries in dataset {self.data_root.name}")

        for qid in query_dict:
            schema_query = self.loader.load_schema_query(qid)
            res_path = self.out_root / PATH_TEMPLATES.data_population_result(qid, self.param_str)
            res_data = self.load_processed_res(res_path)
            pgbar_name = f"{self.data_root.name}-{qid}"

            logging.info(f"[{self.__class__.__name__}:_process_dataset] Start processing query-{qid} -> {res_path}")
            self._process_documents(
                schema_query=schema_query,
                res_data=res_data,
                res_path=res_path,
                pgbar_name=pgbar_name,
            )

    def _process_documents(
        self,
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
        max_table_retries: int = DEFAULT_MAX_TABLE_RETRIES,
        max_attr_retries: int = DEFAULT_MAX_ATTR_RETRIES,
    ) -> None:
        """Iterate over documents and populate table/attribute data."""
        all_tables = [s[SCHEMA_NAME_KEY] for s in self.schema_general]
        table2schema = {s[SCHEMA_NAME_KEY]: s for s in self.schema_general}
        table2attr: Dict[str, List[str]] = {
            s[SCHEMA_NAME_KEY]: [a[ATTRIBUTE_NAME_KEY] for a in s[ATTRIBUTES_KEY]]
            for s in schema_query
            if s[SCHEMA_NAME_KEY] in all_tables
        }

        total_docs = self.loader.num_docs
        already_processed = len(res_data)
        progress_bar = tqdm(
            total=total_docs, initial=0,
            desc=f"Processing {pgbar_name} ({already_processed}/{total_docs})"
        )
        
        for did, (doc_text, *_rest) in zip(self.loader.doc_ids, self.loader.iter_docs()):
            if did in res_data:
                progress_bar.update(1)
                continue

            # ‑‑ table assignment -----------------------------------------------------
            table_attempt = 0
            table_failed = False
            while True:
                if table_attempt > max_table_retries:
                    logging.info(f"[{self.__class__.__name__}:_process_documents] Table fail "
                                 f">{max_table_retries}x for doc {did}. Skipping.")
                    table_failed = True
                    break
                tbl_input = {DOCUMENT_KEY: doc_text, SCHEMA_KEY: self.schema_general}
                tbl_input = json.dumps(tbl_input, ensure_ascii=False)
                raw_text = self.prompt_table(msg=tbl_input, response_format="text", **self.retry_params).strip()
                res_tbl = self._extract_json_block(raw_text)
                if not res_tbl or TABLE_ASSIGNMENT_KEY not in res_tbl:
                    table_attempt += 1
                    continue
                table_assigned = res_tbl[TABLE_ASSIGNMENT_KEY]
                if table_assigned not in all_tables and not is_none_value(table_assigned):
                    table_attempt += 1
                    continue
                break
            
            if table_failed:
                progress_bar.update(1)
                continue

            table_assigned = NULL_VALUE if is_none_value(table_assigned) else table_assigned
            result_entry = {RESULT_TABLE_KEY: table_assigned, RESULT_DATA_KEY: {}}

            # ‑‑ attribute extraction -------------------------------------------------
            for attr in table2attr.get(table_assigned, []):
                attr_attempt = 0
                attr_val = None
                while True:
                    if attr_attempt > max_attr_retries:
                        logging.info(f"[{self.__class__.__name__}:_process_documents] Attr fail "
                                     f">{max_attr_retries}x for doc {did} attr {attr}. Skipping attr.")
                        break
                    attr_input = {
                        DOCUMENT_KEY: doc_text,
                        SCHEMA_KEY: table2schema[table_assigned],
                        TARGET_ATTRIBUTE_KEY: attr,
                    }
                    attr_input = json.dumps(attr_input, ensure_ascii=False)
                    raw_text = self.prompt_attr(msg=attr_input, response_format="text", **self.retry_params).strip()
                    res_attr = self._extract_json_block(raw_text)
                    if not res_attr or attr not in res_attr:
                        attr_attempt += 1
                        continue
                    attr_val = res_attr[attr]
                    break

                attr_val = NULL_VALUE if is_none_value(attr_val) else attr_val
                if isinstance(attr_val, str) and len(attr_val) > MAX_ATTRIBUTE_VALUE_LENGTH:
                    logging.info(f"[{self.__class__.__name__}:_process_documents] Attr too long "
                                    f"(>{len(attr_val)}): doc {did} attr {attr}")
                result_entry[RESULT_DATA_KEY][attr] = attr_val

            res_data[did] = result_entry
            self.save_results(res_path, res_data)
            progress_bar.update(1)

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}:_process_documents] Done docs -> {res_path}")

    def get_res_schema(self, res_schema_path):
        """ Load Result Schema from <res_schema_path> """
        if not os.path.exists(res_schema_path):
            logging.error(f"[{self.__class__.__name__}:get_res_schema] Result Schema not found: {res_schema_path}")
            exit()
        return self.load_json(res_schema_path)
    
    @staticmethod
    def _extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from raw text.
        Args: raw_text: Raw text that may contain JSON content
        Returns: Parsed JSON object as dictionary, or None if no valid JSON found
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

