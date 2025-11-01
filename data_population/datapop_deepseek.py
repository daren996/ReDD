# data_population/datapop_deepseek.py
"""
LLM model for data population using API / cloud inference.

Write the results to `<dataset_root>/res_tabular_data_{qid}.json`.
Format:
[
    <doc_id>: {
        "res": <table_name>,
        "data": [{<attribute_name>: <value>}, ...],
        "reason": "...",
    },
    ...
]
"""

from __future__ import annotations

import os
import re
import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

import constants
from data_loader.data_loader_spider import SpiderDatasetLoader
from data_population.datapop_basic import DataPopBasic
from utils.prompt_utils import PromptDeepSeek

__all__ = ["DataPopDeepSeek"]


class DataPopDeepSeek(DataPopBasic):
    
    def __init__(self, config, api_key=None):
        super().__init__(config)
        
        if config["mode"] in ["deepseek"]:
            if api_key:
                self.api_key = api_key
            elif "api_key" in config:
                self.api_key = config["api_key"]
            else: 
                logging.error("API key is required for DeepSeek mode.")
                exit()

            self.param_str = config["res_param_str"]
            self.prompt_table = PromptDeepSeek(
                    config["mode"],
                    config["prompts"]["prompt_table"],
                    llm_model=config["llm_model"], 
                    api_key=self.api_key
                )
            self.prompt_attr = PromptDeepSeek(
                    config["mode"],
                    config["prompts"]["prompt_attr"],
                    llm_model=config["llm_model"], 
                    api_key=self.api_key
                )

    def __call__(self, dn_list: List[str] | None = None, fn_list: List[str] | None = None):
        """
        Extract tabular data from the documents in the datasets.
        Args:
            - dn_list: list, dataset names, 
                e.g., ["store_1", "wine_1", "soccer_1", "college_2", "flight_4"]
            - fn_list: list, list of filenames in the dataset, 
                e.g., ["customers-invoices", "customers-employees"]

        # TODO: use config to determine the file names and formats.
        """
        dn_list = constants.SPIDER_DN_LIST if dn_list is None else dn_list
        for dn in dn_list:
            fn_list = constants.EXP_DN2FN[dn] if fn_list is None else fn_list
            fn_list = [exp_fn for exp_fn in fn_list if exp_fn in constants.EXP_DN2FN[dn]]
            for exp_fn in fn_list:
                dataset_root = Path(self.config["out_main"]) / dn / exp_fn
                dataset_root.mkdir(parents=True, exist_ok=True)
                logging.info(f"[{self.__class__.__name__}] Start processing dataset: {dataset_root}")
                self._process_dataset(dataset_root)

    def _process_dataset(self, dataset_root: str | Path):
        """ Process all queries and documents in a dataset. """
        loader = SpiderDatasetLoader(dataset_root)
        query_dict = loader.load_query_dict()
        schema_general = loader.load_schema_general()

        for qid in query_dict:
            schema_query = loader.load_schema_query(qid)
            res_path = dataset_root / f"res_tabular_data_{qid}_{self.param_str}.json"
            res_data = self.load_processed_res(res_path)
            pgbar_name = f"{dataset_root.name}-{qid}"

            logging.info(f"[{self.__class__.__name__}] Start processing query-{qid} -> {res_path}")
            self._process_documents(
                loader=loader,
                schema_general=schema_general,
                schema_query=schema_query,
                res_data=res_data,
                res_path=res_path,
                pgbar_name=pgbar_name,
            )

    def _process_documents(
        self,
        *,
        loader: SpiderDatasetLoader,
        schema_general: List[Dict[str, Any]],
        schema_query: List[Dict[str, Any]],
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
    ) -> None:
        """ Iterate over documents and populate table/attribute data. """
        all_tables = [s["Schema Name"] for s in schema_general]
        table2schema = {s["Schema Name"]: s for s in schema_general}
        table2attr: Dict[str, List[str]] = {
            s["Schema Name"]: [a["Attribute Name"] for a in s["Attributes"]]
            for s in schema_query
            if s["Schema Name"] in all_tables
        }

        progress_bar = tqdm(total=loader.num_docs, desc=f"Processing {pgbar_name}")
        processed = 0
        for did, (doc_text, *_rest) in zip(loader.doc_ids, loader.iter_docs()):
            if did in res_data:
                progress_bar.update(1)
                continue

            # ‑‑ table assignment -----------------------------------------------------
            tbl_input = {"Document": doc_text, "Schema": schema_general}
            table_assigned = self.prompt_table(msg=json.dumps(tbl_input), response_format="text").strip()
            if table_assigned != "None" and table_assigned not in all_tables:
                logging.info(f"[{self.__class__.__name__}] Table {table_assigned} not in schema - skip doc {did}")
                progress_bar.update(1)
                continue

            result_entry = {"res": table_assigned, "data": {}}

            # ‑‑ attribute extraction -------------------------------------------------
            for attr in table2attr.get(table_assigned, []):
                attr_input = {
                    "Document": doc_text,
                    "Schema": table2schema[table_assigned],
                    "Target Attribute": attr,
                }
                attr_val = self.prompt_attr(msg=json.dumps(attr_input), response_format="text").strip()
                if len(attr_val) > 100:
                    logging.warning(f"[{self.__class__.__name__}] Attr too long "
                                    f"(>{len(attr_val)}): doc {did} attr {attr}")
                result_entry["data"][attr] = attr_val

            res_data[did] = result_entry
            self.save_results(res_path, res_data)
            progress_bar.update(1)
            processed += 1

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}] Done docs -> {res_path} (+{processed})")

    def get_res_schema(self, res_schema_path):
        """ Load Result Schema from <res_schema_path> """
        if not os.path.exists(res_schema_path):
            logging.error(f"[{self.__class__.__name__}:get_res_schema] Result Schema not found: {res_schema_path}")
            exit()
        return self.load_json(res_schema_path)
    