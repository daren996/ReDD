import os
import json
import logging
import traceback
from pathlib import Path
from tqdm import tqdm

from ..utils import constants
from ..utils.constants import PATH_TEMPLATES
from .schemagen_basic import SchemaGenBasic
from ..utils.prompt_utils import PromptGPT, get_api_key
from ..utils import output_utils
from ..doc_clustering.doc_clustering import DocumentClustering
from ..data_loader import create_data_loader
from .opt import AdaptiveSamplingMixin


class SchemaGenGPT(AdaptiveSamplingMixin, SchemaGenBasic):
    
    def __init__(self, config, api_key=None):
        super().__init__(config)

        self.config = config
        self.mode = config["mode"]
        
        # Data loader configuration
        self.loader_type = config.get("data_loader_type", "spider")
        self.loader_config = config.get("data_loader_config", {})
        
        # Initialize adaptive sampling
        self.init_adaptive_sampling(config)
        
        if self.mode == "cgpt":
            self.log_init_file = config.get("log_init_file")
            self.doc_cluster_file = config.get("doc_cluster_file")
            
            prompt_config = config["prompt"]
            self.param_str = config["res_param_str"]
            self.general_param_str = config.get("general_param_str")

            api_key = get_api_key(config, self.mode, api_key)
            self.prompt = PromptGPT(
                    self.mode,
                    prompt_config["prompt_path"],
                    llm_model=config["llm_model"], 
                    api_key=api_key
                )
    
    def __call__(self, dn_fn_list=None):
        """
        Extract attributes from documents in a dataset.

        Write the result log and generate query-specific schema. 
        Format:
        [
            {
                "Schema Name": <table_name>,
                "Attributes": [
                    {<attribute_name>: <explanation>},
                    ...
                ]
            },
            ...
        ]
        """
        if dn_fn_list is None:
            if "exp_dn_fn_list" in self.config:
                dn_fn_list = self.config["exp_dn_fn_list"]
            else:
                logging.warning(f"[{self.__class__.__name__}:__call__] No datasets specified, using default SPIDER_DN_FN_LIST")
                dn_fn_list = constants.SPIDER_DN_FN_LIST
        total_datasets = len(dn_fn_list)
        logging.info(f"[{self.__class__.__name__}:__call__] Schema Generation {total_datasets} datasets: {dn_fn_list}")

        for dn_fn in dn_fn_list:
            # Separate data path and output path
            data_root = Path(self.config.get("data_main", self.config["out_main"])) / dn_fn
            out_root = Path(self.config["out_main"]) / dn_fn
            out_root.mkdir(parents=True, exist_ok=True)
            logging.info(f"[{self.__class__.__name__}:__call__] Start processing dataset: data_root={data_root}, out_root={out_root}")
            
            if "query" in self.config["in_fields"]:
                self._process_dataset(data_root, out_root)
            else:
                self._process_dataset_no_query(data_root, out_root)
    
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
        
        # Load queries and general schema using data loader
        query_dict = self.loader.load_query_dict()
        
        if not query_dict:
            logging.warning(f"[{self.__class__.__name__}:_process_dataset] No queries found in dataset {self.data_root}")
            return
        
        # Build doc_dict for backward compatibility (legacy code needs this)
        doc_dict = self._build_doc_dict()
        
        general_schema = self.get_general_schema(self.out_root, doc_dict)
        doc_cluster = self.doc_clustering(self.out_root, doc_dict)
        
        logging.info(f"[{self.__class__.__name__}:_process_dataset] Processing {len(query_dict)} queries in dataset {self.data_root.name}")
        
        for qid in query_dict:
            query = query_dict[qid]["query"]
            res_path = self.out_root / PATH_TEMPLATES.schema_gen_result_query(qid, self.param_str)
            res_dict = self.load_processed_res(res_path)
            log_init = self.load_log_init(self.out_root, qid)
            general_schema_q = self.get_general_schema(self.out_root, doc_dict, qid, query)
            pgbar_name = f"{self.data_root.name}-{qid}"

            # Use adaptive sampling if enabled, otherwise use standard processing
            if self.adaptive_enabled:
                self.process_documents_adaptive(doc_dict, query, res_dict, log_init, general_schema_q, res_path, pgbar_name)
            else:
                self.process_documents(doc_dict, query, res_dict, log_init, general_schema_q, res_path, pgbar_name)
            self.tailor_schema(self.out_root, doc_dict, qid, query)

    def _process_dataset_no_query(self, data_root: str | Path, out_root: str | Path):
        """Process all documents in a dataset without queries."""
        self.data_root = Path(data_root)
        self.out_root = Path(out_root)
        
        # Create data loader based on configuration
        self.loader = create_data_loader(
            data_root=self.data_root,
            loader_type=self.loader_type,
            loader_config=self.loader_config
        )
        logging.info(f"[{self.__class__.__name__}:_process_dataset_no_query] Created {self.loader.__class__.__name__} for {self.data_root}")
        
        # Build doc_dict for backward compatibility
        doc_dict = self._build_doc_dict()
        
        res_path = self.out_root / PATH_TEMPLATES.schema_gen_result_general(self.param_str)
        res_dict = self.load_processed_res(res_path)
        pgbar_name = f"{self.data_root.name}"

        # Use adaptive sampling if enabled, otherwise use standard processing
        if self.adaptive_enabled:
            self.process_documents_adaptive(doc_dict, "", res_dict, dict(), None, res_path, pgbar_name)
        else:
            self.process_documents(doc_dict, "", res_dict, dict(), None, res_path, pgbar_name)
    
    def _build_doc_dict(self):
        """Build doc_dict from data loader for backward compatibility.
        
        Format: {doc_id: [doc_text, source_info], ...}
        """
        doc_dict = {}
        for doc_text, doc_id, metadata in self.loader.iter_docs():
            # Get source info (could be filename, table name, etc.)
            source_info = metadata.get("source_file") or metadata.get("table_name") or ""
            doc_dict[str(doc_id)] = [doc_text, source_info]
        
        logging.info(f"[{self.__class__.__name__}:_build_doc_dict] Built doc_dict with {len(doc_dict)} documents")
        return doc_dict

    def process_documents(self, doc_dict, query, res_dict, log_init, general_schema, res_path, pgbar_name):
        """ Process individual documents in the dataset. """
        num_doc = len(doc_dict)
        i, cnt, progress_bar = 0, 0, tqdm(total=num_doc, desc=f"Processing {pgbar_name}")

        logging.info(f"[{self.__class__.__name__}:process_documents] Start processing query: {query}")
        logging.info(f"[{self.__class__.__name__}:process_documents] Start processing documents: {res_path}")
        logging.info(f"[{self.__class__.__name__}:process_documents] Processed documents: {len(res_dict)} / {len(doc_dict)}")
        while i < num_doc:
            if str(i) in res_dict:
                i, cnt = i + 1, 0
                progress_bar.update(1)
                continue

            log = log_init if i == 0 else res_dict[str(i-1)]["log"]
            input_json = self.prepare_input_json(doc_dict, i, query, log, general_schema)
            out_dict = self.process_single_document(input_json, cnt, i)
            result_data = self.extract_result_data(out_dict)

            if not result_data or len(result_data["log"]) < len(log):
                cnt += 1
                if cnt > 10:
                    if not result_data:
                        logging.warning(f"[{self.__class__.__name__}:process_documents] Failed to process document {i} after {cnt} retries!")
                    else:
                        logging.warning(f"[{self.__class__.__name__}:process_documents] Schema num decrease, retry_count {cnt}, doc_index {i}")
                    exit()
                continue

            res_dict[str(i)] = result_data
            self.save_results(res_path, res_dict)

            i, cnt = i + 1, 0
            progress_bar.update(1)
            logging.info(f"[{self.__class__.__name__}:process_documents] Processed document {i} / {num_doc}")

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}:process_documents] Finished processing documents in {res_path}")
    
    def process_single_document(self, input_json, retry_count, doc_index):
        """ Process a single document and handle errors. """
        attr_msg = "New Input:\n" + json.dumps(input_json)
        result_str = self.apply_prompt(attr_msg)
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            logging.warning(f"[{self.__class__.__name__}:process_single_document] JSON LOAD ERROR, retry_count {retry_count}, "
                            f"doc_index {doc_index}, {repr(result_str)}")
            return None
    
    def apply_prompt(self, attr_msg):
        return self.prompt(msg=attr_msg).strip()

    def prepare_input_json(self, doc_dict, doc_index, query, log, general_schema):
        """ 
        Prepare the input JSON for a single document.
        """
        input_info = {
            "doc": doc_dict[str(doc_index)][0], 
            "query": query,
            "log": log,
            "general_schema": general_schema,
            "doc_cluster": None,  # os.path.splitext(doc_dict[str(doc_index)][1])[0]
            "all_clusters": None
        }
        input_json = {}
        for info_key, json_field in self.config["in_fields"].items():
            if info_key not in input_info:
                logging.error(f"[{self.__class__.__name__}:prepare_input_json] Error: input info key <{info_key}> not supported!")
                exit()
            input_json[json_field] = input_info[info_key]
        return input_json
        
    def extract_result_data(self, out_dict):
        if not out_dict:
            return None
        result_data = dict()
        for res_key, json_field in self.config["out_fields"].items():
            if json_field not in out_dict:
                logging.warning(f"[{self.__class__.__name__}:extract_result_data] output field <{json_field}> not found in out_dict!")
                return None
            result_data[res_key] = out_dict[json_field]
        return result_data
    
    def get_general_schema(self, out_root, doc_dict, qid=None, query=None):
        """Get the general schema (optionally query-specific)."""
        out_root = Path(out_root)
        if "general_schema" not in self.config["in_fields"]:
            return None
        schema_path = out_root / PATH_TEMPLATES.schema_general(self.general_param_str, qid)
        if not schema_path.exists():
            logging.info(f"[{self.__class__.__name__}:get_general_schema] General Schema Not Found in {schema_path}")
            try:
                res_dict = self.load_json(out_root / PATH_TEMPLATES.schema_gen_result_general(self.general_param_str))
            except FileNotFoundError:
                logging.error(f"[{self.__class__.__name__}:get_general_schema] {PATH_TEMPLATES.schema_gen_result_general(self.general_param_str)} Not Found!")
                exit(2)
            logging.info(f"[{self.__class__.__name__}:get_general_schema] Start Extracting General Schema ...")
            output_utils.create_general_schema(self.config, res_dict, doc_dict, str(out_root), self.general_param_str, qid, query)
            logging.info(f"[{self.__class__.__name__}:get_general_schema] General Schema Extracted in {schema_path}")
        logging.info(f"[{self.__class__.__name__}:get_general_schema] General Schema Loaded from {schema_path}")
        return self.load_json(schema_path)
    
    def tailor_schema(self, out_root, doc_dict, qid, query):
        """Generate the tailored query-specific schema."""
        out_root = Path(out_root)
        schema_path = out_root / PATH_TEMPLATES.schema_query_tailored(qid, self.param_str)
        if not schema_path.exists():
            try:
                res_dict = self.load_json(out_root / PATH_TEMPLATES.schema_gen_result_query(qid, self.param_str))
            except FileNotFoundError:
                logging.error(f"[{self.__class__.__name__}:tailor_schema] {PATH_TEMPLATES.schema_gen_result_query(qid, self.param_str)} Not Found!")
                exit(2)
            logging.info(f"[{self.__class__.__name__}:tailor_schema] Start Tailoring Schema ...")
            output_utils.create_tailored_schema(self.config, res_dict, doc_dict, str(out_root), self.param_str, qid, query)
            logging.info(f"[{self.__class__.__name__}:tailor_schema] Tailored Schema Created in {schema_path}")
        else:
            logging.info(f"[{self.__class__.__name__}:tailor_schema] Tailored Schema Already Exists in {schema_path}")
    
    def doc_clustering(self, out_root, doc_dict):
        """Conduct document clustering for the dataset in `out_root/<doc_cluster_file>`."""
        out_root = Path(out_root)
        if self.doc_cluster_file:
            doc_cluster_path = out_root / self.doc_cluster_file
            if not doc_cluster_path.exists():
                logging.warning(f"[{self.__class__.__name__}:doc_clustering] Document Cluster Not Found in {self.doc_cluster_file}")
                logging.info(f"[{self.__class__.__name__}:doc_clustering] Start Document Clustering ...")
                documents = []
                for doc_id in doc_dict:
                    documents.append(doc_dict[doc_id][0])
                doc_clustering = DocumentClustering(documents, 2, None, None)
                cluster_ids = doc_clustering.cluster()
                # TODO: save doc cluster to `out_root/<doc_cluster_file>`
                logging.info(f"[{self.__class__.__name__}:doc_clustering] Document Cluster Generated in {self.doc_cluster_file}")
        return self.load_doc_cluster(out_root)  # TODO: use it

    def load_log_init(self, out_root, qid):
        """Load Log Init from `out_root/<log_init_file>`"""
        out_root = Path(out_root)
        log_init = dict()
        if self.log_init_file is not None:
            try:
                log_init = self.load_json(out_root / self.log_init_file.format(qid))
                logging.info(f"[{self.__class__.__name__}:load_log_init] Query-specific Prompt Log Init ({self.log_init_file}, "
                             f"len={len(log_init)}) Imposed for Log Init!")
            except FileNotFoundError:
                logging.warning(f"[{self.__class__.__name__}:load_log_init] Query-specific Prompt Log Init ({self.log_init_file}) "
                                f"Not Found!")
        else:
            logging.info(f"[{self.__class__.__name__}:load_log_init] No Query-specific Prompt Log Init!")
        return log_init
    
    def load_doc_cluster(self, out_root):
        """Load the document cluster results from `out_root/<doc_cluster_file>`."""
        out_root = Path(out_root)
        doc_cluster = dict()
        if self.doc_cluster_file is not None:
            try:
                doc_cluster = self.load_json(out_root / self.doc_cluster_file)
                logging.info(f"[{self.__class__.__name__}:load_doc_cluster] Document Cluster ({self.doc_cluster_file}, "
                             f"len={len(doc_cluster)}) Loaded!")
            except FileNotFoundError:
                logging.warning(f"[{self.__class__.__name__}:load_doc_cluster] Document Cluster ({self.doc_cluster_file}) Not Found!")
        else:
            logging.info(f"[{self.__class__.__name__}:load_doc_cluster] No Document Cluster!")
        return doc_cluster

    def __str__(self):
        return f"{self.__class__.__name__}: \n{self.param_str}\n{self.prompt}"
