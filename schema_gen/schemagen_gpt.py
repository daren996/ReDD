import os
import json
import logging
from tqdm import tqdm

import constants
from schema_gen.schemagen_basic import SchemaGenBasic
from utils.prompt_utils import PromptGPT
from utils import output_utils
from doc_clustering.doc_clustering import DocumentClustering


class SchemaGenGPT(SchemaGenBasic):
    
    def __init__(self, config):
        super().__init__(config)
        if config["mode"] == "cgpt":
            try:
                self.log_init_file = config["cgpt"]["log_init_file"] if "log_init_file" in config["cgpt"] else None
                self.doc_cluster_file = config["doc_cluster_file"] if "doc_cluster_file" in config else None
                param_str_tmp = "mdl{model}_prm%s_tmp{temp}_tpp{top_p}".format(
                    model=config["llm_model"], temp=config["cgpt"]["temperature"],
                    top_p=config["cgpt"]["top_p"]                 
                )
                self.param_str = param_str_tmp % config["cgpt"]["prompt_version"]
                self.general_param_str = param_str_tmp % config["cgpt"]["general_prompt_version"] if "general_prompt_version" in config["cgpt"] else None
                self.prompt = PromptGPT(
                        config["mode"],
                        config["cgpt"]["prompt_path"],
                        llm_model=config["llm_model"], 
                        api_key=config["api_key"] if "api_key" in config else None
                    )
                self.temperature, self.top_p = config["cgpt"]["temperature"], config["cgpt"]["top_p"]
            except Exception as e:
                logging.error(f"When initializing {self.__class__.__name__}, Error: {e}")
                exit()
    
    def __call__(self, dn_list=None, fn_list=None):
        """
        Extract attributes from documents in a dataset.
        Args:
            - dn_list: list, dataset names, 
                e.g., ["store_1", "wine_1", "soccer_1", "college_2", "flight_4"]
            - fn_list: list, list of filenames in the dataset, 
                e.g., ["customers-invoices", "customers-employees"]
        Write the result log to `<out_dn>/res_{qid}_{self.param_str}.json`.
        Generate query-specific schema in `<out_dn>/res_schema_{qid}_{self.param_str}.json`. 
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

        # TODO: use config to determine the file names and formats.
        """
        dn_list = constants.SPIDER_DN_LIST if dn_list is None else dn_list
        for dn in dn_list:
            fn_list = constants.EXP_DN2FN[dn] if not fn_list else fn_list
            fn_list = [exp_fn for exp_fn in fn_list if exp_fn in constants.EXP_DN2FN[dn]]
            for exp_fn in fn_list:
                out_dn = os.path.join(self.config["out_main"], f"{dn}/{exp_fn}")
                if exp_fn == "ALL":
                    out_dn = os.path.join(self.config["out_main"], dn)
                logging.info(f"[{self.__class__.__name__}] Start processing dataset: {out_dn}")
                if "query" in self.config["in_fields"]:
                    self.process_dataset(out_dn)
                else:
                    self.process_dataset_no_query(out_dn)
    
    def process_dataset(self, out_dn):
        """ Process all queries and documents in a dataset. """
        doc_dict = self.load_json(os.path.join(out_dn, "doc_dict.json"))
        query_dict = self.load_json(os.path.join(out_dn, "queries_drc.json"))
        general_schema = self.get_general_schema(out_dn, doc_dict)
        doc_cluster = self.doc_clustering(out_dn, doc_dict)
        
        for qid in query_dict:
            query = query_dict[qid]["query"]
            res_path = os.path.join(out_dn, f"res_{qid}_{self.param_str}.json")
            res_dict = self.load_processed_res(res_path)
            log_init = self.load_log_init(out_dn, qid)
            general_schema_q = self.get_general_schema(out_dn, doc_dict, qid, query)
            pgbar_name = f"{out_dn}-{qid}"

            self.process_documents(doc_dict, query, res_dict, log_init, general_schema_q, res_path, pgbar_name)
            self.tailor_schema(out_dn, doc_dict, qid, query)

    def process_dataset_no_query(self, out_dn):
        """ Process all queries and documents in a dataset. """
        doc_dict = self.load_json(os.path.join(out_dn, "doc_dict.json"))
        
        res_path = os.path.join(out_dn, f"res_{self.param_str}.json")
        res_dict = self.load_processed_res(res_path)
        pgbar_name = f"{out_dn}"

        self.process_documents(doc_dict, "", res_dict, dict(), None, res_path, pgbar_name)

    def process_documents(self, doc_dict, query, res_dict, log_init, general_schema, res_path, pgbar_name):
        """ Process individual documents in the dataset. """
        num_doc = len(doc_dict)
        i, cnt, progress_bar = 0, 0, tqdm(total=num_doc, desc=f"Processing {pgbar_name}")

        logging.info(f"[{self.__class__.__name__}] Start processing query: {query}")
        logging.info(f"[{self.__class__.__name__}] Start processing documents: {res_path}")
        logging.info(f"[{self.__class__.__name__}] Processed documents: {len(res_dict)} / {len(doc_dict)}")
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
                        logging.warning(f"[{self.__class__.__name__}] Failed to process document {i} after {cnt} retries!")
                    else:
                        logging.warning(f"[{self.__class__.__name__}] Schema num decrease, retry_count {cnt}, doc_index {i}")
                    exit()
                continue

            res_dict[str(i)] = result_data
            self.save_results(res_path, res_dict)

            i, cnt = i + 1, 0
            progress_bar.update(1)
            logging.info(f"[{self.__class__.__name__}] Processed document {i} / {num_doc}")

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}] Finished processing documents in {res_path}")
    
    def process_single_document(self, input_json, retry_count, doc_index):
        """ Process a single document and handle errors. """
        attr_msg = "New Input:\n" + json.dumps(input_json)
        result_str = self.apply_prompt(attr_msg)
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            logging.warning(f"[{self.__class__.__name__}] JSON LOAD ERROR, retry_count {retry_count}, "
                            f"doc_index {doc_index}, {repr(result_str)}")
            return None
    
    def apply_prompt(self, attr_msg):
        return self.prompt(
            msg=attr_msg, 
            temperature=self.temperature, 
            top_p=self.top_p
        ).strip()

    def prepare_input_json(self, doc_dict, doc_index, query, log, general_schema):
        """ 
        Prepare the input JSON for a single document. 
        TODO: generalize it
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
                logging.error(f"[{self.__class__.__name__}] Error: input info key <{info_key}> not supported!")
                exit()
            input_json[json_field] = input_info[info_key]
        return input_json
        
    def extract_result_data(self, out_dict):
        if not out_dict:
            return None
        result_data = dict()
        for res_key, json_field in self.config["out_fields"].items():
            if json_field not in out_dict:
                logging.warning(f"[{self.__class__.__name__}] output field <{json_field}> not found in out_dict!")
                return None
            result_data[res_key] = out_dict[json_field]
        return result_data
    
    def get_general_schema(self, out_dn, doc_dict, qid=None, query=None):
        """ Get the general schema from `<out_dn>/schema_general_{self.general_param_str}.json` or 
            `<out_dn>/schema_general_{qid}_{self.general_param_str}.json`. """
        if "general_schema" not in self.config["in_fields"]:
            return None
        if not query:
            schema_path = os.path.join(out_dn, f"schema_general_{self.general_param_str}.json")
        else:
            schema_path = os.path.join(out_dn, f"schema_general_{qid}_{self.general_param_str}.json")
        if not os.path.exists(schema_path):
            logging.info(f"[{self.__class__.__name__}:GeneralSchema] General Schema Not Found in {schema_path}")
            try:
                res_dict = self.load_json(os.path.join(out_dn, f"res_{self.general_param_str}.json"))
            except FileNotFoundError:
                logging.error(f"[{self.__class__.__name__}:GeneralSchema] res_{self.general_param_str}.json Not Found!")
                exit(2)
            logging.info(f"[{self.__class__.__name__}:GeneralSchema] Start Extracting General Schema ...")
            output_utils.create_general_schema(self.config, res_dict, doc_dict, out_dn, self.general_param_str, qid, query)
            logging.info(f"[{self.__class__.__name__}:GeneralSchema] General Schema Extracted in {schema_path}")
        logging.info(f"[{self.__class__.__name__}:GeneralSchema] General Schema Loaded from {schema_path}")
        return self.load_json(schema_path)
    
    def tailor_schema(self, out_dn, doc_dict, qid, query):
        """ Generate the tailored query-specific schema in `<out_dn>/res_schema_{qid}_{self.param_str}.json` """
        schema_path = os.path.join(out_dn, f"res_schema_{qid}_{self.param_str}.json")
        if not os.path.exists(schema_path):
            try:
                res_dict = self.load_json(os.path.join(out_dn, f"res_{qid}_{self.param_str}.json"))
            except FileNotFoundError:
                logging.error(f"[{self.__class__.__name__}:TailorSchema] res_{qid}_{self.param_str}.json Not Found!")
                exit(2)
            logging.info(f"[{self.__class__.__name__}:TailorSchema] Start Tailoring Schema ...")
            output_utils.create_tailored_schema(self.config, res_dict, doc_dict, out_dn, self.param_str, qid, query)
            logging.info(f"[{self.__class__.__name__}:TailorSchema] Tailored Schema Created in {schema_path}")
        else:
            logging.info(f"[{self.__class__.__name__}:TailorSchema] Tailored Schema Already Exists in {schema_path}")
    
    def doc_clustering(self, out_dn, doc_dict):
        """ Conduct document clustering for the dataset in `out_dn/<doc_cluster_file>`. """
        if self.doc_cluster_file:
            if not os.path.exists(os.path.join(out_dn, self.doc_cluster_file)):
                logging.warning(f"[{self.__class__.__name__}:DocCluster] Document Cluster Not Found in {self.doc_cluster_file}")
                logging.info(f"[{self.__class__.__name__}:DocCluster] Start Document Clustering ...")
                documents = []
                for doc_id in doc_dict:
                    documents.append(doc_dict[doc_id][0])
                doc_clustering = DocumentClustering(documents, 2, None, None)
                cluster_ids = doc_clustering.cluster()
                # TODO: save doc cluster to `out_dn/<doc_cluster_file>`
                logging.info(f"[{self.__class__.__name__}:DocCluster] Document Cluster Generated in {self.doc_cluster_file}")
        return self.load_doc_cluster(out_dn)  # TODO: use it

    def load_log_init(self, out_dn, qid):
        """ Load Log Init from `out_dn/<log_init_file>` """
        log_init = dict()
        if self.log_init_file is not None:
            try:
                log_init = self.load_json(os.path.join(out_dn, self.log_init_file.format(qid)))
                logging.info(f"[{self.__class__.__name__}:LogINIT] Query-specific Prompt Log Init ({self.log_init_file}, "
                             f"len={len(log_init)}) Imposed for Log Init!")
            except FileNotFoundError:
                logging.warning(f"[{self.__class__.__name__}:LogINIT] Query-specific Prompt Log Init ({self.log_init_file}) "
                                f"Not Found!")
        else:
            logging.info(f"[{self.__class__.__name__}:LogINIT] No Query-specific Prompt Log Init!")
        return log_init
    
    def load_processed_res(self, res_path):
        """ Load Processed Results from <res_path> """
        res_dict = dict()
        if os.path.exists(res_path):
            res_dict = self.load_json(res_path)
        return res_dict
    
    def load_doc_cluster(self, out_dn):
        """ Load the document cluster results from `out_dn/<doc_cluster_file>`. """
        doc_cluster = dict()
        if self.doc_cluster_file is not None:
            try:
                doc_cluster = self.load_json(os.path.join(out_dn, self.doc_cluster_file))
                logging.info(f"[{self.__class__.__name__}:DocCluster] Document Cluster ({self.doc_cluster_file}, "
                             f"len={len(doc_cluster)}) Loaded!")
            except FileNotFoundError:
                logging.warning(f"[{self.__class__.__name__}:DocCluster] Document Cluster ({self.doc_cluster_file}) Not Found!")
        else:
            logging.info(f"[{self.__class__.__name__}:DocCluster] No Document Cluster!")
        return doc_cluster

    def __str__(self):
        return f"{self.__class__.__name__}: \n{self.param_str}\n{self.prompt}"
