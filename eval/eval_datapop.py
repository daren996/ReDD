import os
import json
import logging

import constants
from eval.eval_basic import EvalBasic
from utils.prompt_utils import PromptGPT, PromptDeepSeek
from data_loader.data_loader_spider import SpiderDatasetLoader


class EvalDataPopDeepSeek(EvalBasic):
    def __init__(self, config, data_loader=None, api_key=None):
        """ 
        Evaluate the data population results in DeepSeek mode.
        TODO: name map not implemented yet.
        TODO: doc_info_{qid}.json
        """
        super().__init__(config, data_loader)

        if config["eval"]["mode"] == "deepseek":
            if api_key:
                self.api_key = api_key
            elif "api_key" in config:
                self.api_key = config["eval"]["api_key"]
            else: 
                logging.error(f"[{self.__class__.__name__}] API key is required for DeepSeek mode.")
                exit()

            self.res_param_str = config["res_param_str"]
            self.prompts = {}
            for prompt_name in config["eval"]["prompts"]:
                self.prompts[prompt_name] = PromptDeepSeek(
                        config["eval"]["mode"],
                        config["eval"]["prompts"][prompt_name],
                        llm_model=config["eval"]["llm_model"], 
                        api_key=self.api_key
                    )
                # self.prompts[prompt_name] = PromptGPT(
                #         config["eval"]["mode"],
                #         config["eval"]["prompts"][prompt_name],
                #         llm_model=config["eval"]["llm_model"]
                #     )
            
    def __call__(self, dn_list=None, fn_list=None):
        dn_list = constants.SPIDER_DN_LIST if dn_list is None else dn_list

        for dn in dn_list:
            fn_list = constants.EXP_DN2FN[dn] if not fn_list else fn_list
            fn_list = [exp_fn for exp_fn in fn_list if exp_fn in constants.EXP_DN2FN[dn]]
            for exp_fn in fn_list:

                logging.info(f"[{self.__class__.__name__}] Start evaluating dataset: {dn}/{exp_fn}")
                prediction_dir = os.path.join(self.config["out_main"], f"{dn}/{exp_fn}")

                data_loader = self.data_loader or SpiderDatasetLoader(prediction_dir)

                query_dict = data_loader.load_query_dict()
                for qid in query_dict:
                    query = query_dict[qid]["query"]
                    res_data_path = os.path.join(prediction_dir, f"res_tabular_data_{qid}_{self.res_param_str}.json")
                    res_data_dict = self.load_json(res_data_path)

                    logging.info(f"[{self.__class__.__name__}] Start evaluating query {qid}: {query}")
                    self.prediction_data, self.gt_data = [], []
                    self.name_map = self.generate_mapping(prediction_dir, qid)

                    for doc_id in res_data_dict:
                        cur_info = data_loader.get_doc_info(doc_id)
                        if not cur_info:
                            logging.error(f"[{self.__class__.__name__}] doc-{doc_id} not found in ground truth.")
                            continue
                        self.prediction_data.append({
                            "doc_id": doc_id,
                            "table": res_data_dict[doc_id]["res"],
                            "data": res_data_dict[doc_id]["data"]
                        })
                        self.gt_data.append({
                            "doc_id": doc_id,
                            "table": cur_info["fn"],
                            "data": cur_info["data"]
                        })

                    _tp, _fp, _fn, _tn, _acc, _total, did2stat = self.compute_stat()
                    print(f"\n{exp_fn} {qid}")
                    print(f"TP: {_tp}, FP: {_fp}, FN: {_fn}, TN: {_tn}")
                    recall, precision, f1 = self.compute_recall_precision_f1(_tp, _fp, _fn)
                    print(f"Recall: {recall}, Precision: {precision}, F1: {f1}")
                    print(f"Accuracy: {_total - _acc} / {_total} = {_acc / _total}")

                    eval_path = os.path.join(prediction_dir, f"eval_{qid}_{self.res_param_str}.json")
                    self.save_results(eval_path, did2stat)
                    
    def compute_stat(self):
        """
        Compute the statistics of the evaluation.
        Format of pred and gt:
        {
            "table": "table_name",
            "data": {
                "column_name": "value",
                ...
            }
        }
        """

        if not self.prediction_data or not self.gt_data:
            logging.error(f"[{self.__class__.__name__}:compute_stat] No data loaded.")
            return None
        if len(self.prediction_data) != len(self.gt_data):
            logging.error(f"[{self.__class__.__name__}:compute_stat] Results and ground truth data have different lengths.")
            return None
        
        true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
        correct_num, total_num = 0, 0
        did2stat = {}  # doc_id -> {table: bool

        for pred, gt in zip(self.prediction_data, self.gt_data):
            total_num += 1
            is_correct = True
            did2stat[pred["doc_id"]] = {"table": True, "attr": {}}

            if self.is_null(gt["table"]):  # this doc is irrelevant to the query
                if not self.is_null(pred["table"]):  # assignment of this doc is not null
                    false_positives += len([attr for attr in pred["data"] if not self.is_null(pred["data"][attr])])
                    did2stat[pred["doc_id"]]["table"] = False
                    logging.info(f"[{self.__class__.__name__}:compute_stat] false_positives (doc irrelevant): {pred}; {gt}")
                    # TODO: more false_positives ???
                else:  # assignment of this doc is null
                    true_negatives += 1
            else:  # this doc is relevant to the query
                attr_map_dict = self.name_map["attribute"][gt["table"]]
                for attr in attr_map_dict:
                    attr_mapped = attr_map_dict[attr]
                    did2stat[pred["doc_id"]]["attr"][attr] = True

                    # get gt values
                    if isinstance(attr_mapped, list):  # one-multiple mappings
                        gt_value = " ".join([str(gt["data"][i]) for i in attr_mapped if gt["data"][i] is not None])
                        gt_attr = "-".join(attr_mapped)
                    else:  # one-one mapping
                        gt_value = str(gt["data"][attr_mapped])
                        gt_attr = attr_mapped
                    gt_value = gt_value.lower()
                    
                    if self.is_null(pred["table"]):  # assignment of this doc is null
                        if not self.is_null(gt_value):
                            false_negatives += 1
                            is_correct = False
                            did2stat[pred["doc_id"]]["table"] = False
                            logging.info(f"[{self.__class__.__name__}:compute_stat] false_negatives (doc null) of {attr}: {pred}; {gt}")
                    elif pred["table"] not in self.name_map["table"] or self.name_map["table"][pred["table"]] != gt["table"]: # table name unseen or not matched
                        if not self.is_null(gt_value):
                            false_negatives += 1
                            is_correct = False
                            did2stat[pred["doc_id"]]["table"] = False
                            logging.info(f"[{self.__class__.__name__}:compute_stat] false_negatives (table name incorrect): {pred}; {gt}")
                    
                    else:  # table names matched
                        if attr not in pred["data"]:  # attribute not in prediction
                            if not self.is_null(gt_value):
                                false_negatives += 1
                                is_correct = False
                                did2stat[pred["doc_id"]]["attr"][attr] = False
                                logging.info(f"[{self.__class__.__name__}:compute_stat] false_negatives (attribute not in prediction) of {attr}: {pred}; {gt}")
                        else:  # attribute in prediction
                            pred_value = str(pred["data"][attr]).lower()
                            if self.is_null(gt_value):
                                if self.is_null(pred_value):
                                    true_positives += 1
                                else:
                                    false_positives += 1
                                    did2stat[pred["doc_id"]]["attr"][attr] = False
                                    logging.info(f"[{self.__class__.__name__}:compute_stat] false_positives (gt_value null) of {attr}: {pred}; {gt}")
                            else:
                                if self.is_null(pred_value):
                                    false_negatives += 1
                                    is_correct = False
                                    did2stat[pred["doc_id"]]["attr"][attr] = False
                                    logging.info(f"[{self.__class__.__name__}:compute_stat] false_negatives (pred_value null) of {attr}: {pred}; {gt}")
                                elif pred_value == gt_value or gt_value in pred_value:
                                    true_positives += 1
                                elif pred_value != gt_value:
                                    semantical_equal = self.prompt_cmp_str(attr, pred_value, gt_attr, gt_value)
                                    if not semantical_equal:
                                        false_positives += 1
                                        is_correct = False
                                        did2stat[pred["doc_id"]]["attr"][attr] = False
                                        logging.info(f"[{self.__class__.__name__}:compute_stat] false_positives (semantical dismatch) of {attr}: {pred}; {gt}")
                                    else:
                                        true_positives += 1
                                else:
                                    true_positives += 1
                
                for attr in pred["data"]:
                    if attr not in attr_map_dict and not self.is_null(pred["data"][attr]):
                        false_positives += 1
                        logging.info(f"[{self.__class__.__name__}:compute_stat] false_positives (attribute not in gt) of {attr}: {pred}; {gt}")
                
            if is_correct:
                correct_num += 1
            did2stat[pred["doc_id"]]["final"] = is_correct

        return true_positives, false_positives, false_negatives, true_negatives, correct_num, total_num, did2stat

    def is_null(self, data_value):
        """
        Check if the data is null.
        Format:
        {
            "table": "table_name",
            "data": {
                "column_name": "value",
                ...
            }
        }
        """
        null_words = ["", "null", "none", "nan", "undisclosed", "unspecified", 
                      "unknown", "n/a", "na", "n.a.", "na.", "n/a.", "not available", 
                      "not applicable"]
        if data_value is None or str(data_value).strip().lower() in null_words:
            return True
        return False

    def prompt_cmp_str(self, pred_attr, pred_str, gt_attr, gt_data_str):
        """
        Compare the prediction and ground truth data (both in lower-case string) 
            using prompt `eval_datapop_cmp_str.txt`.
        Store cmp results in `cmp_results.json`.
        """
        cmp_results_path = os.path.join(self.config["out_main"], "cmp_results.json")
        cmp_results = {}
        if os.path.exists(cmp_results_path):
            cmp_results = self.load_json(cmp_results_path)

        if " -- ".join([pred_str, gt_data_str]) in cmp_results:
            return cmp_results[" -- ".join([pred_str, gt_data_str])]

        cmp_input_json = {
            "Prediction": {"Attribute Name": pred_attr, "Attribute Value": pred_str}, 
            "Ground Truth": {"Attribute Name": gt_attr, "Attribute Value": gt_data_str}
        }
        cmp_msg = json.dumps(cmp_input_json)
        while True:
            res_cmp = None
            try:
                res_cmp = json.loads(self.prompts["datapop_cmp_str"](msg=cmp_msg).strip())
                if res_cmp["Result"] in [True, False]:
                    break
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:prompt_cmp_str] Error: {e} \n\tPrompt Output: {res_cmp}")
        
        cmp_results[" -- ".join([pred_str, gt_data_str])] = res_cmp["Result"]
        self.save_results(cmp_results_path, cmp_results)
        logging.info(f"[{self.__class__.__name__}:prompt_cmp_str] Comparison of `{cmp_input_json}`: {res_cmp}")
        
        return res_cmp["Result"]

    def generate_mapping(self, prediction_dir, qid):
        """ 
        Generate the mapping between the prediction and ground truth data.
        Generated Table Name -> Ground Truth Table Name
        Generated Attribute Name -> Ground Truth Attribute Name
        Mapping saved in `name_map_{qid}.json`
        Format: 
        {
            "table": {
                "generated_table_name": "ground_truth_table_name",
                ...
            },
            "attribute": {
                "ground_truth_table_name": {
                    "generated_attribute_name": "ground_truth_attribute_name",
                    ...
                },
                ...
            }
        }
        """
        if not os.path.exists(os.path.join(prediction_dir, f"name_map_{qid}.json")):
            # create `name_map_{qid}.json`
            # TODO: implement the mapping generation
            pass
        return self.load_json(os.path.join(prediction_dir, f"name_map_{qid}.json"))
