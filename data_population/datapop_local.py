# data_population/datapop_local.py
"""
GPU / local LLM model for data population.
"""

from __future__ import annotations

import os
import re
import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
import string

import torch
if torch.cuda.is_available():
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import constants
from data_loader.data_loader_spider import SpiderDatasetLoader
from data_population.datapop_deepseek import DataPopDeepSeek

__all__ = ["DataPopLocal"]


class DataPopLocal(DataPopDeepSeek):

    def __init__(self, config):
        super().__init__(config)
        # no API key is required for local models.

        if not torch.cuda.is_available():
            logging.error(f"[{self.__class__.__name__}] CUDA unavailable. Exiting...")
            exit()

        self.mode = config["mode"]
        if self.mode in ["ds7b", "dsv2lite", "cogito32b", "cogito70b"]:
            self.llm_model_path = config["llm_model_path"]
            self.llm_model_name = config["llm_model"]
            self.param_str = config["res_param_str"]

            self.tokenizer = self.model = None
            self.load_model()

            self.prompt_table = self._load_prompt(config["prompts"]["prompt_table"])
            self.prompt_attr = self._load_prompt(config["prompts"]["prompt_attr"])

    def load_model(self):
        """ Load the LLM weights from disk or download if missing. """
        if not os.path.exists(self.llm_model_path):
            logging.info(f"[{self.__class__.__name__}:load_model] Downloading model ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name, trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.bfloat16,
                # device_map="auto",
                trust_remote_code=True, 
                # offload_folder="offload/",
            ).cuda()
            # cache for further runs
            self.tokenizer.save_pretrained(self.llm_model_path)
            self.model.save_pretrained(self.llm_model_path)
        else:
            logging.info(f"[{self.__class__.__name__}:load_model] Loading model from local ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_path, trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_path,
                torch_dtype=torch.bfloat16,
                # device_map="auto",
                trust_remote_code=True, 
                # offload_folder="offload/",
            ).cuda()
        if self.mode in ["ds7b", "dsv2lite"]:
            self.model.generation_config = GenerationConfig.from_pretrained(self.llm_model_name)
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def _process_dataset(self, dataset_root: str | Path):
        """ Process all queries for one dataset. 

        The folder must follow the Spider layout - 
        see :pyclass:`SpiderDatasetLoader` for details.
        """
        dataset_root = Path(dataset_root)
        loader = SpiderDatasetLoader(dataset_root)
        query_dict = loader.load_query_dict()
        schema_general = loader.load_schema_general()

        for qid in query_dict:
            schema_query = loader.load_schema_query(qid)
            res_path = dataset_root / f"res_tabular_data_{qid}_{self.param_str}.json"
            hs_dir = dataset_root / f"hidden_states_{qid}_{self.param_str}"
            hs_dir.mkdir(parents=True, exist_ok=True)

            res_data = self.load_processed_res(res_path)
            pgbar_name = f"{dataset_root.name}-{qid}"

            logging.info(f"[{self.__class__.__name__}] Start processing query-{qid}")
            logging.info(f"[{self.__class__.__name__}] Writing results to: {res_path}")
            self._process_documents(
                loader, schema_general, schema_query, res_data, res_path, hs_dir, pgbar_name
            )
    
    def _process_documents(
            self, 
            loader: SpiderDatasetLoader,
            schema_general: List[Dict[str, Any]],
            schema_query: List[Dict[str, Any]],
            res_data: Dict[str, Any],
            res_path: Path,
            hs_dir: Path,
            pgbar_name: str,
            max_table_retries: int = 10,
            max_attr_retries: int = 10,
        ):
        """ Iterate over documents and populate table/attribute data. """
        all_tables = [s["Schema Name"] for s in schema_general]
        table2schema = {s["Schema Name"]: s for s in schema_general}
        table2attr: Dict[str, List[str]] = {
            s["Schema Name"]: [a["Attribute Name"] for a in s["Attributes"]]
            for s in schema_query
            if s["Schema Name"] in all_tables
        }
        attr_general = []
        for tn in table2attr:
            attrs = [a["Attribute Name"] for a in table2schema[tn]["Attributes"]]
            attr_general.append({"Schema Name": tn, "Attributes": attrs})

        num_doc = loader.num_docs
        table_attempt = attr_attempt = 0
        progress_bar = tqdm(total=num_doc, desc=f"Processing {pgbar_name}")

        for did, (doc_text, *_rest) in zip(loader.doc_ids, loader.iter_docs()):
            if did in res_data:
                progress_bar.update(1)
                continue

            failed = False
            # ----------- Table assignment -------------------------
            table_attempt = 0
            while True:
                if table_attempt > max_table_retries:
                    logging.info(f"[{self.__class__.__name__}] Table fail "
                                 f">{max_table_retries}x for doc {did}. Skipping.")
                    failed = True
                    break
                tbl_input = {
                    "Document": doc_text,
                    "Schema": attr_general,
                }
                tbl_input = json.dumps(tbl_input, ensure_ascii=False)
                raw_text, token_info = self.llm_generate(self.prompt_table, tbl_input)
                res_tbl, ts, te = self._extract_json_block(raw_text, token_info)
                if not res_tbl or "Table Assignment" not in res_tbl:
                    table_attempt += 1
                    continue
                table_name = res_tbl["Table Assignment"]
                if table_name not in all_tables:
                    table_attempt += 1
                    continue
                # Try to save span hidden‑states; retry generation if span mismatch
                if self._save_span_hs(table_name, token_info[ts:te], hs_dir / f"doc-{did}-table.pt"):
                    break  # success
                table_attempt += 1
            if failed:
                progress_bar.update(1)
                continue  # doc failed
            result_entry: Dict[str, Any] = {"res": table_name, "data": {}}

            # ----------- Attribute extraction per attribute -------
            for attr in table2attr[table_name]:
                attr_attempt = 0
                while True:
                    if attr_attempt > max_attr_retries:
                        logging.info(f"[{self.__class__.__name__}] Attr {attr} fail "
                                     f">{max_attr_retries}x doc {did}")
                        failed = True
                        break
                    attr_input = {
                        "Document": doc_text,
                        "Schema": table2schema[table_name],
                        "Target Attribute": attr,
                    }
                    attr_input = json.dumps(attr_input, ensure_ascii=False)
                    raw_text, token_info = self.llm_generate(self.prompt_attr, attr_input)
                    res_attr, ts, te = self._extract_json_block(raw_text, token_info)
                    if not res_attr or attr not in res_attr:
                        logging.info(f"[{self.__class__.__name__}] Attr {attr} not found in "
                                     f"response. raw_text: {raw_text}")
                        attr_attempt += 1
                        continue
                    attr_val = str(res_attr[attr])
                    if len(attr_val) > 100:
                        logging.info(f"[{self.__class__.__name__}] Attribute data too long: "
                                     f"{attr_val}, please check it")
                    # Try save span; regenerate if mismatch
                    if self._save_span_hs(attr_val, token_info[ts:te], hs_dir / f"doc-{did}-attr-{attr}.pt"):
                        result_entry["data"][attr] = attr_val
                        break
                    logging.info(f"[{self.__class__.__name__}] Span mismatch for attr {attr}. "
                                 f"raw_text: {raw_text}; attr_val: {attr_val}")
                    attr_attempt += 1
                if attr_attempt > max_attr_retries:
                    break  # abandon rest attrs
            if failed:
                progress_bar.update(1)
                continue

            # Executed only if *no* break occurred – all attrs succeeded
            res_data[did] = result_entry
            self.save_results(res_path, res_data)
            progress_bar.update(1)

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}] Done → {res_path}")
    
    def llm_generate(self, prompt: str, msg: str):
        """ Generate text using the LLM model. """
        messages = [{"role": "user", "content": prompt + "\n\n" + msg}]
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_tensor,
                max_new_tokens=1000,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
            )
        gen_tokens = outputs.sequences[0][input_tensor.shape[1]:]
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # ---------- get hidden states ------------
        all_token_hiddens = torch.stack([
            torch.stack([layer_h[:, -1, :].cpu()           # (num_layers, hidden)
                        for layer_h in step_hiddens], 0)
            for step_hiddens in outputs.hidden_states      # len == #generated tokens
        ])                                                 # -> (num_tokens,num_layers,hidden)

        # ------------- get scores ----------------
        # outputs.scores: list[T[vocab]]
        token_scores = torch.stack([s.squeeze(0).cpu() for s in outputs.scores])
        token_probs = torch.softmax(token_scores, dim=-1)

        token_info_pairs = []
        for step, (tok_id, hidden_states) in enumerate(zip(gen_tokens, all_token_hiddens)):
            token_text = self.tokenizer.decode(tok_id, skip_special_tokens=True)
            info = {
                "token_id":   tok_id.item(),
                "token_text": token_text.encode("utf-8"),
                "hidden_states": hidden_states,  # shape: (num_layers, hidden_size)
                # "logit":  token_scores[step, tok_id].item(),
                "prob":   token_probs[step, tok_id].item(),
            }
            token_info_pairs.append(info)
                    
        return gen_text, token_info_pairs

    # ------------------ utils -------------------

    def _save_span_hs(
            self, 
            span_text: str, 
            token_info_pairs: List[Dict[str, Any]], 
            out_path: Path
        ) -> bool:
        """
        Attempt to save the hidden states corresponding to span_text.
        First try exact token-ID matching; if that fails, fall back to
        concatenated token_text matching to handle merged-punctuation cases.
        """
        # 1) exact token-ID matching
        target_ids = self.tokenizer.encode(span_text, add_special_tokens=False)
        gen_ids = [info["token_id"] for info in token_info_pairs]
        for idx in range(len(gen_ids) - len(target_ids) + 1):
            if gen_ids[idx : idx + len(target_ids)] == target_ids:
                torch.save(token_info_pairs[idx : idx + len(target_ids)], out_path)
                return True

        # 2) concatenated token_text matching
        t_texts = [info["token_text"].decode("utf-8") for info in token_info_pairs]
        n = len(t_texts)
        for start in range(n):
            acc = ""
            for end in range(start, n):
                acc += t_texts[end]
                if self._compare_values(acc, span_text):
                    torch.save(token_info_pairs[start : end + 1], out_path)
                    return True
                if len(self.strip_punct(acc)) > len(self.strip_punct(span_text)):
                    break

        return False
    
    def _compare_values(self, val1: str, val2: str) -> bool:
        """ Compare two values for equality. """
        def is_null(val):
            if val is None or val in ["", "null", "NULL", "None", "none"]:
                return True
            return False

        val1 = self.strip_punct(val1)
        val2 = self.strip_punct(val2)
        if is_null(val1) and is_null(val2):
            return True
        return val1 == val2
    
    @staticmethod
    def strip_punct(text):
        # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        # punct = string.punctuation
        punct = "\"'\n\t "
        res = text.lstrip(punct).rstrip(punct)
        res = res.replace("\\\"", "")
        res = res.replace("\"", "")
        return 

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r") as fp:
            return fp.read()

    @staticmethod
    def save_tensor_dict(file_path, data_dict):
        torch.save(data_dict, file_path)

    @staticmethod       
    def _extract_json_block_old(raw_text):
        # Step 1: extract ```json ... ``` block
        match = re.search(r'```json(.*?)```', raw_text, re.S | re.I)
        if match:
            content = match.group(1).strip()
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(content)
                except Exception:
                    continue

        # Step 2: extract json-like patterns
        json_like_pattern = r'\{.*?\}'
        candidates = re.findall(json_like_pattern, raw_text, re.S)
        for candidate in candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    result = parser(candidate)
                    if isinstance(result, dict):
                        return result
                except Exception:
                    continue

        # Step 3: try to parse the entire raw_text as JSON
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(raw_text.strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                continue

        return None
    
    @staticmethod
    def _extract_json_block(raw_text: str, token_info_pairs: list):
        """
        Extract JSON object from raw_text and locate its start and end index in token_info_pairs.

        Returns:
            json_obj: parsed dictionary object (or None if not found)
            start_idx: index in token_info_pairs where JSON starts
            end_idx: index in token_info_pairs where JSON ends (exclusive)
        """

        # Step 1: find JSON block in raw_text
        match = re.search(r'```json(.*?)```', raw_text, re.S | re.I)
        if not match:
            match = re.search(r'```(.*?)```', raw_text, re.S)

        json_str = None
        start_char, end_char = None, None

        if match:
            json_str = match.group(1).strip()
            start_char, end_char = match.start(1), match.end(1)
        else:
            # Step 2: find JSON-like patterns in raw_text
            for m in re.finditer(r'\{.*?\}', raw_text, re.S):
                candidate = m.group(0)
                for parser in (json.loads, ast.literal_eval):
                    try:
                        obj = parser(candidate)
                        if isinstance(obj, dict):
                            json_str = candidate
                            start_char, end_char = m.start(), m.end()
                            break
                    except Exception:
                        continue
                if json_str:
                    break

        if not json_str:
            return None, None, None  # no JSON found

        # Step 3: parse JSON string
        json_obj = None
        for parser in (json.loads, ast.literal_eval):
            try:
                json_obj = parser(json_str)
                if isinstance(json_obj, dict):
                    break
            except Exception:
                continue

        if json_obj is None:
            return None, None, None

        # Step 4: map start and end characters to token_info_pairs
        char_count = 0
        start_idx = end_idx = None
        for idx, tok in enumerate(token_info_pairs):
            char_count += len(tok["token_text"])
            if start_idx is None and char_count > start_char:
                start_idx = idx
            if start_idx is not None and char_count >= end_char:
                end_idx = idx + 1
                break

        return json_obj, start_idx, end_idx
        