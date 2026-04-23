import json
import logging
import random
from pathlib import Path

from redd.exceptions import ArtifactNotFoundError, PromptExecutionError

from .constants import ASSIGN_THRESHOLD, PATH_TEMPLATES
from .prompt_utils import create_prompt


class ResDictName:
    def __init__(self):
        self.res = "res"
        self.log = "log"
        self.schema = "Schema Name"
        self.attr = "Attributes"


dict_name = ResDictName()


def load_json(file_path, encoding="utf-8"):
    with Path(file_path).open("r", encoding=encoding) as f:
        return json.load(f)


def save_results(out_path, out_dict, encoding="utf-8"):
    """Save results dictionary to JSON file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as f:
        json.dump(out_dict, f, indent=2)


def _run_schema_prompt(prompt, prompt_input_json, *, context: str, max_retries: int = 10):
    for attempt in range(1, max_retries + 1):
        try:
            result_str = prompt(json.dumps(prompt_input_json)).strip()
            payload = json.loads(result_str)
            updated_schema = payload["Updated Schema"]
            if not isinstance(updated_schema, list):
                raise TypeError(
                    f"`Updated Schema` must be a list, got {type(updated_schema).__name__}"
                )
            return updated_schema
        except Exception as error:
            logging.warning("[%s] prompt error on attempt %s/%s: %s", context, attempt, max_retries, error)

    raise PromptExecutionError(f"[{context}] prompt error: retry count exceeded {max_retries}")


def create_general_schema(config, res_dict, doc_dict, out_dn, param_str, qid=None, query=None):
    """
    Generate general schema based on the results of schema generation 
        using prompt `general_schema_revise_xxx.txt` to revise the schema.
    """
    if not res_dict:
        raise ArtifactNotFoundError("[create_general_schema] res_dict is empty")
    if len(res_dict) != len(doc_dict):
        logging.warning(f"[create_general_schema] res_dict and doc_dict mismatch: {len(res_dict)} != {len(doc_dict)}")
    
    # generate original general schema
    out_dir = Path(out_dn)
    schema_original_path = out_dir / PATH_TEMPLATES.schema_general_original(param_str)
    if not schema_original_path.exists():
        schema_all_doc = {}
        schema_assign = {}
        for doc_id in res_dict:
            if dict_name.log not in res_dict[doc_id]:
                raise ArtifactNotFoundError(
                    f"[create_general_schema] log not found in doc_id={doc_id}"
                )
            for schema in res_dict[doc_id][dict_name.log]:
                name = schema[dict_name.schema]
                attr = schema[dict_name.attr]
                schema_all_doc[name] = attr
            assign = res_dict[doc_id][dict_name.res]
            if assign not in schema_assign:
                schema_assign[assign] = 0
            schema_assign[assign] += 1
        logging.info(f"[create_general_schema] schema_assign: {schema_assign}")
        schema_general = {}
        for schema in schema_all_doc:
            if schema in schema_assign and schema_assign[schema] > ASSIGN_THRESHOLD:
                schema_general[schema] = schema_all_doc[schema]
        save_results(schema_original_path, schema_general)
    else:
        schema_general = load_json(schema_original_path)
    
    # use prompt `general_schema_revise_xxx.txt` to revise the schema
    schema_path = out_dir / PATH_TEMPLATES.schema_general(param_str, qid)
    if schema_path.exists():
        return
    schema_revise_prompt = create_prompt(
        config.get("mode", "cgpt"),
        "prompts/general_schema_revise_1_0.txt",
        llm_model=config.get("llm_model", "gpt-4o"),
        api_key=config.get("api_key"),
    )
    prompt_input_json = {
        "Schema": [], 
        "Example Query": query
    }
    for schema_name in schema_general:
        schema_docs = [doc_dict[doc_id][0] for doc_id in res_dict if res_dict[doc_id][dict_name.res] == schema_name]
        prompt_input_json["Schema"].append(
            {
                "Schema Name": schema_name,
                "Attributes": schema_general[schema_name],
                "Example Documents": random.sample(schema_docs, k=min(3, len(schema_docs)))
            }
        )
    schema_general_revised = _run_schema_prompt(
        schema_revise_prompt,
        prompt_input_json,
        context="create_general_schema",
    )
    save_results(schema_path, schema_general_revised)


def create_tailored_schema(config, res_dict, doc_dict, out_dn, param_str, qid, query):
    """
    Store the results of schema generation and tailor it using prompt `schema_tailor_xxx.txt`.
    """
    if not res_dict:
        raise ArtifactNotFoundError("[create_tailored_schema] res_dict is empty")
    if len(res_dict) != len(doc_dict):
        logging.warning(f"[create_tailored_schema] res_dict and doc_dict mismatch: {len(res_dict)} != {len(doc_dict)}")
    
    # generate original tailored schema
    out_dir = Path(out_dn)
    res_schema_path = out_dir / PATH_TEMPLATES.schema_query_original(qid, param_str)
    if not res_schema_path.exists():
        res_schema = res_dict[list(res_dict.keys())[-1]][dict_name.log]
        save_results(res_schema_path, res_schema)
    else:
        res_schema = load_json(res_schema_path)
    
    # use prompt `schema_tailor_xxx.txt` to tailor the schema
    tailored_schema_path = out_dir / PATH_TEMPLATES.schema_query_tailored(qid, param_str)
    if tailored_schema_path.exists():
        return
    schema_tailor_prompt = create_prompt(
        config.get("mode", "cgpt"),
        "prompts/schema_tailor_1_0.txt",
        llm_model=config.get("llm_model", "gpt-4o"),
        api_key=config.get("api_key"),
    )
    prompt_input_json = {
        "Schema": [], 
        "Query": query
    }
    for schema in res_schema:
        schema_docs = [doc_dict[doc_id][0] for doc_id in res_dict if res_dict[doc_id][dict_name.res] == schema[dict_name.schema]]
        prompt_input_json["Schema"].append(
            {
                "Schema Name": schema[dict_name.schema],
                "Attributes": schema[dict_name.attr],
                "Example Documents": random.sample(schema_docs, k=min(3, len(schema_docs)))
            }
        )
    schema_tailored = _run_schema_prompt(
        schema_tailor_prompt,
        prompt_input_json,
        context="create_tailored_schema",
    )
    save_results(tailored_schema_path, schema_tailored)
