import os
import json
import random
import logging

import constants


def data_init(config):
    """
    Prepare data for schema generation.
    If `doc_dict.json` does not exist, then:
        1. Read documents from all files in each dataset.
        2. Shuffle documents and create a new file to store the 
           documents in `<out_spider_path>/<dn>/doc_dict.json`.
        3. Copy `queries_drc.json` from the source to the output 
           directory for experiment datasets. 
           TODO: generalize it, implement data_loader
    """
    logging.info("Data Initialization for Schema Generation ...")
    os.makedirs(config["out_main"], exist_ok=True)

    # create `doc_dict.json` for each dataset
    for dn in constants.SPIDER_DN_LIST:
        create_doc_dict(config, dn, is_experiment=False)

    # create `doc_dict.json` for each experiment
    for dn in constants.SPIDER_DN_LIST:
        for exp_fn in constants.EXP_DN2FN[dn]:
            create_doc_dict(config, dn, exp_fn, is_experiment=True)
            copy_queries(config, dn, exp_fn)

    logging.info("Data Initialization Done!")

def create_doc_dict(config, dn, exp_fn=None, is_experiment=False):
    """ Process a dataset or an experiment-specific dataset. """
    out_path = os.path.join(config["out_main"], f"{dn}/{exp_fn}" if is_experiment else dn)
    os.makedirs(out_path, exist_ok=True)
    doc_dict_path = os.path.join(out_path, "doc_dict.json")
    if os.path.exists(doc_dict_path):
        with open(doc_dict_path, "r", encoding='utf-8') as f:
            doc_dict = json.load(f)
        logging.info(f"[Data Initialization] {len(doc_dict)} documents already generated in {doc_dict_path}")
        return

    # Gather document list
    doc_list = gather_documents(config, dn, exp_fn, is_experiment)
    # Shuffle and save documents
    random.shuffle(doc_list)
    doc_dict = {i: doc for i, doc in enumerate(doc_list)}
    with open(doc_dict_path, "w", encoding='utf-8') as f:
        json.dump(doc_dict, f, indent=2)
    logging.info(f"[Data Initialization] All documents shuffled and stored in {doc_dict_path}")

def copy_queries(config, dn, exp_fn):
    """ Copy `queries_drc.json` for each experiment datasets. """
    queries_src_path = os.path.join(config["spider_path"], dn, exp_fn, "queries_drc.json")
    queries_dest_path = os.path.join(config["out_main"], dn, exp_fn, "queries_drc.json")
    if os.path.exists(queries_dest_path):
        logging.info(f"[Data Initialization] When Copying Queries File ... {queries_dest_path} "
                     f"already exists and was not copied.")
        return
    if os.path.exists(queries_src_path):
        os.makedirs(os.path.dirname(queries_dest_path), exist_ok=True)
        with open(queries_src_path, "r", encoding='utf-8') as src, \
                open(queries_dest_path, "w", encoding='utf-8') as dest:
            dest.write(src.read())
        logging.info(f"[Data Initialization] When Copying Queries File ... Copied "
                     f"{queries_src_path} to {queries_dest_path}")
    else:
        logging.info(f"[Data Initialization] When Copying Queries File ... {queries_src_path} "
                     f"does not exist and was not copied.")

def gather_documents(config, dn, exp_fn=None, is_experiment=False):
    """ Gather documents from the source files. """
    doc_list = []
    dn_path = os.path.join(config["spider_path"], dn)

    if not is_experiment:
        for fn in constants.SPIDER_DN2FN[dn]:
            add_documents_from_file(doc_list, dn_path, fn)
    else:
        if "-" not in exp_fn:
            add_documents_from_file(doc_list, dn_path, exp_fn)
        else:
            for sub_fn in exp_fn.split("-"):
                add_documents_from_file(doc_list, dn_path, sub_fn)

    logging.info(f"[Data Initialization] {len(doc_list)} documents gathered from {dn} "
                 f"({'experiment' if is_experiment else 'dataset'})")
    return doc_list

def add_documents_from_file(doc_list, dn_path, fn):
    """ Add documents from a specific file to the document list. """
    file_path = os.path.join(dn_path, f"{fn}.json")
    with open(file_path, "r", encoding='utf-8') as f:
        id2doc = json.load(f)
        for doc_id in id2doc:
            doc_list.append((id2doc[doc_id], fn, doc_id))
    # file_path = os.path.join(dn_path, f"{fn}.txt")
    # with open(file_path, "r", encoding='utf-8') as f:
    #     for doc in f:
    #         doc_list.append((doc.strip(), f"{fn}.txt"))
    