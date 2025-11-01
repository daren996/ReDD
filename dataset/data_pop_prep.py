import os
import json
import random
import logging
import sqlite3
import pandas as pd

import constants


def data_init(config):
    """
    Prepare data for data population.
    Create `doc_info.json` for each dataset.
    TODO: Create `doc_info_{qid}.json` for each query: add `"fn"=None` for irrelevant doc.
    """
    logging.info("Data Initialization for Data Population ...")
    os.makedirs(config["out_main"], exist_ok=True)

    dn2fn2doc_list = read_doc(config)
    dn2fn2doc_csv = read_csv(config)
    
    for dn in constants.SPIDER_DN_LIST:
        for exp_fn in constants.EXP_DN2FN[dn]:
            create_doc_info(config, dn2fn2doc_list[dn], dn2fn2doc_csv[dn], dn, exp_fn, is_experiment=True)

    logging.info("Data Initialization Done!")

def create_doc_info(config, fn2doc_list, fn2doc_csv, dn, exp_fn=None, is_experiment=True):

    out_dir = os.path.join(config["out_main"], f"{dn}/{exp_fn}" if is_experiment else dn)
    os.makedirs(out_dir, exist_ok=True)
    doc_info_path = os.path.join(out_dir, "doc_info.json")
    
    if not os.path.exists(doc_info_path):

        doc_dict_path = os.path.join(
                config["out_main"],  # config["schemagen"]["out_main"], 
                f"{dn}/{exp_fn}" if is_experiment else dn,
                "doc_dict.json"
            )
        with open(doc_dict_path, "r", encoding='utf-8') as f:
            doc_dict = json.load(f)

        doc_info = {}
        for i in doc_dict:
            doc = doc_dict[i][0]
            fn = os.path.splitext(doc_dict[i][1])[0]
            doc_info[i] = {
                    "doc": doc, 
                    "fn": fn, 
                    "data": []
                }
            doc_id = int(doc_dict[i][2])  # doc_id = fn2doc_list[fn].index(doc)
            row = fn2doc_csv[fn].iloc[doc_id]
            doc_info[i]["data"] = {col: str(row[col]) for col in fn2doc_csv[fn].columns}

        with open(doc_info_path, "w", encoding='utf-8') as f:
            json.dump(doc_info, f, indent=2)
        logging.info(f"[Data Initialization] All document info extracted in {doc_info_path}")
    
    else:
        logging.info(f"[Data Initialization] Document info already exists in {doc_info_path}")

def read_doc(config):
    dn2fn2doc_list = {}
    for dn in constants.SPIDER_DN_LIST:
        dn2fn2doc_list[dn] = {}
        dn_path = os.path.join(config["spider_path"], dn)
        for fn in constants.SPIDER_DN2FN[dn]:
            dn2fn2doc_list[dn][fn] = []
            file_path = os.path.join(dn_path, f"{fn}.json")
            with open(file_path, "r", encoding='utf-8') as f:
                id2doc = json.load(f)
                for doc_id in id2doc:
                    dn2fn2doc_list[dn][fn].append((id2doc[doc_id].strip()))
            # file_path = os.path.join(dn_path, f"{fn}.txt")
            # with open(file_path, "r", encoding='utf-8') as f:
            #     for doc in f:
            #         dn2fn2doc_list[dn][fn].append(doc.strip())
    return dn2fn2doc_list

def read_csv(config):
    dn2fn2doc_csv = {}
    for dn in constants.SPIDER_DN_LIST:
        dn2fn2doc_csv[dn] = {}
        dn_path = os.path.join(config["spider_path"], dn)
        conn = sqlite3.connect(f"{dn_path}/{dn}.sqlite")
        for fn in constants.SPIDER_DN2FN[dn]:
            dn2fn2doc_csv[dn][fn] = pd.read_sql(f"SELECT * FROM {fn};", conn)
        # for fn in constants.SPIDER_DN2FN[dn]:
        #     file_path = os.path.join(dn_path, f"{fn}.csv")
        #     dn2fn2doc_csv[dn][fn] = pd.read_csv(file_path)
    return dn2fn2doc_csv
