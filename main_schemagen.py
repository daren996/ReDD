import os
import yaml
import logging
import argparse

from utils import logging_utils
from dataset import schema_gen_prep as schema_gen_prep
from schema_gen.schemagen_gpt import SchemaGenGPT
from schema_gen.schemagen_deepseek import SchemaGenDeepSeek


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfg/schemagen.yaml")
    parser.add_argument("--exp", type=str, default="spider_4d1_1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--init", action="store_true")
    # parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    logging_utils.setup_logging(exp=args.exp, log_dir="logs")

    try:
        with open(args.config, "r") as file: 
            config = yaml.safe_load(file)
        config = config[args.exp]
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        exit()

    if args.init:
        schema_gen_prep.data_init(config)
        exit()
    
    if config["mode"] == "cgpt":
        schema_gen = SchemaGenGPT(config)
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
    elif config["mode"] == "deepseek":
        schema_gen = SchemaGenDeepSeek(config, api_key=args.api_key)
    else:
        logging.error(f"Invalid mode {config['mode']}")
        exit()    

    schema_gen(config["exp_dn_list"], config["exp_fn_list"])

