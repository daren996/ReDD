import os
import yaml
import logging
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import logging_utils
# from dataset import schema_gen_prep
from core.schema_gen import SchemaGenGPT, SchemaGenDeepSeek, SchemaGenTogether, SchemaGenSiliconFlow
# from core.evaluation import EvalSchemaGen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/schemagen.yaml")
    parser.add_argument("--exp", type=str, default="spider_4d1_1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as file: 
            config = yaml.safe_load(file)
        config = config[args.exp]
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit()

    # Setup logging with console log level from config
    console_log_level = logging_utils.get_log_level(config.get("console_log_level", "WARNING"))
    logging_utils.setup_logging(exp=args.exp, log_dir="logs", console_log_level=console_log_level)

    # if args.init:
    #     schema_gen_prep.data_init(config)
    #     exit()
    
    if config["mode"] == "cgpt":
        if args.api_key:
            config["api_key"] = args.api_key
        schema_gen = SchemaGenGPT(config)
    elif config["mode"] == "deepseek":
        if args.api_key:
            config["api_key"] = args.api_key
        schema_gen = SchemaGenDeepSeek(config, api_key=args.api_key)
    elif config["mode"] == "together":
        if args.api_key:
            config["api_key"] = args.api_key
        schema_gen = SchemaGenTogether(config, api_key=args.api_key)
    elif config["mode"] == "siliconflow":
        if args.api_key:
            config["api_key"] = args.api_key
        schema_gen = SchemaGenSiliconFlow(config, api_key=args.api_key)
    else:
        logging.error(f"Invalid mode {config['mode']}")
        exit()    

    schema_gen(config["exp_dn_fn_list"])

    # if args.eval:
    #     eval = EvalSchemaGen(config)
    #     eval(config["exp_dn_fn_list"])
    