import os
import yaml
import logging
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import logging_utils
# from dataset import data_pop_prep as data_pop_prep
from core.data_population import DataPopGPT, DataPopDeepSeek, DataPopTogether, DataPopSiliconFlow, DataPopLocal
from core.evaluation import EvalDataPop
# from core.correction import ClassifierTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/datapop_ds7b.yaml")
    parser.add_argument("--exp", type=str, default="spider_1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-classifier", action="store_true")
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
    #     data_pop_prep.data_init(config)
    #     exit()
    
    # if config["mode"] == "cgpt":
    #     datapop = DataPopGPT(config, api_key=args.api_key)
    # elif config["mode"] == "deepseek":
    #     datapop = DataPopDeepSeek(config, api_key=args.api_key)
    # elif config["mode"] == "together":
    #     datapop = DataPopTogether(config, api_key=args.api_key)
    # elif config["mode"] == "siliconflow":
    #     datapop = DataPopSiliconFlow(config, api_key=args.api_key)
    # elif config["mode"] in ["local"]:  # local model, no api
    #     datapop = DataPopLocal(config)
    # else:
    #     logging.error(f"Invalid mode {config['mode']}")
    #     exit()

    # datapop(config["exp_dn_fn_list"])

    if args.eval:
        if "committee" in config["eval"] or config["eval"]["mode"] in ["deepseek", "cgpt"]:
            eval = EvalDataPop(config, api_key=args.api_key)
        else:
            logging.error(f"Invalid eval mode {config['eval']['mode']}")
            exit()

        eval(config["exp_dn_fn_list"])

    # if args.train_classifier:
    #     trainer = ClassifierTrainer(config)
    #     trainer(config["exp_dn_fn_list"])
