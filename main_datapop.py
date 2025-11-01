import os
import yaml
import logging
import argparse

from utils import logging_utils
from dataset import data_pop_prep as data_pop_prep
from data_population.datapop_deepseek import DataPopDeepSeek
from data_population.datapop_local import DataPopLocal
from data_population.datapop_gpt import DataPopGPT
from eval.eval_datapop import EvalDataPopDeepSeek
from h_classifier.train_classifier import ClassifierTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfg/datapop_cogital32b.yaml")
    parser.add_argument("--exp", type=str, default="spider_1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train-classifier", action="store_true")
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
        data_pop_prep.data_init(config)
        exit()
    
    if config["mode"] == "cgpt":
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
        datapop = DataPopGPT(config)
    elif config["mode"] == "deepseek":
        datapop = DataPopDeepSeek(config, api_key=args.api_key)
    elif config["mode"] in ["ds7b", "dsv2lite", "cogito32b", "cogito70b"]:  # local model, no api
        datapop = DataPopLocal(config)
    else:
        logging.error(f"Invalid mode {config['mode']}")
        exit()

    datapop(config["exp_dn_list"], config["exp_fn_list"])

    if args.eval:
        if config["eval"]["mode"] == "cgpt":
            pass  # TODO
            # eval = EvalDataPopGPT(config)
        elif config["eval"]["mode"] == "deepseek":
            eval = EvalDataPopDeepSeek(config, api_key=args.api_key)
        else:
            logging.error(f"Invalid eval mode {config['eval']['mode']}")
            exit()

    eval(config["exp_dn_list"], config["exp_fn_list"])

    if args.train_classifier:
        trainer = ClassifierTrainer(config)
        trainer(config["exp_dn_list"], config["exp_fn_list"])
