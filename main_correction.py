import yaml
import logging
import argparse

from utils import logging_utils
from h_classifier.test_classifier import ClassifierVal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cfg/datapop_dsv2lite.yaml")
    parser.add_argument("--exp", type=str, default="my_exp")
    args = parser.parse_args()

    logging_utils.setup_logging(exp=args.exp, log_dir="logs")

    try:
        with open(args.config, "r") as file: 
            config = yaml.safe_load(file)
        config = config[args.exp]
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        exit()

    val = ClassifierVal(config)
    val(config["model_dn_fn_list"], config["test_dn_fn"], mode="diffsize")
    val(config["model_dn_fn_list"], config["test_dn_fn"], mode="ensemble")
    val(config["model_dn_fn_list"], config["test_dn_fn"], mode="SCAPE")
    val(config["model_dn_fn_list"], config["test_dn_fn"], mode="SCAPE-Hyb")
