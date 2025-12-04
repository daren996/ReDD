import yaml
import logging
import argparse

from core.utils import logging_utils
from core.correction import ClassifierVal, ClassifierValCodeCorrection, EnsembleAnalyses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/datapop_dsv2lite.yaml")
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
    # val(config["model_dn_fn_list"], config["test_dn_fn"], test_mode="diffsize")
    val(config["model_dn_fn_list"], config["test_dn_fn"], test_mode="ensemble")
    # val(config["model_dn_fn_list"], config["test_dn_fn"], test_mode="incremental")
    # val(config["model_dn_fn_list"], config["test_dn_fn"], test_mode="errorbound")
    # val(config["model_dn_fn_list"], config["test_dn_fn"], test_mode="multiconformal")
    # val(config["model_dn_fn_list"], config["test_dn_fn"], test_mode="multihead")

    # val = ClassifierValCodeCorrection(config)
    # val.correction_marginal_voting(config["model_dn_fn_list"], config["test_dn_fn"])

    # ana = EnsembleAnalyses(config)
    # ana.analyses(config["test_dn_fn"])
    # ana.analyse_multiconformal(config["test_dn_fn"])