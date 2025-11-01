import os
import json
import logging


class EvalBasic:
    def __init__(self, config, data_loader=None):
        self.config = config
        self.data_loader = data_loader

        self.prediction_data: list = None
        self.gt_data: list = None
    
    def __call__(self):
        raise NotImplementedError

    def load_data(self, prediction_path: str, gt_path: str):
        raise NotImplementedError

    def is_null(self, data) -> bool:
        raise NotImplementedError

    def is_match(self, pred, gt) -> bool:
        raise NotImplementedError

    def compute_stat(self):
        """
        Compute the statistics of the evaluation.
        True Positives (TP): prediction != null, prediction matches gt
        False Positives (FP): prediction != null, prediction dismatches gt
        False Negatives (FN): prediction == null, prediction dismatches gt
        True Negatives (TN): prediction == null, prediction matches gt
        """

        if not self.prediction_data or not self.gt_data:
            logging.error(f"[{self.__class__.__name__}:compute_stat] No data loaded.")
            return None
        if len(self.prediction_data) != len(self.gt_data):
            logging.error(f"[{self.__class__.__name__}:compute_stat] Results and ground truth data have different lengths.")
            return None
        
        true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0
        for predicted, gt in zip(self.prediction_data, self.gt_data):
            if not self.is_null(predicted["table"]):
                if self.is_match(predicted, gt):
                    true_positives += 1
                else:
                    false_positives += 1
                    logging.info(f"[{self.__class__.__name__}:compute_stat] false_positives: \n\t{predicted}\n\t{gt}")
            else:
                if not self.is_null(gt["table"]):
                    false_negatives += 1
                    logging.info(f"[{self.__class__.__name__}:compute_stat] false_negatives: \n\t{predicted}\n\t{gt}")
                else:
                    true_negatives += 1
                    logging.info(f"[{self.__class__.__name__}:compute_stat] true_negatives: \n\t{predicted}\n\t{gt}")

        return true_positives, false_positives, false_negatives, true_negatives
    
    def compute_recall_precision_f1(self, tp, fp, fn):
        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0.0 else 0.0
        return recall, precision, f1

    def save_results(self, res_path, res_dict):
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=2)
    
    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
