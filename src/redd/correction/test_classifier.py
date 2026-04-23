import logging
import os
import random
from collections import Counter

import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from redd.exceptions import UnsupportedInputError
from redd.core.data_loader import create_data_loader
from redd.core.utils.constants import PATH_TEMPLATES
from redd.core.utils.progress import tqdm
from .classifier_structure import BinaryClassifier0 as BinaryClassifier
from .classifier_structure import MultiHeadBinaryClassifier
from .hidden_states_loader import LazyHiddenStatesDataset
from .train_classifier import ClassifierTrainer
from .voting_error_estimation import chernoff_bound, estimate_mv_error_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassifierVal(ClassifierTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.out_main = config["out_main"]
        self.cls_train_trials = config["cls_train_trials"]

        self.tokenizer = None
        self.load_tokenizer()

        self.hidden_size = config.get("hidden_size", 2048)
        self.exp_layers = config.get("exp_layers", [20, 21, 22, 23, 24, 25, 26])
        self.exp_train_sizes = config.get("exp_train_sizes", 256)
        self.max_train_did = 0

        self.classifier_threshold = config.get("classifier_threshold", 0.5)
        self.ensemble_size = config.get("ensemble_size", 3)
        self.num_ensembles = config.get("num_ensembles", 2)
        self.ensemble_sample_trials = config.get("ensemble_sample_trials", 20)
        self.qids = config.get("qids", None)

        self.alpha_range = [0.001, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
        self.num_cells = config.get("num_cells", 50)
        self.max_cells_did = self.max_train_did + self.num_cells
        self.cell_dids = None
        self.num_recal = config.get("num_recal", 200)
        self.max_recal_did = self.max_cells_did + self.num_recal
        self.recal_dids = None

        self.min_test_did = self.max_recal_did
        self.max_test_did = 99999
        self.test_dids = None
        self.num_heads = config.get("num_heads", 3)

    def __call__(self, model_dn_fn_list, test_dn_fn, test_mode="diffsize"):
        self.test_mode = test_mode
        for self.cls_train_trial in self.cls_train_trials:
            model_dict = self._get_model_dict(model_dn_fn_list)
            logging.info(
                "[%s] **START EXP** test_dn_fn: %s; test_mode: %s; model_dict: %s",
                self.__class__.__name__,
                test_dn_fn,
                self.test_mode,
                model_dict,
            )
            data_root = os.path.join(self.config.get("data_main", self.out_main), test_dn_fn)
            out_root = os.path.join(self.out_main, test_dn_fn)
            loader = create_data_loader(
                data_root=data_root,
                loader_type=self.config.get("data_loader_type", "sqlite"),
                loader_config=self.config.get("data_loader_config", {}),
            )
            query_dict = loader.load_query_dict()
            if not self.qids:
                self.qids = list(query_dict.keys())
            results = {}
            for qid in self.qids:
                logging.info("[%s] Processing query %s...", self.__class__.__name__, qid)
                eval_output = self.load_json(os.path.join(out_root, PATH_TEMPLATES.eval_result(qid, self.param_str)))
                if not self.all_dids:
                    self.all_dids = [int(did) for did in eval_output]
                self.test_dids = [did for did in self.all_dids if self.min_test_did <= did < self.max_test_did]
                logging.info(
                    "[%s] All DIDs: %s-%s; Sample test DIDs: %s-%s; Sample cell DIDs: %s-%s; Sample recal DIDs: %s-%s",
                    self.__class__.__name__,
                    min(self.all_dids),
                    max(self.all_dids),
                    min(self.test_dids),
                    max(self.test_dids),
                    self.max_train_did,
                    self.max_cells_did,
                    self.max_cells_did,
                    self.max_recal_did,
                )

                classifier_outputs = self._get_classifier_outputs(loader, model_dict, out_root, qid, self.all_dids, eval_output)
                gt_all = {}
                for did in self.all_dids:
                    gt_all[did] = 0 if eval_output[str(did)]["final"] else 1

                if self.test_mode == "multiconformal":
                    self.cell_dids = self.all_dids[self.max_train_did : self.max_cells_did]
                    self.recal_dids = self.all_dids[self.max_cells_did : self.max_recal_did]
                    results[qid] = self._evaluate_multi_conformal(qid, gt_all, model_dict, classifier_outputs)
                elif self.test_mode == "multihead":
                    self.cell_dids = self.all_dids[self.max_train_did : self.max_cells_did]
                    self.recal_dids = self.all_dids[self.max_cells_did : self.max_recal_did]
                    results[qid] = self._evaluate_multi_head(qid, gt_all, model_dict, classifier_outputs)
                else:
                    results[qid] = self._evaluate_modes(qid, gt_all, model_dict, classifier_outputs)

                exp_save_path = os.path.join(
                    self.out_main,
                    test_dn_fn,
                    "eval_classifiers_best",
                    f"eval_classifiers{self.cls_train_trial}_{qid}_{self.test_mode}.json",
                )
                self.save_results(exp_save_path, results[qid])
            logging.info(
                "[%s] **END EXP** test_dn_fn: %s; test_mode: %s; results: %s",
                self.__class__.__name__,
                test_dn_fn,
                self.test_mode,
                results,
            )

    def _get_model_dict(self, model_dn_fn_list):
        return {
            model_dn_fn: {
                model_qid: {
                    layer: {
                        f"s{_size}": os.path.join(
                            self.out_main,
                            model_dn_fn,
                            "classifiers",
                            f"classifiers{self.cls_train_trial}_{model_qid}",
                            f"best_model_layer{layer}_s{_size}.pt",
                        )
                        for _size in self.exp_train_sizes
                    }
                    for layer in self.exp_layers
                }
                for model_qid in self.load_json(os.path.join(self.out_main, model_dn_fn, "queries.json"))
            }
            for model_dn_fn in model_dn_fn_list
        }

    def _get_classifier_outputs(self, loader, model_dict, out_root, qid, dids, eval_output):
        classifier_outputs_path = os.path.join(
            out_root,
            "classifier_outputs_best",
            f"classifier_outputs{self.cls_train_trial}_{qid}_{self.param_str}_dids{dids[0]}-{dids[-1]}.json",
        )
        if not os.path.exists(classifier_outputs_path):
            layer2dataset = self._load_dataset(loader, eval_output, out_root, qid, dids)
            logging.info(
                "[%s:_get_classifier_outputs] Loaded query dataset (in LazyHiddenStatesDataset) for qid %s",
                self.__class__.__name__,
                qid,
            )
            classifier_outputs = self._run_classifier(qid, layer2dataset, model_dict, dids)
            self.save_results(classifier_outputs_path, classifier_outputs)
        else:
            classifier_outputs = self.load_json(classifier_outputs_path)
        logging.info("[%s:_get_classifier_outputs] Loaded classifier outputs for qid %s", self.__class__.__name__, qid)
        return classifier_outputs

    def _load_dataset(self, loader, eval_output, out_root, qid, dids):
        res_data_path = os.path.join(out_root, PATH_TEMPLATES.data_population_result(qid, self.param_str))
        res_data_dict = self.load_processed_res(res_data_path)
        states_dir = os.path.join(out_root, PATH_TEMPLATES.hidden_states_dir(qid, self.param_str))
        layer2dataset = {}
        for layer in self.exp_layers:
            dataset = LazyHiddenStatesDataset(
                trainer=self,
                loader=loader,
                res_data_dict=res_data_dict,
                states_dir=states_dir,
                eval_output=eval_output,
                layer_index=layer,
                sample_dids=dids,
                qid=qid,
            )
            layer2dataset[layer] = dataset
        return layer2dataset

    def _run_classifier(self, qid, layer2dataset, model_dict, dids):
        classifier_outputs = {}
        for model_dn_fn, qid_models in model_dict.items():
            classifier_outputs[model_dn_fn] = {}
            for model_qid, layer_models in qid_models.items():
                if model_qid != qid:
                    continue
                classifier_outputs[model_dn_fn][model_qid] = {}
                for layer in self.exp_layers:
                    dataset: LazyHiddenStatesDataset = layer2dataset[layer]
                    classifier_outputs[model_dn_fn][model_qid][str(layer)] = {}
                    for size, model_path in layer_models[layer].items():
                        logging.info(
                            "[%s:_run_classifier] Processing model %s for qid %s, layer %s, size %s",
                            self.__class__.__name__,
                            model_path,
                            qid,
                            layer,
                            size,
                        )
                        model = BinaryClassifier(self.hidden_size).to(device)
                        self.load_model(model, model_path)
                        model.eval()
                        all_outputs = {}
                        with torch.no_grad():
                            for did in dids:
                                inputs = dataset.get_did_hidden_state(did).to(device)
                                outputs = model(inputs)
                                outputs = outputs.squeeze(1)
                                outputs = torch.sigmoid(outputs).tolist()
                                all_outputs[str(did)] = outputs
                        classifier_outputs[model_dn_fn][model_qid][str(layer)][size] = all_outputs
        return classifier_outputs

    def _run_classifier_multi(self, qid, layer2dataset, model_dict, dids):
        classifier_outputs = {}
        for model_dn_fn, qid_models in model_dict.items():
            classifier_outputs[model_dn_fn] = {}
            for model_qid, layer_models in qid_models.items():
                if model_qid != qid:
                    continue
                classifier_outputs[model_dn_fn][model_qid] = {}
                for layer in self.exp_layers:
                    dataset: LazyHiddenStatesDataset = layer2dataset[layer]
                    classifier_outputs[model_dn_fn][model_qid][str(layer)] = {}
                    for size, model_path in layer_models[layer].items():
                        logging.info(
                            "[%s:_run_classifier] Processing model %s for qid %s, layer %s, size %s",
                            self.__class__.__name__,
                            model_path,
                            qid,
                            layer,
                            size,
                        )
                        model = MultiHeadBinaryClassifier(self.hidden_size, num_heads=self.num_heads).to(device)
                        self.load_model(model, model_path)
                        model.eval()
                        all_outputs = {}
                        with torch.no_grad():
                            for did in dids:
                                inputs = dataset.get_did_hidden_state(did).to(device)
                                outputs = model(inputs)
                                head_outputs = []
                                for head_output in outputs:
                                    head_output = head_output.squeeze(1)
                                    head_output = torch.sigmoid(head_output).tolist()
                                    head_outputs.append(head_output)
                                all_outputs[str(did)] = head_outputs
                        classifier_outputs[model_dn_fn][model_qid][str(layer)][size] = all_outputs
        return classifier_outputs

    def _evaluate_modes(self, qid, gt_all, model_dict, classifier_outputs):
        results = {}
        for model_dn_fn, qid_models in model_dict.items():
            results[model_dn_fn] = {}
            for model_qid, layer_models in qid_models.items():
                if model_qid != qid:
                    continue
                results[model_dn_fn][model_qid] = {}
                for _size in self.exp_train_sizes:
                    size = f"s{_size}"
                    results[model_dn_fn][model_qid][size] = {}

                    if self.test_mode == "diffsize":
                        for layer in self.exp_layers:
                            model_path = layer_models[layer][size]
                            acc = self._evaluate_row_level(gt_all, classifier_outputs[model_dn_fn][model_qid][str(layer)][size])
                            results[model_dn_fn][model_qid][size][layer] = acc
                            logging.info(
                                "[%s:_evaluate_modes] diffsize test_mode, model_path %s-%s, layer %s, size %s, acc %s. model_path %s",
                                self.__class__.__name__,
                                model_dn_fn,
                                model_qid,
                                layer,
                                size,
                                acc,
                                model_path,
                            )

                    elif self.test_mode == "ensemble":
                        layers_selects = self.select_layers()
                        for layers in layers_selects:
                            model_paths = [layer_models[layer][size] for layer in layers]
                            cls_outputs_list = [classifier_outputs[model_dn_fn][model_qid][str(layer)][size] for layer in layers]
                            acc = self._evaluate_row_level_voting(gt_all, cls_outputs_list)
                            ensemble_layer = "_".join(str(l) for l in layers)
                            results[model_dn_fn][model_qid][size][ensemble_layer] = acc
                            logging.info(
                                "[%s:_evaluate_modes] ensemble test_mode, model %s-%s, layers %s, size %s, acc %s. model_paths %s",
                                self.__class__.__name__,
                                model_dn_fn,
                                model_qid,
                                ensemble_layer,
                                size,
                                acc,
                                model_paths,
                            )

                    elif self.test_mode == "incremental":
                        results[model_dn_fn][model_qid][size] = []
                        for _ in range(self.ensemble_sample_trials):
                            ensname2acc = {}
                            layers_selects = self.select_layers()
                            for i in range(len(layers_selects)):
                                ensembles = layers_selects[: i + 1]
                                cls_outputs_list_ensembles = []
                                ensemble_name = []
                                for layers in ensembles:
                                    ensemble_name.append("_".join(str(l) for l in layers))
                                    cls_outputs_list = [classifier_outputs[model_dn_fn][model_qid][str(layer)][size] for layer in layers]
                                    cls_outputs_list_ensembles.append(cls_outputs_list)
                                acc = self._evaluate_row_level_voting_ensembles(gt_all, cls_outputs_list_ensembles)
                                if len(ensembles) == 3:
                                    ensnum2errest = self._estimate_voting_error_ensemble(gt_all, cls_outputs_list_ensembles)
                                    acc["residual_error_estimation"] = ensnum2errest.get(3)
                                ensname2acc["__".join(ensemble_name)] = acc
                            results[model_dn_fn][model_qid][size].append(ensname2acc)
                            logging.info(
                                "[%s:_evaluate_modes] incremental test_mode, model %s-%s, size %s, trial %s",
                                self.__class__.__name__,
                                model_dn_fn,
                                model_qid,
                                size,
                                _,
                            )

                    elif self.test_mode == "errorbound":
                        layers_selects = self.select_layers()
                        for layers in layers_selects:
                            model_paths = [layer_models[layer][size] for layer in layers]
                            cls_outputs_list = [classifier_outputs[model_dn_fn][model_qid][str(layer)][size] for layer in layers]
                            acc = self._evaluate_row_level_voting(gt_all, cls_outputs_list)
                            error_bounds = self._estimate_voting_error_classifiers(gt_all, cls_outputs_list)
                            error_bounds["error_rate"] = acc["error_rate"]
                            error_bounds["residual_error_rate"] = acc["residual_error_rate"]
                            error_bounds["extra_cost_rate"] = acc["extra_cost_rate"]
                            ensemble_layer = "_".join(str(l) for l in layers)
                            results[model_dn_fn][model_qid][size][ensemble_layer] = error_bounds
                            logging.info(
                                "[%s:_evaluate_modes] errorbound test_mode, model %s-%s, layers %s, size %s, acc %s. model_paths %s",
                                self.__class__.__name__,
                                model_dn_fn,
                                model_qid,
                                ensemble_layer,
                                size,
                                acc,
                                model_paths,
                            )
        return results

    def select_layers(self, size=None):
        size = self.ensemble_size if size is None else size
        layers_selected = set()
        layers_selects = []
        for _ in range(self.num_ensembles):
            available_layers = list(set(self.exp_layers) - layers_selected)
            layers = random.sample(available_layers, size)
            layers_selects.append(layers)
            layers_selected.update(layers)
        return layers_selects

    def _evaluate_multi_conformal(self, qid, gt_all, model_dict, classifier_outputs):
        results = {}
        for model_dn_fn, qid_models in model_dict.items():
            results[model_dn_fn] = {}
            for model_qid, layer_models in qid_models.items():
                if model_qid != qid:
                    continue
                results[model_dn_fn][model_qid] = {}
                for _size in self.exp_train_sizes:
                    size = f"s{_size}"
                    results[model_dn_fn][model_qid][size] = {}

                    layer_outputs = []
                    for layer in self.exp_layers:
                        layer_outputs.append(classifier_outputs[model_dn_fn][model_qid][str(layer)][size])
                    for alpha in self.alpha_range:
                        did2prediction_sets = self._multi_conformal_prediction(layer_outputs, gt_all, alpha)
                        fndids = [did for did in self.test_dids if gt_all[did] == 1 and 1 not in did2prediction_sets[did]]
                        coverage = np.mean([gt_all[did] in did2prediction_sets[did] for did in self.test_dids])
                        avg_set_size = np.mean([len(set_) for _, set_ in did2prediction_sets.items()])
                        abstain = len([set_ for _, set_ in did2prediction_sets.items() if len(set_) > 1]) / len(self.test_dids)
                        all_preds = [1 if 1 in set_ else 0 for _, set_ in did2prediction_sets.items()]
                        acc = self._compute_acc([gt_all[did] for did in self.test_dids], all_preds)
                        acc = {"coverage": float(coverage), "abstain": abstain, "avg_set_size": float(avg_set_size)} | acc
                        acc = acc | {"fndids": fndids}
                        results[model_dn_fn][model_qid][size][alpha] = acc
                        logging.info(
                            "[%s:_perform_multi_conformal_prediction] model %s-%s, size %s, alpha %s, acc %s",
                            self.__class__.__name__,
                            model_dn_fn,
                            model_qid,
                            size,
                            alpha,
                            acc,
                        )
        return results

    def _evaluate_multi_head(self, qid, gt_all, model_dict, classifier_outputs):
        results = {}
        for model_dn_fn in model_dict:
            results[model_dn_fn] = {}
            for model_qid in model_dict[model_dn_fn]:
                if model_qid != qid:
                    continue
                results[model_dn_fn][model_qid] = {}
                for _size in self.exp_train_sizes:
                    size = f"s{_size}"
                    results[model_dn_fn][model_qid][size] = {}

                    layer2results = {}
                    for layer in self.exp_layers:
                        layer_outputs = classifier_outputs[model_dn_fn][model_qid][str(layer)][size]
                        head_outputs = [{} for _ in range(self.num_heads)]
                        for did in layer_outputs:
                            for head_idx, outputs in enumerate(layer_outputs[did]):
                                head_outputs[head_idx][did] = outputs
                        layer2results[layer] = {}
                        for alpha in self.alpha_range:
                            did2prediction_sets = self._multi_conformal_prediction(head_outputs, gt_all, alpha)
                            fndids = [did for did in self.test_dids if gt_all[did] == 1 and 1 not in did2prediction_sets[did]]
                            coverage = np.mean([gt_all[did] in did2prediction_sets[did] for did in self.test_dids])
                            avg_set_size = np.mean([len(set_) for _, set_ in did2prediction_sets.items()])
                            abstain = len([set_ for _, set_ in did2prediction_sets.items() if len(set_) > 1]) / len(self.test_dids)
                            all_preds = [1 if 1 in set_ else 0 for _, set_ in did2prediction_sets.items()]
                            acc = self._compute_acc([gt_all[did] for did in self.test_dids], all_preds)
                            acc = {"coverage": float(coverage), "abstain": abstain, "avg_set_size": float(avg_set_size)} | acc
                            acc = acc | {"fndids": fndids}
                            layer2results[layer][alpha] = acc
                            logging.info(
                                "[%s:_evaluate_multi_head] model %s-%s, size %s, layer %s, alpha %s, acc %s",
                                self.__class__.__name__,
                                model_dn_fn,
                                model_qid,
                                size,
                                layer,
                                alpha,
                                acc,
                            )
                    results[model_dn_fn][model_qid][size] = layer2results
        return results

    def _multi_conformal_prediction(self, layer_outputs, gt_all, alpha):
        s_cells = []
        for did in self.cell_dids:
            y = gt_all[did]
            s = []
            for outputs in layer_outputs:
                output_max = max(outputs[str(did)])
                s.append(1 - output_max if y == 1 else output_max)
            s_cells.append(s)
        s_cells = np.array(s_cells)

        tree = KDTree(s_cells)

        center2f = {}
        center2t = {}
        for did in self.recal_dids:
            for y in [0, 1]:
                s = []
                for outputs in layer_outputs:
                    output_max = max(outputs[str(did)])
                    s.append(1 - output_max if y == 1 else output_max)
                _, idx = tree.query(s)
                center = idx
                if y == gt_all[did]:
                    center2t[center] = center2t.get(center, 0) + 1
                else:
                    center2f[center] = center2f.get(center, 0) + 1

        center2d = {}
        for center in range(len(s_cells)):
            f = center2f.get(center, 0)
            t = center2t.get(center, 0)
            center2d[center] = float("inf") if t == 0 else (f + t) / t

        sorted_centers = sorted(range(len(s_cells)), key=lambda x: center2d[x])
        selected_centers = []
        target_coverage = np.ceil((1 - alpha) * (self.num_recal + 1))
        for center in sorted_centers:
            selected_centers.append(center)
            covered = 0
            for did in self.recal_dids:
                s_true = []
                for outputs in layer_outputs:
                    output_max = max(outputs[str(did)])
                    y = gt_all[did]
                    s_true.append(1 - output_max if y == 1 else output_max)
                _, idx = tree.query(s_true)
                if idx in selected_centers:
                    covered += 1
            if covered >= target_coverage:
                break

        did2prediction_sets = {}
        for did in self.test_dids:
            sets = []
            for y in [0, 1]:
                s = []
                for outputs in layer_outputs:
                    output_max = max(outputs[str(did)])
                    s.append(1 - output_max if y == 1 else output_max)
                _, idx = tree.query(s)
                if idx in selected_centers:
                    sets.append(y)
            did2prediction_sets[did] = sets if sets else [0]
        return did2prediction_sets

    def _evaluate_row_level(self, gt_all, cls_outputs, voting_mode="half"):
        return self._evaluate_row_level_voting(gt_all, [cls_outputs], voting_mode=voting_mode)

    def _evaluate_row_level_voting(self, gt_all, cls_outputs_list, voting_mode="half"):
        return self._evaluate_row_level_voting_ensembles(gt_all, [cls_outputs_list], voting_mode=voting_mode)

    def _evaluate_row_level_voting_ensembles(self, gt_all, cls_outputs_list_ensembles, voting_mode="half"):
        all_preds, all_labels, _ = self._apply_voting(gt_all, cls_outputs_list_ensembles, voting_mode=voting_mode)
        metrics = self._compute_acc(all_labels, all_preds)

        if self.test_mode == "diffsize":
            auc_scores = self._compute_auc_for_each_model(gt_all, cls_outputs_list_ensembles)
            metrics["auc_scores"] = auc_scores
        return metrics

    def _compute_acc(self, all_labels, all_preds):
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        precision = float(precision_score(all_labels, all_preds, zero_division=0))
        recall = float(recall_score(all_labels, all_preds, zero_division=0))
        f1 = float(f1_score(all_labels, all_preds, zero_division=0))
        error_rate = 1 - float(accuracy_score(all_labels, all_preds))
        residual_error_rate = float(fn) / len(all_preds)
        negative_count = len(all_preds) - (tp + fn)
        extra_cost_rate = float(fp) / negative_count if negative_count > 0 else 0.0

        original_accuracy = all_labels.count(0) / len(all_labels)
        corrected_labels = [0 if label == 1 and pred == 1 else label for label, pred in zip(all_labels, all_preds)]
        corrected_accuracy = corrected_labels.count(0) / len(corrected_labels)

        logging.info(
            "[%s:_evaluate_row_level_voting_ensembles] Counter Labels %s; Preds %s",
            self.__class__.__name__,
            Counter(all_labels),
            Counter(all_preds),
        )
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        total_errors = int(np.sum(all_labels))
        predicted_errors = int(np.sum(all_preds))
        detected_errors = int(np.sum((all_labels == 1) & (all_preds == 1)))
        undetected_errors = total_errors - detected_errors

        return {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "residual_error_rate": residual_error_rate,
            "extra_cost_rate": extra_cost_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "error_rate": error_rate,
            "total_errors": total_errors,
            "predicted_errors": predicted_errors,
            "detected_errors": detected_errors,
            "undetected_errors": undetected_errors,
            "original_accuracy": original_accuracy,
            "corrected_accuracy": corrected_accuracy,
        }

    def _compute_auc_for_each_model(self, gt_all, cls_outputs_list_ensembles):
        preds, labels = [], []
        for did in self.test_dids:
            outputs = np.array(cls_outputs_list_ensembles[0][0][str(did)])
            pred_prob = outputs.max().item()
            preds.append(pred_prob)
            labels.append(gt_all[did])
        return roc_auc_score(labels, preds)

    def _apply_voting(self, gt_all, cls_outputs_list_ensembles, voting_mode):
        all_preds, all_labels = [], []
        ensemble_preds = []
        for did in self.test_dids:
            if voting_mode == "hard":
                ensemble_votes = []
                for ensemble_i in range(len(cls_outputs_list_ensembles)):
                    votes = []
                    for layer_i in range(len(cls_outputs_list_ensembles[ensemble_i])):
                        outputs = np.array(cls_outputs_list_ensembles[ensemble_i][layer_i][str(did)])
                        vote = 1 if (outputs >= self.classifier_threshold).any() else 0
                        votes.append(vote)
                    ensemble_vote = 1 if sum(votes) >= (len(votes) / 2) else 0
                    ensemble_votes.append(ensemble_vote)
                ensemble_preds.append(ensemble_votes)
                row_pred_voting = 1 if sum(ensemble_votes) >= (len(ensemble_votes) / 2) else 0
            else:
                ensemble_votes, avg_probs = [], []
                for ensemble_i in range(len(cls_outputs_list_ensembles)):
                    probs = []
                    for layer_i in range(len(cls_outputs_list_ensembles[ensemble_i])):
                        outputs = np.array(cls_outputs_list_ensembles[ensemble_i][layer_i][str(did)])
                        model_prob = outputs.max().item()
                        probs.append(model_prob)
                    avg_prob = sum(probs) / len(probs)
                    avg_probs.append(avg_prob)
                    ensemble_votes.append(1 if avg_prob >= self.classifier_threshold else 0)
                ensemble_preds.append(ensemble_votes)
                if voting_mode == "half":
                    row_pred_voting = 1 if sum(ensemble_votes) >= (len(ensemble_votes) / 2) else 0
                elif voting_mode == "soft":
                    avg_avg_prob = sum(avg_probs) / len(avg_probs)
                    row_pred_voting = 1 if avg_avg_prob >= self.classifier_threshold else 0
                else:
                    raise UnsupportedInputError(
                        f"{self.__class__.__name__} received invalid voting_mode `{voting_mode}`"
                    )

            all_preds.append(row_pred_voting)
            all_labels.append(gt_all[did])
        return all_preds, all_labels, ensemble_preds

    def _estimate_voting_error_classifiers(self, gt_all, cls_outputs_list, voting_mode="half"):
        cls_preds, labels = [], []
        for did in self.test_dids:
            votes = []
            for layer_i in range(len(cls_outputs_list)):
                outputs = np.array(cls_outputs_list[layer_i][str(did)])
                vote = 1 if (outputs >= self.classifier_threshold).any() else 0
                votes.append(vote)
            cls_preds.append(votes)
            labels.append(gt_all[did])

        y_gt = np.array(labels)
        preds = np.array(cls_preds)
        k2error = {}
        for k in range(3, 9):
            k2error[k] = round(estimate_mv_error_fn(y_gt, preds, k=k), 4)

        p_e = k2error[3]
        n2prob = {}
        for n in range(1, 9):
            n2prob[n] = round(chernoff_bound(n, p_e), 4)

        return {
            "k3": k2error[3],
            "k4": k2error[4],
            "k5": k2error[5],
            "k6": k2error[6],
            "k7": k2error[7],
            "k8": k2error[8],
            "n1": n2prob[1],
            "n2": n2prob[2],
            "n3": n2prob[3],
            "n4": n2prob[4],
            "n5": n2prob[5],
            "n6": n2prob[6],
            "n7": n2prob[7],
            "n8": n2prob[8],
        }

    def _estimate_voting_error_ensemble(self, gt_all, cls_outputs_list_ensembles, voting_mode="half"):
        _, _, ensemble_preds = self._apply_voting(gt_all, cls_outputs_list_ensembles, voting_mode=voting_mode)
        y_gt = np.array([gt_all[did] for did in self.test_dids])
        preds = np.array(ensemble_preds)
        k2error = {}
        for k in range(3, 9):
            k2error[k] = round(estimate_mv_error_fn(y_gt, preds, k=k), 4)
        return k2error

    def nonconformity_score(output_prob, label):
        return 1 - output_prob if label == 1 else output_prob


__all__ = ["ClassifierVal"]
