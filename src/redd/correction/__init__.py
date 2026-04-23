"""Future-facing namespace for correction and reliability workflows."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "BinaryClassifier0",
    "BinaryClassifier1",
    "ClassifierTrainer",
    "ClassifierVal",
    "ClassifierValCodeCorrection",
    "EnsembleAnalyses",
    "LazyHiddenStatesDataset",
    "MultiHeadBinaryClassifier",
    "VotingErrorEstimation",
]

_EXPORT_MAP = {
    "BinaryClassifier0": ".classifier_structure",
    "BinaryClassifier1": ".classifier_structure",
    "ClassifierTrainer": ".train_classifier",
    "ClassifierVal": ".test_classifier",
    "ClassifierValCodeCorrection": ".codeword_correction",
    "EnsembleAnalyses": ".ensemble_analyses",
    "LazyHiddenStatesDataset": ".hidden_states_loader",
    "MultiHeadBinaryClassifier": ".classifier_structure",
    "VotingErrorEstimation": ".voting_error_estimation",
}


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)
