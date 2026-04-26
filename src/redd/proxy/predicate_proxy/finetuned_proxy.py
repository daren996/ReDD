"""
Text classification proxies for the proxy runtime.

Uses GLiClass model (zero-shot or fine-tuned) for document predicate filtering.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    from transformers import set_seed as hf_set_seed
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    hf_set_seed = None

GLICLASS_AVAILABLE = False
GLICLASS_TRAINING_AVAILABLE = False
try:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    GLICLASS_AVAILABLE = True
    try:
        from gliclass.data_processing import (
            AugmentationConfig,
            DataCollatorWithPadding,
            GLiClassDataset,
        )
        from gliclass.training import Trainer, TrainingArguments
        GLICLASS_TRAINING_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

# Default model: GLiClass
DEFAULT_MODEL = "knowledgator/gliclass-small-v1.0"


def _is_gliclass_model(model_name: str) -> bool:
    """Return True if model_name indicates GLiClass or a pretrained model path."""
    if "gliclass" in model_name.lower():
        return True
    # Local path to pretrained GLiClass (e.g. outputs/pretrained_proxies/.../final_model)
    p = Path(model_name)
    return p.exists() and p.is_dir()


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve model path to absolute when it's a local directory.
    HuggingFace from_pretrained treats relative paths as repo IDs; use absolute path for local load.
    """
    p = Path(model_name)
    if p.exists() and p.is_dir():
        return str(p.resolve())
    return model_name


def _set_reproducible_seed(seed: int) -> None:
    """Best-effort deterministic seeding for fine-tuned proxy training."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if hf_set_seed is not None:
        hf_set_seed(seed)


def _load_tokenizer(model_path: str, load_kw: Dict[str, Any]) -> Any:
    """
    Load tokenizer with Mistral regex fix when available.

    Newer transformers exposes `fix_mistral_regex`; older versions may not.
    """
    tokenizer_kw = dict(load_kw)
    tokenizer_kw["fix_mistral_regex"] = True
    try:
        return AutoTokenizer.from_pretrained(model_path, **tokenizer_kw)
    except TypeError:
        tokenizer_kw.pop("fix_mistral_regex", None)
        logging.warning(
            "[GLiClassProxy:_load_tokenizer] `fix_mistral_regex` not supported by "
            "current transformers version; loading tokenizer without the fix."
        )
        return AutoTokenizer.from_pretrained(model_path, **tokenizer_kw)


def _format_predicate_context(predicate: Any, query_text: str = "") -> str:
    """
    Format predicate into a good prompt for the fine-tuned classifier.
    Uses query context when available for richer signal.
    """
    attr = predicate.attribute
    op = predicate.operator
    val = predicate.value
    pred_str = f"{attr} {op} {val!r}"
    
    if query_text and len(query_text.strip()) > 0:
        # Natural language prompt: query context + specific filter to check
        return (
            f"Query: {query_text.strip()[:300]}\n\n"
            f"Does this document satisfy the filter: {pred_str}?"
        )
    return f"Does this document satisfy the filter: {pred_str}?"


class FineTunedTextProxy:
    """
    Proxy that uses a fine-tuned transformer for document classification.
    Takes raw text (document + predicate context) instead of embeddings.
    """

    uses_documents = True  # Signal to executor to pass documents

    def __init__(
        self,
        name: str,
        model: Any,
        tokenizer: Any,
        threshold: float = 0.5,
        cost: float = 0.2,
        pass_rate: float = 0.5,
        device: Optional[str] = None,
        predicate_context: str = "",
    ):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self._threshold = threshold
        self._cost = cost
        self._pass_rate = pass_rate
        self.predicate_context = predicate_context

        if device is None:
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        self._total_seen = 0
        self._total_passed = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def pass_rate(self) -> float:
        if self._total_seen > 10:
            return self._total_passed / self._total_seen
        return self._pass_rate

    @property
    def rejection_efficiency(self) -> float:
        if self.cost <= 0:
            return float("inf")
        return (1.0 - self.pass_rate) / self.cost

    def evaluate(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Required by interface but we use documents. Call evaluate_documents instead.
        This raises if called - executor should use evaluate_documents for this proxy.
        """
        raise NotImplementedError(
            "FineTunedTextProxy uses documents. Call evaluate_documents() or ensure "
            "executor passes documents for proxies with uses_documents=True."
        )

    def evaluate_documents(self, documents: List[str], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a proxy on raw documents.

        Args:
            documents: List of document texts
            **kwargs: Ignored (e.g. doc_ids for other proxies)

        Returns:
            (scores, passed_mask)
        """
        if not documents:
            return np.array([]), np.array([], dtype=bool)

        # Truncate long docs
        docs_truncated = [doc[:4000] for doc in documents]
        # Use tokenizer's encode_plus for proper [CLS] doc [SEP] pred [SEP] format
        inputs = [
            (doc, self.predicate_context) for doc in docs_truncated
        ]

        with torch.no_grad():
            encoding = self.tokenizer(
                [p[0] for p in inputs],
                [p[1] for p in inputs],
                truncation="longest_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        passed_mask = probs >= self._threshold

        self._total_seen += len(probs)
        self._total_passed += passed_mask.sum()

        return probs, passed_mask


class GLiClassProxy:
    """
    Proxy that uses GLiClass zero-shot classification (no fine-tuning).
    Same interface as FineTunedTextProxy.
    """

    uses_documents = True

    def __init__(
        self,
        name: str,
        pipeline: Any,
        threshold: float = 0.5,
        cost: float = 0.2,
        pass_rate: float = 0.5,
        predicate_context: str = "",
        icl_examples: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.pipeline = pipeline
        self._threshold = threshold
        self._cost = cost
        self._pass_rate = pass_rate
        self.predicate_context = predicate_context
        self.icl_examples = icl_examples or []
        self._total_seen = 0
        self._total_passed = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def pass_rate(self) -> float:
        if self._total_seen > 10:
            return self._total_passed / self._total_seen
        return self._pass_rate

    @property
    def rejection_efficiency(self) -> float:
        if self.cost <= 0:
            return float("inf")
        return (1.0 - self.pass_rate) / self.cost

    def evaluate(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "GLiClassProxy uses documents. Call evaluate_documents() or ensure "
            "executor passes documents for proxies with uses_documents=True."
        )

    def evaluate_documents(self, documents: List[str], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Batch inference: pass all documents in one call (no per-doc loop)."""
        if not documents:
            return np.array([]), np.array([], dtype=bool)

        docs_truncated = [doc[:4000] for doc in documents]
        labels = ["satisfies", "does not satisfy"]
        prompt = self.predicate_context if self.predicate_context else "Does this document satisfy the filter?"

        pipeline_kwargs: Dict[str, Any] = {
            "prompt": [prompt] * len(docs_truncated),
            "threshold": 0.0,
        }
        if self.icl_examples:
            pipeline_kwargs["examples"] = self.icl_examples

        try:
            # Single batch call: all documents at once (like notebook: pipeline(docs, labels, prompt=prompts))
            all_results = self.pipeline(docs_truncated, labels, **pipeline_kwargs)
        except Exception as e:
            logging.warning(f"[GLiClassProxy] Pipeline failed: {e}")
            return np.zeros(len(documents), dtype=np.float32), np.zeros(len(documents), dtype=bool)

        probs_list: List[float] = []
        for j, results in enumerate(all_results):
            score_satisfies = 0.0
            res = results if isinstance(results, list) else results
            for r in res:
                if r.get("label") == "satisfies":
                    score_satisfies = float(r.get("score", 0.0))
                    break
            probs_list.append(score_satisfies)

        probs = np.array(probs_list, dtype=np.float32)
        passed_mask = probs >= self._threshold
        self._total_seen += len(probs)
        self._total_passed += passed_mask.sum()
        return probs, passed_mask


def _build_icl_examples(
    documents: List[str],
    labels: List[int],
    examples_per_class: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample few-shot examples for in-context learning (GLiClass format)."""
    rng = random.Random(seed)
    pos_indices = [i for i, label in enumerate(labels) if label == 1]
    neg_indices = [i for i, label in enumerate(labels) if label == 0]
    examples: List[Dict[str, Any]] = []
    for indices, label in [(pos_indices, "satisfies"), (neg_indices, "does not satisfy")]:
        k = min(examples_per_class, len(indices))
        if k > 0:
            chosen = rng.sample(indices, k)
            for i in chosen:
                examples.append({"text": documents[i][:4000], "labels": [label]})
    return examples


def _train_gliclass_proxy(
    attribute: str,
    predicate_context: str,
    documents: List[str],
    labels: List[int],
    model_name: str,
    target_recall: float = 0.95,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
    use_icl: bool = False,
    icl_examples_per_class: int = 3,
) -> Tuple[GLiClassProxy, None]:
    """
    Create a GLiClass proxy. Fine-tunes when epochs > 0 and training is available;
    otherwise uses zero-shot. When use_icl=True, adds few-shot examples for in-context learning.
    """
    device = "cuda:0" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and TORCH_AVAILABLE:
        logging.info(
            f"[GLiClassProxy] Using GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        logging.info("[GLiClassProxy] Using CPU (no GPU available)")
    model_path = _resolve_model_path(model_name)
    is_local = Path(model_path).exists() and Path(model_path).is_dir()
    load_kw = {"local_files_only": True} if is_local else {}
    model = GLiClassModel.from_pretrained(model_path, **load_kw)
    tokenizer = _load_tokenizer(model_path, load_kw)

    # Fine-tune when epochs > 0 and gliclass.training is available
    if epochs > 0 and GLICLASS_TRAINING_AVAILABLE and len(documents) >= 4:
        try:
            _set_reproducible_seed(seed)
            train_data = []
            labels_gliclass = ["satisfies", "does not satisfy"]
            prompt = (
                predicate_context
                if predicate_context
                else "Does this document satisfy the filter?"
            )
            for doc, label in zip(documents, labels):
                true_label = "satisfies" if label == 1 else "does not satisfy"
                train_data.append(
                    {
                        "text": doc[:4000],
                        "all_labels": labels_gliclass,
                        "true_labels": [true_label],
                        "prompt": prompt,
                    }
                )

            aug_config = AugmentationConfig(enabled=False)
            train_dataset = GLiClassDataset(
                train_data,
                tokenizer,
                aug_config,
                label2description={},
                max_length=512,
                problem_type="multi_label_classification",
                architecture_type="uni-encoder",
                prompt_first=True,
                shuffle_labels=True,
            )
            data_collator = DataCollatorWithPadding(device=device)

            save_path = (
                Path(output_dir) / attribute if output_dir else Path(f"./proxy_checkpoints/gliclass_{attribute}")
            )
            save_path.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(save_path),
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                num_train_epochs=epochs,
                save_strategy="no",
                logging_steps=10,
                report_to="none",
                seed=seed,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            trainer.train()
            model = trainer.model
            logging.info(
                f"[GLiClassProxy] Fine-tuned {attribute} for {epochs} epochs "
                f"(lr={learning_rate}, {len(documents)} samples)"
            )
        except Exception as e:
            logging.warning(
                f"[GLiClassProxy] Fine-tuning failed: {e}. Using zero-shot."
            )
    elif epochs > 0 and not GLICLASS_TRAINING_AVAILABLE:
        logging.warning(
            "[GLiClassProxy] gliclass.training not available. Using zero-shot. "
            "Install from source for fine-tuning."
        )

    pipeline = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="multi-label", device=device
    )

    labels_gliclass = ["satisfies", "does not satisfy"]
    prompt = predicate_context if predicate_context else "Does this document satisfy the filter?"

    icl_examples: List[Dict[str, Any]] = []
    if use_icl and len(documents) >= 2:
        icl_examples = _build_icl_examples(
            documents, labels, icl_examples_per_class, seed
        )
        if icl_examples:
            logging.info(
                f"[GLiClassProxy] In-context learning: {len(icl_examples)} examples"
            )

    docs_truncated = [doc[:4000] for doc in documents]
    pipeline_kwargs: Dict[str, Any] = {
        "prompt": [prompt] * len(docs_truncated),
        "threshold": 0.0,
    }
    if icl_examples:
        pipeline_kwargs["examples"] = icl_examples

    try:
        all_results = pipeline(docs_truncated, labels_gliclass, **pipeline_kwargs)
    except Exception as e:
        logging.warning(f"[GLiClassProxy] Pipeline failed during calibration: {e}")
        all_results = []

    all_scores: List[float] = []
    for j, results in enumerate(all_results):
        score_satisfies = 0.0
        res = results if isinstance(results, list) else results
        for r in res:
            if r.get("label") == "satisfies":
                score_satisfies = float(r.get("score", 0.0))
                break
        all_scores.append(score_satisfies)
    if len(all_scores) < len(documents):
        all_scores.extend([0.0] * (len(documents) - len(all_scores)))

    all_scores_arr = np.array(all_scores, dtype=np.float32)
    pos_scores = all_scores_arr[np.array(labels) == 1]
    if len(pos_scores) > 0:
        quantile = 1.0 - target_recall
        threshold = float(np.quantile(pos_scores, quantile))
        threshold = max(threshold, 0.01)
        logging.info(
            f"[GLiClassProxy] Calibrated threshold: {threshold:.4f} (target recall {target_recall})"
        )
    else:
        threshold = 0.5

    n_pos = sum(labels)
    proxy = GLiClassProxy(
        name=f"gliclass_{attribute}",
        pipeline=pipeline,
        threshold=threshold,
        cost=0.2,
        pass_rate=n_pos / len(labels) if labels else 0.5,
        predicate_context=predicate_context,
        icl_examples=icl_examples if icl_examples else None,
    )
    return proxy, None


def train_finetuned_proxy(
    attribute: str,
    predicate_context: str,
    documents: List[str],
    labels: List[int],
    model_name: str = DEFAULT_MODEL,
    output_dir: Optional[Union[str, Path]] = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    target_recall: float = 0.95,
    seed: int = 42,
    use_icl: bool = False,
    icl_examples_per_class: int = 3,
) -> Tuple[GLiClassProxy, None]:
    """
    Train or create a GLiClass proxy for binary classification.

    Fine-tunes when epochs > 0 (if gliclass.training available),
    else zero-shot with threshold calibration.

    Args:
        attribute: Attribute name (for proxy naming)
        predicate_context: Formatted predicate string for model input
        documents: Training document texts
        labels: Binary labels (1 = satisfies predicate)
        model_name: GLiClass model name (e.g. knowledgator/gliclass-small-v1.0)
        output_dir: Where to save checkpoint (fine-tuned path only)
        epochs: Training epochs (GLiClass and DeBERTa paths)
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning
        target_recall: Target recall for threshold calibration
        seed: Random seed for reproducible training

    Returns:
        (Proxy, tokenizer or None for GLiClass)
    """
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise ImportError("PyTorch and transformers required for fine-tuned proxies")

    _set_reproducible_seed(seed)

    # GLiClass path only (no fallback)
    if not GLICLASS_AVAILABLE:
        raise ImportError(
            "gliclass package required for learned proxies. Install with: pip install gliclass"
        )
    if not _is_gliclass_model(model_name):
        raise ValueError(
            f"Learned proxies require a GLiClass model (e.g. knowledgator/gliclass-small-v1.0). "
            f"Got: {model_name}"
        )

    return _train_gliclass_proxy(
        attribute=attribute,
        predicate_context=predicate_context,
        documents=documents,
        labels=labels,
        model_name=model_name,
        target_recall=target_recall,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        seed=seed,
        use_icl=use_icl,
        icl_examples_per_class=icl_examples_per_class,
    )
