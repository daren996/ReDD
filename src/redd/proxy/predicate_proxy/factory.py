"""Predicate proxy creation and training helpers."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from redd.core.utils.sql_filter_parser import AttributePredicate, predicates_to_filter_dict
from redd.embedding import EmbeddingManager
from redd.proxy.proxy_runtime.executor import ConformalProxy, EmbeddingProxy
from redd.proxy.proxy_runtime.types import ProxyPipelineConfig

from .heuristic_proxy import HeuristicPredicateProxy

try:
    from .finetuned_proxy import (
        GLiClassProxy,
        _format_predicate_context,
        train_finetuned_proxy,
    )
    FINETUNED_AVAILABLE = True
except ImportError:
    FINETUNED_AVAILABLE = False


def _generate_filter_description(
    predicate: AttributePredicate,
    mode: str = "gemini",
    llm_model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a short textual description of a filter predicate using LLM.
    Falls back to a template if LLM fails.
    """
    attr = predicate.attribute
    op = predicate.operator
    val = predicate.value
    fallback = f"Documents where {attr} {op} {val!r}."
    normalized_mode = str(mode or "").strip().lower()
    llm_modes = {"gemini", "openai", "deepseek", "together", "siliconflow"}
    # In ground-truth / non-LLM modes, never require API keys for proxy description.
    if normalized_mode in {"ground_truth", "gt", "disabled", "none", ""}:
        return fallback
    if normalized_mode not in llm_modes:
        return fallback
    try:
        from redd.llm import CompletionRequest, LLMRuntime, get_api_key
        try:
            key = get_api_key({"mode": mode}, mode, api_key)
        except ValueError:
            logging.warning(
                "[PredicateProxyFactory] Missing API key for filter description "
                f"(mode={mode}); using fallback template."
            )
            return fallback
        msg = (
            f"Generate a short one-sentence description (under 20 words) of a filter "
            f"for database records: attribute '{attr}' with condition '{op} {val!r}'. "
            f"Describe what it means for a document/record to satisfy this filter. "
            f"Reply with only the description, no quotes."
        )
        runtime = LLMRuntime.from_config(
            mode,
            llm_model,
            config={"mode": mode, "llm_model": llm_model},
            api_key=key,
        )
        response = runtime.complete_text(
            CompletionRequest(
                messages=[{"role": "user", "content": msg}],
                response_format="text",
            )
        ).text
        resp = (response or "").strip()
        return resp[:200] if resp else fallback
    except Exception as e:
        logging.warning(f"[PredicateProxyFactory] LLM filter description failed: {e}. Using fallback.")
        return fallback


class PredicateProxyFactory:
    """Factory for predicate-level proxy creation and training."""
    
    def __init__(
        self, 
        config: ProxyPipelineConfig,
        embedding_manager: EmbeddingManager
    ):
        self.config = config
        self.embedding_manager = embedding_manager
    
    def create_proxies(
        self,
        predicates: List[AttributePredicate],
        query_text: str,
        expected_embedding_dim: Optional[int] = None
    ) -> List[Union[ConformalProxy, EmbeddingProxy]]:
        """
        Create proxies for the given predicates.
        
        Args:
            predicates: List of attribute predicates
            query_text: Query text for embedding-based proxies
            expected_embedding_dim: Dimension of document embeddings. If provided,
                ensures query embedding matches (avoids mismatch with cached embeddings).
            
        Returns:
            List of proxy objects
        """
        proxies = []
        
        # Create embedding-based proxy for overall relevance
        if self.config.use_embedding_proxies:
            try:
                logging.debug(f"[PredicateProxyFactory] Computing query embedding with model={self.config.embedding_model}")
                query_emb = np.array(
                    self.embedding_manager.embed_single(query_text),
                    dtype=np.float32,
                )
                
                # Validate dimension matches document embeddings.
                if expected_embedding_dim is not None and query_emb.shape[0] != expected_embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: query embedding has {query_emb.shape[0]} dims "
                        f"but document embeddings have {expected_embedding_dim}. "
                        "This suggests document embeddings were created with a different model."
                    )
                
                relevance_proxy = EmbeddingProxy(
                    name="query_relevance",
                    query_embedding=query_emb,
                    threshold=self.config.proxy_threshold,
                    cost=self.config.embedding_proxy_cost
                )
                proxies.append(relevance_proxy)
                logging.info(
                    f"[PredicateProxyFactory] Created embedding-based relevance proxy "
                    f"(model={self.config.embedding_model}, dim={query_emb.shape[0]})"
                )
                
            except Exception as e:
                logging.warning(f"[PredicateProxyFactory] Could not create embedding proxy: {e}")
        
        # Create classifier-based proxies for each predicate
        if self.config.use_classifier_proxies and TORCH_AVAILABLE:
            for pred in predicates:
                classifier_path = self.config.classifier_paths.get(pred.attribute)
                if classifier_path and Path(classifier_path).exists():
                    try:
                        # Load classifier
                        classifier = torch.load(classifier_path)
                        classifier.eval()
                        
                        proxy = ConformalProxy(
                            name=f"clf_{pred.attribute}",
                            classifier=classifier,
                            threshold=self.config.proxy_threshold,
                            cost=self.config.classifier_proxy_cost,
                            pass_rate=0.5  # Will be updated from runtime stats
                        )
                        proxies.append(proxy)
                        logging.info(f"[PredicateProxyFactory] Created classifier proxy for {pred.attribute}")
                        
                    except Exception as e:
                        logging.warning(f"[PredicateProxyFactory] Could not load classifier for {pred.attribute}: {e}")
        
        return proxies

    def train_proxies(
        self,
        predicates: List[AttributePredicate],
        query_text: str,
        doc_embeddings: np.ndarray,
        doc_ids: List[str],
        extractions: Dict[str, Dict[str, Any]]
    ) -> List[ConformalProxy]:
        """
        Train binary classifiers for each predicate using collected data.
        
        Each proxy uses: [doc_embedding, filter_embedding] as input.
        Filter embedding is generated by: (1) LLM generates textual description of filter,
        (2) embed that description. Both embeddings are concatenated for the classifier.
        
        Args:
            predicates: List of predicates to train proxies for
            query_text: Query text (unused; kept for API compatibility)
            doc_embeddings: Embeddings of training documents
            doc_ids: IDs of training documents
            extractions: Extracted values for training documents
            
        Returns:
            List of trained ConformalProxy objects
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("[PredicateProxyFactory] Scikit-learn not available, cannot train proxies. Installing sklearn is recommended.")
            return []
            
        proxies = []
        mode = getattr(self.config, "llm_mode", "gemini")
        model = getattr(self.config, "llm_model", "gemini-2.5-flash-lite")
        api_key = getattr(self.config, "api_key", None)
        
        # Create predicate check functions
        predicate_fns = predicates_to_filter_dict(predicates)
        
        # Train a proxy for each predicate
        for pred in predicates:
            attr = pred.attribute
            if attr not in predicate_fns:
                continue
                
            pred_fn = predicate_fns[attr]
            
            # 1. Generate filter description via LLM
            filter_desc = _generate_filter_description(pred, mode=mode, llm_model=model, api_key=api_key)
            logging.debug(f"[PredicateProxyFactory] Filter description for {attr}: {filter_desc[:60]}...")
            
            # 2. Embed the filter description
            try:
                filter_emb = np.array(
                    self.embedding_manager.embed_single(filter_desc),
                    dtype=np.float32,
                )
            except Exception as e:
                logging.warning(f"[PredicateProxyFactory] Failed to embed filter for {attr}: {e}")
                continue
            
            # 3. Prepare training data (X, y): [doc_emb, filter_emb] per document
            X = []
            y = []
            
            for i, doc_id in enumerate(doc_ids):
                doc_emb = doc_embeddings[i]
                feature_vec = np.concatenate([doc_emb, filter_emb])
                X.append(feature_vec)
                
                extracted = extractions.get(doc_id, {})
                val = extracted.get(attr)
                try:
                    label = 1 if (val is not None and pred_fn(val)) else 0
                except Exception:
                    label = 0
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            
            if n_pos < 2 or n_neg < 2:
                logging.warning(f"[PredicateProxyFactory] Not enough samples to train proxy for {attr} (pos={n_pos}, neg={n_neg}). Skipping.")
                continue
                
            logging.info(f"[PredicateProxyFactory] Training proxy for {attr} with {len(y)} samples (pos={n_pos}, neg={n_neg})")
            
            try:
                clf = LogisticRegression(class_weight='balanced', max_iter=1000)
                clf.fit(X, y)
                
                # Predict wrapper: [doc_embs, filter_emb] -> concatenate and predict
                def predict_wrapper(doc_embs, model=clf, f_emb=filter_emb):
                    batch_size = len(doc_embs)
                    f_embs = np.tile(f_emb, (batch_size, 1))
                    features = np.concatenate([doc_embs, f_embs], axis=1)
                    probs = model.predict_proba(features)[:, 1]
                    return probs
                
                proxy = ConformalProxy(
                    name=f"learned_{attr}",
                    classifier=predict_wrapper,
                    threshold=0.5,
                    cost=self.config.classifier_proxy_cost,
                    pass_rate=n_pos / len(y)
                )
                
                # Calibrate on training data
                scores, _ = proxy.evaluate(doc_embeddings)
                pos_scores = scores[y == 1]
                if len(pos_scores) > 0:
                    target_recall = self.config.target_recall
                    quantile = 1.0 - target_recall
                    threshold = np.quantile(pos_scores, quantile)
                    proxy.threshold = max(float(threshold), 0.01)
                    logging.info(f"  Calibrated threshold: {proxy.threshold:.4f} (target recall {target_recall})")
                
                proxies.append(proxy)
                
            except Exception as e:
                logging.warning(f"[PredicateProxyFactory] Failed to train proxy for {attr}: {e}")
                
        return proxies

    def train_proxies_finetuned(
        self,
        predicates: List[AttributePredicate],
        query_text: str,
        documents: List[str],
        doc_ids: List[str],
        extractions: Dict[str, Dict[str, Any]],
        model_name: str = "knowledgator/gliclass-small-v1.0",
        output_dir: Optional[Union[str, Path]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        seed: Optional[int] = None,
    ) -> List["GLiClassProxy"]:
        """
        Train GLiClass proxies for each predicate.

        Args:
            predicates: List of predicates to train proxies for
            query_text: Query text (for predicate context)
            documents: Document texts (same order as doc_ids)
            doc_ids: Document IDs
            extractions: Ground truth extractions per doc_id
            model_name: HuggingFace model name
            output_dir: Checkpoint directory
            epochs: Training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
            seed: Random seed for deterministic training

        Returns:
            List of GLiClassProxy objects
        """
        if not FINETUNED_AVAILABLE:
            logging.warning("[PredicateProxyFactory] Fine-tuned proxies not available. Install transformers and torch.")
            return []

        predicate_fns = predicates_to_filter_dict(predicates)
        proxies = []

        for pred in predicates:
            attr = pred.attribute
            if attr not in predicate_fns:
                continue

            pred_fn = predicate_fns[attr]
            predicate_context = _format_predicate_context(pred, query_text)

            labels = []
            for doc_id in doc_ids:
                val = extractions.get(doc_id, {}).get(attr)
                try:
                    label = 1 if (val is not None and pred_fn(val)) else 0
                except Exception:
                    label = 0
                labels.append(label)

            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            if n_pos < 2 or n_neg < 2:
                logging.warning(
                    f"[PredicateProxyFactory] Not enough samples for fine-tuned proxy {attr} "
                    f"(pos={n_pos}, neg={n_neg}). Skipping."
                )
                continue

            try:
                proxy, _ = train_finetuned_proxy(
                    attribute=attr,
                    predicate_context=predicate_context,
                    documents=documents,
                    labels=labels,
                    model_name=model_name,
                    output_dir=output_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    target_recall=self.config.target_recall,
                    seed=int(self.config.random_seed if seed is None else seed),
                    use_icl=getattr(self.config, "use_gliclass_icl", False),
                    icl_examples_per_class=getattr(
                        self.config, "gliclass_icl_examples_per_class", 3
                    ),
                )
                proxies.append(proxy)
            except Exception as e:
                logging.warning(f"[PredicateProxyFactory] Failed to train fine-tuned proxy for {attr}: {e}")

        return proxies

    def create_pretrained_proxies(
        self,
        predicates: List[AttributePredicate],
        query_text: str,
        model_name: str = "knowledgator/gliclass-small-v1.0",
        threshold: Optional[float] = None,
    ) -> List["GLiClassProxy"]:
        """Create one pre-trained GLiClass proxy per predicate without runtime training."""
        if str(model_name).strip().lower() in {"heuristic", "rule_based", "value_heuristic"}:
            proxy_threshold = 0.51 if threshold is None else float(threshold)
            proxies = [
                HeuristicPredicateProxy(
                    pred,
                    name=f"heuristic_{pred.attribute}",
                    threshold=proxy_threshold,
                    pass_through_attributes=getattr(
                        self.config,
                        "heuristic_pass_through_attributes",
                        None,
                    ),
                )
                for pred in predicates
            ]
            for pred in predicates:
                logging.info(
                    "[PredicateProxyFactory] Created heuristic predicate proxy for %s",
                    pred.attribute,
                )
            return proxies

        if not FINETUNED_AVAILABLE:
            logging.warning(
                "[PredicateProxyFactory] Pretrained predicate proxies unavailable. "
                "Install gliclass/transformers/torch or configure classifier_paths."
            )
            return []

        proxies = []
        for pred in predicates:
            predicate_context = _format_predicate_context(pred, query_text)
            try:
                proxy, _ = train_finetuned_proxy(
                    attribute=pred.attribute,
                    predicate_context=predicate_context,
                    documents=[],
                    labels=[],
                    model_name=model_name,
                    epochs=0,
                    target_recall=self.config.target_recall,
                    seed=int(self.config.random_seed),
                    use_icl=False,
                )
                if threshold is not None:
                    proxy.threshold = float(threshold)
                proxies.append(proxy)
                logging.info(
                    "[PredicateProxyFactory] Created pretrained predicate proxy for %s",
                    pred.attribute,
                )
            except Exception as exc:
                logging.warning(
                    "[PredicateProxyFactory] Failed to create pretrained proxy for %s: %s",
                    pred.attribute,
                    exc,
                )
        return proxies
