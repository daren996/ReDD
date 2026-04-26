"""
Proxy Runtime Execution Engine

A cost-optimized data extraction pipeline that filters documents using
lightweight proxy models before invoking expensive LLM extraction.

Algorithm:
1. Proxy Chain: Lightweight binary classifiers on embeddings, one per attribute
2. Fail-Fast Ordering: Sort proxies by Rejection Efficiency = (1 - PassRate) / Cost
3. Conformal Filtering: For each document, run proxies in order; REJECT if score < threshold
4. Atomic Oracle: Only if document passes ALL proxies, invoke LLM for extraction
5. Feedback Loop: Collect hard negatives (LLM rejects after proxies pass) for retraining

Example Usage:
    ```python
    from redd.proxy.proxy_runtime.executor import ProxyExecutor, ConformalProxy
    
    # Create proxies from trained classifiers
    proxies = [
        ConformalProxy(name="price_filter", classifier=price_clf, 
                       threshold=0.3, cost=0.1, pass_rate=0.6),
        ConformalProxy(name="category_filter", classifier=cat_clf,
                       threshold=0.4, cost=0.15, pass_rate=0.4),
    ]
    
    # Create executor
    executor = ProxyExecutor(proxies=proxies, llm_oracle=oracle)
    
    # Process batch
    results, stats = executor.process_batch(documents, embeddings)
    
    # Get hard negatives for retraining
    hard_negatives = executor.get_hard_negatives()
    ```
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from .ordering import reording

# Try importing torch for GPU-accelerated inference
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

__all__ = [
    "ConformalProxy",
    "create_proxy_executor_from_classifiers",
    "EmbeddingProxy",
    "ProxyDecision",
    "DocumentBatch",
    "ProxyExecutor",
    "ProxyRuntimeConfig",
    "LLMOracleProtocol",
    "HardNegative",
    "ExecutionStats",
]


# ============================================================================
# Type Definitions and Protocols
# ============================================================================

class LLMOracleProtocol(Protocol):
    """Protocol for LLM Oracle that extracts attributes from documents."""
    
    def extract(
        self, 
        document: str, 
        schema: Dict[str, Any],
        attributes: List[str]
    ) -> Dict[str, Any]:
        """
        Extract attribute values from a document.
        
        Args:
            document: Raw document text
            schema: Schema definition for the target table
            attributes: List of attribute names to extract
            
        Returns:
            Dictionary mapping attribute names to extracted values
        """
        ...
    
    def check_predicates(
        self,
        extracted_values: Dict[str, Any],
        predicates: Dict[str, Callable[[Any], bool]]
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if extracted values satisfy the query predicates.
        
        Args:
            extracted_values: Extracted attribute values
            predicates: Dictionary mapping attribute names to predicate functions
            
        Returns:
            (all_passed, per_attribute_results) tuple
        """
        ...


class FilterDecision(Enum):
    """Decision made by a proxy for a document."""
    PASS = "pass"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ProxyRuntimeConfig:
    """Configuration for proxy-chain execution."""
    # Proxy-chain configuration
    min_proxies_before_oracle: int = 1  # Minimum proxies that must pass before LLM
    max_proxies_per_attribute: int = 1  # Max proxies to run per attribute
    
    # Threshold configuration
    default_threshold: float = 0.5
    use_conformal_thresholds: bool = True
    
    # Batch processing
    batch_size: int = 32
    use_gpu: bool = True
    
    # Feedback loop
    collect_hard_negatives: bool = True
    hard_negative_buffer_size: int = 1000
    
    # Logging
    verbose: bool = False
    log_interval: int = 100


@dataclass
class ProxyDecision:
    """Result from a single proxy evaluation."""
    proxy_name: str
    score: float
    threshold: float
    decision: FilterDecision
    latency_ms: float
    
    @property
    def passed(self) -> bool:
        return self.decision == FilterDecision.PASS


@dataclass
class HardNegative:
    """
    A hard negative sample: document that passed all proxies but failed LLM predicates.
    Used for retraining proxy models.
    """
    doc_id: str
    document: str
    embedding: np.ndarray
    failed_attribute: str
    extracted_value: Any
    expected_predicate: str
    proxy_scores: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class DocumentBatch:
    """
    Batch of documents with pre-computed embeddings.
    
    Supports both numpy arrays and torch tensors for efficient batch processing.
    """
    doc_ids: List[str]
    documents: List[str]
    embeddings: Union[np.ndarray, "torch.Tensor"]  # Shape: (batch_size, embedding_dim)
    
    # Optional metadata
    metadata: Optional[List[Dict[str, Any]]] = None
    
    def __len__(self) -> int:
        return len(self.doc_ids)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, np.ndarray]:
        emb = self.embeddings[idx]
        if TORCH_AVAILABLE and isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        return self.doc_ids[idx], self.documents[idx], emb
    
    def get_embeddings_numpy(self) -> np.ndarray:
        """Get embeddings as numpy array."""
        if TORCH_AVAILABLE and isinstance(self.embeddings, torch.Tensor):
            return self.embeddings.cpu().numpy()
        return self.embeddings
    
    def get_embeddings_torch(self, device: str = "cpu") -> "torch.Tensor":
        """Get embeddings as torch tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        if isinstance(self.embeddings, torch.Tensor):
            return self.embeddings.to(device)
        return torch.tensor(self.embeddings, dtype=torch.float32, device=device)


@dataclass
class ExecutionStats:
    """Statistics from batch processing."""
    total_documents: int = 0
    rejected_by_proxies: int = 0
    passed_to_oracle: int = 0
    oracle_accepted: int = 0
    oracle_rejected: int = 0  # Hard negatives

    # Per-proxy statistics
    proxy_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-proxy per-document decisions: proxy_name -> list of rejected doc_ids
    proxy_rejected_doc_ids: Dict[str, List[str]] = field(default_factory=dict)
    # Doc IDs that passed all proxies
    passed_doc_ids: List[str] = field(default_factory=list)
    # All doc IDs in the batch (for recall computation)
    all_doc_ids: List[str] = field(default_factory=list)

    # Timing
    total_proxy_time_ms: float = 0.0
    total_oracle_time_ms: float = 0.0
    avg_proxies_per_doc: float = 0.0
    
    @property
    def proxy_rejection_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.rejected_by_proxies / self.total_documents
    
    @property
    def oracle_precision(self) -> float:
        """Precision of proxy chain (how many oracle calls were useful)."""
        if self.passed_to_oracle == 0:
            return 0.0
        return self.oracle_accepted / self.passed_to_oracle
    
    @property
    def cost_savings(self) -> float:
        """Estimated cost savings from proxy filtering."""
        if self.total_documents == 0:
            return 0.0
        return self.rejected_by_proxies / self.total_documents


# ============================================================================
# Conformal Proxy Class
# ============================================================================

class ConformalProxy:
    """
    A lightweight proxy model with conformal calibration.
    
    Wraps a binary classifier and provides:
    - Prediction scores (vectorized)
    - Calibrated threshold for conformal guarantees
    - Cost and pass rate for optimization
    
    Args:
        name: Unique identifier for this proxy (usually attribute name)
        classifier: PyTorch classifier or callable that returns scores
        threshold: Calibrated threshold (scores >= threshold pass)
        cost: Computational cost of running this proxy (relative units)
        pass_rate: Estimated fraction of documents that pass this proxy
        device: Device for PyTorch inference ("cpu" or "cuda")
    """
    
    def __init__(
        self,
        name: str,
        classifier: Union[Callable, "nn.Module"],
        threshold: float = 0.5,
        cost: float = 1.0,
        pass_rate: float = 0.5,
        device: str = "cpu"
    ):
        self.name = name
        self.classifier = classifier
        self._threshold = threshold
        self._cost = cost
        self._pass_rate = pass_rate
        self.device = device
        
        # Runtime statistics for adaptive tuning
        self._total_seen = 0
        self._total_passed = 0
        
        # Move classifier to device if it's a PyTorch module
        if TORCH_AVAILABLE and isinstance(classifier, nn.Module):
            self.classifier = classifier.to(device)
            self.classifier.eval()
    
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
        """
        Return the pass rate. If we have runtime observations, use empirical rate.
        """
        if self._total_seen > 10:  # Use empirical after sufficient samples
            return self._total_passed / self._total_seen
        return self._pass_rate
    
    @property
    def rejection_efficiency(self) -> float:
        """
        Rejection Efficiency = (1 - PassRate) / Cost
        
        Higher values indicate proxies that are cheap and strict (good for fail-fast).
        """
        if self.cost <= 0:
            return float('inf')
        return (1.0 - self.pass_rate) / self.cost
    
    def predict(self, embeddings: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Compute prediction scores for a batch of embeddings.
        
        Args:
            embeddings: Input embeddings, shape (batch_size, embedding_dim)
            
        Returns:
            Scores array, shape (batch_size,). Higher scores indicate positive class.
        """
        if TORCH_AVAILABLE and isinstance(self.classifier, nn.Module):
            return self._predict_torch(embeddings)
        else:
            return self._predict_callable(embeddings)
    
    def _predict_torch(self, embeddings: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """PyTorch-based prediction."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Convert to tensor if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        else:
            embeddings = embeddings.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.classifier(embeddings)
            # Apply sigmoid to get probabilities
            scores = torch.sigmoid(logits).squeeze(-1)
        
        return scores.cpu().numpy()
    
    def _predict_callable(self, embeddings: np.ndarray) -> np.ndarray:
        """Callable-based prediction (for custom models)."""
        if isinstance(embeddings, np.ndarray):
            return np.array(self.classifier(embeddings))
        # Handle torch tensor
        if TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        return np.array(self.classifier(embeddings))
    
    def evaluate(
        self, 
        embeddings: Union[np.ndarray, "torch.Tensor"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a proxy on embeddings and return (scores, passed_mask).
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            (scores, passed_mask) where passed_mask[i] = True if scores[i] >= threshold
        """
        scores = self.predict(embeddings)
        passed_mask = scores >= self.threshold
        
        # Update runtime statistics
        batch_size = len(scores)
        self._total_seen += batch_size
        self._total_passed += passed_mask.sum()
        
        return scores, passed_mask
    
    def calibrate(
        self,
        calibration_embeddings: np.ndarray,
        calibration_labels: np.ndarray,
        target_recall: float = 0.95
    ) -> float:
        """
        Calibrate threshold using conformal prediction to achieve target recall.
        
        Args:
            calibration_embeddings: Embeddings of calibration set
            calibration_labels: Binary labels (1 = positive/should pass)
            target_recall: Target recall rate (e.g., 0.95 for 95% recall)
            
        Returns:
            Calibrated threshold
        """
        scores = self.predict(calibration_embeddings)
        
        # Get scores for positive samples
        positive_scores = scores[calibration_labels == 1]
        
        if len(positive_scores) == 0:
            logging.warning(f"[{self.name}] No positive samples for calibration")
            return self._threshold
        
        # Set threshold at (1 - target_recall) quantile of positive scores
        # This ensures target_recall fraction of positives pass
        quantile = 1.0 - target_recall
        self._threshold = float(np.quantile(positive_scores, quantile))
        
        logging.info(f"[{self.name}] Calibrated threshold: {self._threshold:.4f} "
                    f"(target_recall={target_recall}, n_positive={len(positive_scores)})")
        
        return self._threshold
    
    def reset_stats(self):
        """Reset runtime statistics."""
        self._total_seen = 0
        self._total_passed = 0


# ============================================================================
# Embedding Proxy Class
# ============================================================================

class EmbeddingProxy:
    """
    A proxy that uses embedding similarity to filter documents.
    
    Computes similarity between query embedding and document embedding,
    passing documents that are sufficiently similar.
    """
    
    def __init__(
        self,
        name: str,
        query_embedding: np.ndarray,
        threshold: float = 0.5,
        cost: float = 0.1
    ):
        self.name = name
        self.query_embedding = query_embedding / np.linalg.norm(query_embedding)
        self._threshold = threshold
        self._cost = cost
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
        return 0.5
    
    @property
    def rejection_efficiency(self) -> float:
        if self.cost <= 0:
            return float('inf')
        return (1.0 - self.pass_rate) / self.cost
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity with query embedding."""
        # Ensure numpy array
        if TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
            
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms
        
        # Compute similarities
        similarities = np.dot(normalized, self.query_embedding)
        return similarities
    
    def evaluate(self, embeddings: Union[np.ndarray, "torch.Tensor"]) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a proxy and return (scores, passed_mask)."""
        scores = self.predict(embeddings)
        passed = scores >= self.threshold
        
        self._total_seen += len(scores)
        self._total_passed += passed.sum()
        
        return scores, passed
    
    def reset_stats(self):
        self._total_seen = 0
        self._total_passed = 0


# ============================================================================
# Proxy Executor
# ============================================================================

class ProxyExecutor:
    """
    Cascaded conformal proxy execution engine.
    
    Orchestrates the fail-fast filtering pipeline:
    1. Sort proxies by rejection efficiency
    2. For each document, run proxies in order (short-circuit on reject)
    3. Only invoke LLM oracle if all proxies pass
    4. Collect hard negatives for feedback loop
    
    Args:
        proxies: List of proxy objects
        llm_oracle: LLM Oracle for attribute extraction (optional)
        config: ProxyRuntimeConfig for tuning behavior
    """
    
    def __init__(
        self,
        proxies: Sequence[Union[ConformalProxy, EmbeddingProxy]],
        llm_oracle: Optional[LLMOracleProtocol] = None,
        config: Optional[ProxyRuntimeConfig] = None
    ):
        self.proxies = list(proxies)
        self.llm_oracle = llm_oracle
        self.config = config or ProxyRuntimeConfig()
        
        # Execution plan (sorted proxies)
        self._execution_plan: List[Union[ConformalProxy, EmbeddingProxy]] = []
        
        # Hard negative buffer for feedback loop
        self._hard_negatives: List[HardNegative] = []
        
        # Build initial execution plan
        self.optimize_execution_plan()
        
        logging.info(f"[ProxyExecutor] Initialized with {len(proxies)} proxies")
    
    def optimize_execution_plan(
        self,
        proxies: Optional[Sequence[Union[ConformalProxy, EmbeddingProxy]]] = None,
    ) -> List[Union[ConformalProxy, EmbeddingProxy]]:
        """
        Build or update the execution plan using centralized **reording** logic.

        The plan is simply the list of proxies sorted by their
        `rejection_efficiency` (see the shared `reording` helper).

        Args:
            proxies: Optional list of proxies to sort. If None, uses `self.proxies`.

        Returns:
            Sorted list of proxies (highest rejection efficiency first).
        """
        proxies_to_order = proxies or self.proxies

        # Delegate ordering to shared utility so the same logic can be reused
        # by other components or experiments.
        ordered = reording(proxies_to_order)  # type: ignore[arg-type]
        self._execution_plan = ordered

        if self.config.verbose:
            logging.info("[ProxyExecutor] Execution plan optimized via reording:")
            for i, proxy in enumerate(self._execution_plan):
                logging.info(
                    f"  {i + 1}. {proxy.name}: "
                    f"efficiency={proxy.rejection_efficiency:.4f}, "
                    f"pass_rate={proxy.pass_rate:.4f}, "
                    f"cost={proxy.cost:.4f}",
                )

        return self._execution_plan
    
    def process_batch(
        self,
        batch: DocumentBatch,
        predicates: Optional[Dict[str, Callable[[Any], bool]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        attributes: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], ExecutionStats]:
        """
        Process a batch of documents through the proxy pipeline.
        
        Args:
            batch: DocumentBatch with documents and embeddings
            predicates: Dictionary mapping attribute names to predicate functions
            schema: Schema definition for LLM extraction
            attributes: List of attributes to extract
            
        Returns:
            (results, stats) tuple where:
            - results: List of extraction results for documents that passed
            - stats: ExecutionStats with detailed metrics
        """
        stats = ExecutionStats(total_documents=len(batch))
        stats.all_doc_ids = list(batch.doc_ids)
        results = []
        
        # Track which documents are still active (not yet rejected)
        active_mask = np.ones(len(batch), dtype=bool)
        
        # Track proxy scores for potential hard negatives
        all_proxy_scores: Dict[int, Dict[str, float]] = {i: {} for i in range(len(batch))}
        
        # Track total proxies run per document
        proxies_run_per_doc = np.zeros(len(batch), dtype=int)
        
        # ====================================================================
        # Phase 1: Cascaded Proxy Filtering
        # ====================================================================
        proxy_start_time = time.perf_counter()
        
        for proxy in self._execution_plan:
            if not active_mask.any():
                # All documents rejected, no need to continue
                break
            
            # Get embeddings/documents for active documents only
            active_idx = np.where(active_mask)[0]
            active_embeddings = batch.embeddings[active_idx]

            # Evaluate proxy (support both embedding-based and document-based proxies)
            if getattr(proxy, "uses_documents", False):
                active_documents = [batch.documents[i] for i in active_idx]
                active_doc_ids = [batch.doc_ids[i] for i in active_idx]
                scores, passed = proxy.evaluate_documents(active_documents, doc_ids=active_doc_ids)
            else:
                scores, passed = proxy.evaluate(active_embeddings)
            
            # Update proxy statistics
            if proxy.name not in stats.proxy_stats:
                stats.proxy_stats[proxy.name] = {
                    "evaluated": 0,
                    "passed": 0,
                    "rejected": 0,
                    "avg_score": 0.0,
                    "scores_sum": 0.0
                }
            
            gs = stats.proxy_stats[proxy.name]
            gs["evaluated"] += len(scores)
            gs["passed"] += passed.sum()
            gs["rejected"] += (~passed).sum()
            gs["scores_sum"] += scores.sum()
            gs["avg_score"] = gs["scores_sum"] / gs["evaluated"]
            
            # Store scores for hard negative analysis
            for local_idx, global_idx in enumerate(active_idx):
                all_proxy_scores[global_idx][proxy.name] = float(scores[local_idx])
            
            # Update proxy-run count
            proxies_run_per_doc[active_idx] += 1
            
            # Update active mask: reject documents that failed this proxy.
            # CRITICAL: Documents with active_mask=False are NEVER added to results
            # and thus NEVER sent to LLM extraction.
            rejected_local = ~passed
            rejected_global = active_idx[rejected_local]
            active_mask[rejected_global] = False

            # Track rejected doc_ids per proxy for recall analysis
            if len(rejected_global) > 0:
                rejected_ids = [batch.doc_ids[i] for i in rejected_global]
                stats.proxy_rejected_doc_ids.setdefault(proxy.name, []).extend(rejected_ids)
            
            if self.config.verbose and len(rejected_global) > 0:
                logging.debug(f"[ProxyExecutor] Proxy '{proxy.name}' rejected "
                            f"{len(rejected_global)} documents")
        
        stats.total_proxy_time_ms = (time.perf_counter() - proxy_start_time) * 1000
        stats.rejected_by_proxies = int((~active_mask).sum())
        stats.passed_to_oracle = int(active_mask.sum())
        stats.avg_proxies_per_doc = float(proxies_run_per_doc.mean())
        stats.passed_doc_ids = [batch.doc_ids[i] for i in np.where(active_mask)[0]]
        
        # Sanity: passed + rejected must equal batch size
        assert stats.passed_to_oracle + stats.rejected_by_proxies == len(batch), (
            f"Proxy filter invariant violated: passed={stats.passed_to_oracle}, "
            f"rejected={stats.rejected_by_proxies}, batch_size={len(batch)}"
        )
        
        # ====================================================================
        # Phase 2: LLM Oracle Extraction (for documents that passed all proxies)
        # ====================================================================
        # ONLY documents in passed_indices are processed. Rejected docs are skipped.
        oracle_start_time = time.perf_counter()
        
        passed_indices = np.where(active_mask)[0]
        
        for idx in passed_indices:
            doc_id, document, embedding = batch[idx]
            
            if self.llm_oracle is None:
                # No oracle configured, just return passed documents
                results.append({
                    "doc_id": doc_id,
                    "passed_proxies": True,
                    "proxy_scores": all_proxy_scores[idx],
                    "extracted": None
                })
                stats.oracle_accepted += 1
                continue
            
            # Invoke LLM Oracle
            try:
                extracted = self.llm_oracle.extract(
                    document=document,
                    schema=schema or {},
                    attributes=attributes or []
                )
                
                # Check predicates if provided
                if predicates:
                    all_passed, per_attr = self.llm_oracle.check_predicates(
                        extracted_values=extracted,
                        predicates=predicates
                    )
                    
                    if all_passed:
                        stats.oracle_accepted += 1
                        results.append({
                            "doc_id": doc_id,
                            "passed_proxies": True,
                            "passed_predicates": True,
                            "proxy_scores": all_proxy_scores[idx],
                            "extracted": extracted
                        })
                    else:
                        # HARD NEGATIVE: Passed proxies but failed predicates
                        stats.oracle_rejected += 1
                        
                        if self.config.collect_hard_negatives:
                            self._collect_hard_negative(
                                doc_id=doc_id,
                                document=document,
                                embedding=embedding,
                                extracted=extracted,
                                per_attr_results=per_attr,
                                proxy_scores=all_proxy_scores[idx],
                                predicates=predicates
                            )
                        
                        results.append({
                            "doc_id": doc_id,
                            "passed_proxies": True,
                            "passed_predicates": False,
                            "proxy_scores": all_proxy_scores[idx],
                            "extracted": extracted,
                            "failed_predicates": [k for k, v in per_attr.items() if not v]
                        })
                else:
                    # No predicates, accept all extracted results
                    stats.oracle_accepted += 1
                    results.append({
                        "doc_id": doc_id,
                        "passed_proxies": True,
                        "proxy_scores": all_proxy_scores[idx],
                        "extracted": extracted
                    })
                    
            except Exception as e:
                logging.warning(f"[ProxyExecutor] Oracle extraction failed for {doc_id}: {e}")
                results.append({
                    "doc_id": doc_id,
                    "passed_proxies": True,
                    "error": str(e),
                    "proxy_scores": all_proxy_scores[idx]
                })
        
        stats.total_oracle_time_ms = (time.perf_counter() - oracle_start_time) * 1000
        
        return results, stats
    
    def process_single(
        self,
        doc_id: str,
        document: str,
        embedding: np.ndarray,
        predicates: Optional[Dict[str, Callable[[Any], bool]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        attributes: Optional[List[str]] = None
    ) -> Tuple[Optional[Dict[str, Any]], List[ProxyDecision]]:
        """
        Process a single document through the proxy pipeline.
        
        Useful for streaming processing or debugging.
        
        Args:
            doc_id: Document identifier
            document: Raw document text
            embedding: Pre-computed embedding
            predicates: Predicate functions for validation
            schema: Schema for LLM extraction
            attributes: Attributes to extract
            
        Returns:
            (result, proxy_results) tuple where:
            - result: Extraction result dict or None if rejected
            - proxy_results: List of ProxyDecision objects showing each proxy's decision
        """
        proxy_results = []
        proxy_scores = {}
        
        # Ensure embedding is 2D for batch processing
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Run proxies in execution plan order (short-circuit on reject)
        for proxy in self._execution_plan:
            start_time = time.perf_counter()
            
            scores, passed = proxy.evaluate(embedding)
            score = float(scores[0])
            did_pass = bool(passed[0])
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            proxy_scores[proxy.name] = score
            
            decision = FilterDecision.PASS if did_pass else FilterDecision.REJECT
            proxy_results.append(ProxyDecision(
                proxy_name=proxy.name,
                score=score,
                threshold=proxy.threshold,
                decision=decision,
                latency_ms=latency_ms
            ))
            
            if not did_pass:
                # Short-circuit: document rejected
                return None, proxy_results
        
        # Document passed all proxies, invoke oracle
        if self.llm_oracle is None:
            return {
                "doc_id": doc_id,
                "passed_proxies": True,
                "proxy_scores": proxy_scores,
                "extracted": None
            }, proxy_results
        
        try:
            extracted = self.llm_oracle.extract(
                document=document,
                schema=schema or {},
                attributes=attributes or []
            )
            
            if predicates:
                all_passed, per_attr = self.llm_oracle.check_predicates(
                    extracted_values=extracted,
                    predicates=predicates
                )
                
                if not all_passed:
                    # Hard negative
                    if self.config.collect_hard_negatives:
                        self._collect_hard_negative(
                            doc_id=doc_id,
                            document=document,
                            embedding=embedding.squeeze(),
                            extracted=extracted,
                            per_attr_results=per_attr,
                            proxy_scores=proxy_scores,
                            predicates=predicates
                        )
                    
                    return {
                        "doc_id": doc_id,
                        "passed_proxies": True,
                        "passed_predicates": False,
                        "proxy_scores": proxy_scores,
                        "extracted": extracted,
                        "failed_predicates": [k for k, v in per_attr.items() if not v]
                    }, proxy_results
            
            return {
                "doc_id": doc_id,
                "passed_proxies": True,
                "passed_predicates": True,
                "proxy_scores": proxy_scores,
                "extracted": extracted
            }, proxy_results
            
        except Exception as e:
            logging.warning(f"[ProxyExecutor] Oracle extraction failed for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "passed_proxies": True,
                "error": str(e),
                "proxy_scores": proxy_scores
            }, proxy_results
    
    def _collect_hard_negative(
        self,
        doc_id: str,
        document: str,
        embedding: np.ndarray,
        extracted: Dict[str, Any],
        per_attr_results: Dict[str, bool],
        proxy_scores: Dict[str, float],
        predicates: Dict[str, Callable[[Any], bool]]
    ):
        """
        Collect a hard negative sample for the feedback loop.
        
        Hard negatives are documents that:
        1. Passed all proxies (proxy models predicted positive)
        2. Failed LLM predicate check (actual extraction doesn't match query)
        
        These are valuable for retraining proxies to be more accurate.
        """
        # Find which attributes failed
        for attr_name, passed in per_attr_results.items():
            if not passed:
                hard_neg = HardNegative(
                    doc_id=doc_id,
                    document=document,
                    embedding=embedding.copy() if isinstance(embedding, np.ndarray) else embedding,
                    failed_attribute=attr_name,
                    extracted_value=extracted.get(attr_name),
                    expected_predicate=str(predicates.get(attr_name, "unknown")),
                    proxy_scores=proxy_scores.copy()
                )
                
                self._hard_negatives.append(hard_neg)
                
                # Enforce buffer size limit
                if len(self._hard_negatives) > self.config.hard_negative_buffer_size:
                    # Remove oldest entries
                    self._hard_negatives = self._hard_negatives[-self.config.hard_negative_buffer_size:]
                
                if self.config.verbose:
                    logging.info(f"[ProxyExecutor] Collected hard negative: "
                               f"doc={doc_id}, attr={attr_name}, "
                               f"value={extracted.get(attr_name)}")
    
    def get_hard_negatives(
        self, 
        attribute: Optional[str] = None
    ) -> List[HardNegative]:
        """
        Get collected hard negatives for retraining.
        
        Args:
            attribute: If provided, filter to hard negatives for this attribute
            
        Returns:
            List of HardNegative objects
        """
        if attribute is None:
            return self._hard_negatives.copy()
        
        return [hn for hn in self._hard_negatives if hn.failed_attribute == attribute]
    
    def get_hard_negative_training_data(
        self,
        attribute: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get hard negatives as training data (embeddings, labels) for retraining.
        
        All hard negatives for the attribute get label=0 (negative class).
        
        Args:
            attribute: Attribute name to get data for
            
        Returns:
            (embeddings, labels) tuple for training
        """
        hard_negs = self.get_hard_negatives(attribute)
        
        if not hard_negs:
            return np.array([]), np.array([])
        
        embeddings = np.stack([hn.embedding for hn in hard_negs])
        labels = np.zeros(len(hard_negs), dtype=np.float32)  # All negatives
        
        return embeddings, labels
    
    def clear_hard_negatives(self):
        """Clear the hard negative buffer."""
        self._hard_negatives.clear()
    
    def update_proxy_pass_rates(self):
        """
        Update proxy pass rates from runtime statistics.
        
        Call this periodically to adapt the execution plan based on observed data.
        """
        for proxy in self.proxies:
            if proxy._total_seen > 0:
                empirical_pass_rate = proxy._total_passed / proxy._total_seen
                logging.debug(f"[ProxyExecutor] Proxy '{proxy.name}' empirical pass rate: "
                            f"{empirical_pass_rate:.4f} (from {proxy._total_seen} samples)")
    
    def reset_proxy_stats(self):
        """Reset all proxy runtime statistics."""
        for proxy in self.proxies:
            proxy.reset_stats()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution plan."""
        return {
            "num_proxies": len(self._execution_plan),
            "execution_order": [
                {
                    "name": g.name,
                    "rejection_efficiency": g.rejection_efficiency,
                    "pass_rate": g.pass_rate,
                    "cost": g.cost,
                    "threshold": g.threshold
                }
                for g in self._execution_plan
            ],
            "hard_negatives_collected": len(self._hard_negatives)
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_proxy_from_classifier(
    name: str,
    classifier: "nn.Module",
    calibration_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    target_recall: float = 0.95,
    cost: float = 1.0,
    device: str = "cpu"
) -> ConformalProxy:
    """
    Factory function to create a ConformalProxy from a trained classifier.
    
    Args:
        name: Proxy name (usually attribute name)
        classifier: Trained PyTorch classifier
        calibration_data: Optional (embeddings, labels) tuple for calibration
        target_recall: Target recall for threshold calibration
        cost: Computational cost of this proxy
        device: Device for inference
        
    Returns:
        Configured ConformalProxy
    """
    proxy = ConformalProxy(
        name=name,
        classifier=classifier,
        threshold=0.5,  # Will be calibrated if data provided
        cost=cost,
        pass_rate=0.5,  # Will be updated from calibration
        device=device
    )
    
    if calibration_data is not None:
        embeddings, labels = calibration_data
        proxy.calibrate(embeddings, labels, target_recall=target_recall)
        
        # Estimate pass rate from calibration data
        scores, passed = proxy.evaluate(embeddings)
        proxy._pass_rate = passed.mean()
    
    return proxy


def create_proxy_executor_from_classifiers(
    classifiers: Dict[str, "nn.Module"],
    calibration_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    costs: Optional[Dict[str, float]] = None,
    target_recall: float = 0.95,
    llm_oracle: Optional[LLMOracleProtocol] = None,
    config: Optional[ProxyRuntimeConfig] = None,
    device: str = "cpu"
) -> ProxyExecutor:
    """
    Factory function to create ProxyExecutor from a dictionary of classifiers.
    
    Args:
        classifiers: Dict mapping attribute names to trained classifiers
        calibration_data: Dict mapping attribute names to (embeddings, labels) tuples
        costs: Dict mapping attribute names to costs (default 1.0)
        target_recall: Target recall for calibration
        llm_oracle: LLM Oracle for extraction
        config: ProxyRuntimeConfig
        device: Device for inference
        
    Returns:
        Configured ProxyExecutor
    """
    costs = costs or {}
    calibration_data = calibration_data or {}
    
    proxies = []
    for attr_name, classifier in classifiers.items():
        cal_data = calibration_data.get(attr_name)
        cost = costs.get(attr_name, 1.0)
        
        proxy = create_proxy_from_classifier(
            name=attr_name,
            classifier=classifier,
            calibration_data=cal_data,
            target_recall=target_recall,
            cost=cost,
            device=device
        )
        proxies.append(proxy)
    
    return ProxyExecutor(proxies=proxies, llm_oracle=llm_oracle, config=config)
