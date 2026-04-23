"""
Tests for CCG Executor

Run with: python -m pytest src/redd/proxy/proxy_runtime/test_proxy_executor.py -v
"""

import numpy as np
import pytest
from typing import Any, Callable, Dict, List, Tuple

from .executor import (
    ProxyExecutor,
    ProxyRuntimeConfig,
    ConformalProxy,
    DocumentBatch,
    ExecutionStats,
    HardNegative,
    create_proxy_from_classifier,
    FilterDecision,
)

# Try importing torch for advanced tests
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockClassifier:
    """Simple mock classifier that returns pre-defined scores."""
    
    def __init__(self, scores_fn: Callable[[np.ndarray], np.ndarray]):
        self.scores_fn = scores_fn
    
    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        return self.scores_fn(embeddings)


class MockLLMOracle:
    """Mock LLM Oracle for testing."""
    
    def __init__(
        self, 
        extraction_map: Dict[str, Dict[str, Any]] = None,
        default_values: Dict[str, Any] = None
    ):
        self.extraction_map = extraction_map or {}
        self.default_values = default_values or {"price": 100, "category": "unknown"}
        self.call_count = 0
    
    def extract(
        self, 
        document: str, 
        schema: Dict[str, Any],
        attributes: List[str]
    ) -> Dict[str, Any]:
        self.call_count += 1
        
        # Check if we have a specific extraction for this document
        if document in self.extraction_map:
            return self.extraction_map[document]
        
        return self.default_values.copy()
    
    def check_predicates(
        self,
        extracted_values: Dict[str, Any],
        predicates: Dict[str, Callable[[Any], bool]]
    ) -> Tuple[bool, Dict[str, bool]]:
        per_attr = {}
        all_passed = True
        
        for attr_name, predicate in predicates.items():
            value = extracted_values.get(attr_name)
            passed = predicate(value) if value is not None else False
            per_attr[attr_name] = passed
            if not passed:
                all_passed = False
        
        return all_passed, per_attr


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 128).astype(np.float32)


@pytest.fixture
def sample_batch(sample_embeddings):
    """Create a sample document batch."""
    return DocumentBatch(
        doc_ids=[f"doc_{i}" for i in range(len(sample_embeddings))],
        documents=[f"Document content {i}" for i in range(len(sample_embeddings))],
        embeddings=sample_embeddings
    )


@pytest.fixture
def simple_guard():
    """Create a simple guard that passes 50% of documents."""
    def score_fn(embeddings):
        # Score based on first embedding dimension
        return 1 / (1 + np.exp(-embeddings[:, 0]))  # Sigmoid of first dim
    
    return ConformalProxy(
        name="test_guard",
        classifier=MockClassifier(score_fn),
        threshold=0.5,
        cost=1.0,
        pass_rate=0.5
    )


# ============================================================================
# Unit Tests
# ============================================================================

class TestConformalGuard:
    """Tests for ConformalProxy class."""
    
    def test_initialization(self):
        """Test guard initialization."""
        def score_fn(emb):
            return np.ones(len(emb)) * 0.7
        
        guard = ConformalProxy(
            name="test",
            classifier=MockClassifier(score_fn),
            threshold=0.5,
            cost=2.0,
            pass_rate=0.3
        )
        
        assert guard.name == "test"
        assert guard.threshold == 0.5
        assert guard.cost == 2.0
        assert guard.pass_rate == 0.3
    
    def test_rejection_efficiency(self):
        """Test rejection efficiency calculation."""
        def score_fn(emb):
            return np.ones(len(emb)) * 0.7
        
        # High rejection efficiency: low pass rate, low cost
        guard = ConformalProxy(
            name="efficient",
            classifier=MockClassifier(score_fn),
            threshold=0.5,
            cost=0.5,
            pass_rate=0.2
        )
        
        # Efficiency = (1 - 0.2) / 0.5 = 1.6
        assert guard.rejection_efficiency == pytest.approx(1.6, rel=1e-3)
        
        # Low rejection efficiency: high pass rate, high cost
        guard2 = ConformalProxy(
            name="inefficient",
            classifier=MockClassifier(score_fn),
            threshold=0.5,
            cost=2.0,
            pass_rate=0.8
        )
        
        # Efficiency = (1 - 0.8) / 2.0 = 0.1
        assert guard2.rejection_efficiency == pytest.approx(0.1, rel=1e-3)
    
    def test_predict(self, sample_embeddings):
        """Test prediction method."""
        def score_fn(emb):
            return np.ones(len(emb)) * 0.7
        
        guard = ConformalProxy(
            name="test",
            classifier=MockClassifier(score_fn),
            threshold=0.5
        )
        
        scores = guard.predict(sample_embeddings)
        
        assert len(scores) == len(sample_embeddings)
        assert np.allclose(scores, 0.7)
    
    def test_evaluate(self, sample_embeddings):
        """Test evaluation with threshold."""
        def score_fn(emb):
            # First half gets high scores, second half gets low scores
            scores = np.zeros(len(emb))
            scores[:len(emb)//2] = 0.8
            scores[len(emb)//2:] = 0.2
            return scores
        
        guard = ConformalProxy(
            name="test",
            classifier=MockClassifier(score_fn),
            threshold=0.5
        )
        
        scores, passed = guard.evaluate(sample_embeddings)
        
        assert len(scores) == len(sample_embeddings)
        assert passed[:5].all()  # First 5 should pass
        assert not passed[5:].any()  # Last 5 should fail
    
    def test_calibrate(self, sample_embeddings):
        """Test threshold calibration."""
        # Create scores where positive samples have scores [0.3, 0.4, 0.5, 0.6, 0.7]
        def score_fn(emb):
            return np.linspace(0.3, 0.7, len(emb))
        
        guard = ConformalProxy(
            name="test",
            classifier=MockClassifier(score_fn),
            threshold=0.5
        )
        
        # All samples are positive
        labels = np.ones(len(sample_embeddings))
        
        # Calibrate for 95% recall
        threshold = guard.calibrate(sample_embeddings, labels, target_recall=0.95)
        
        # Threshold should be at 5th percentile of scores
        # With 10 samples, 5th percentile is close to minimum
        assert threshold <= 0.35


class TestCCGExecutor:
    """Tests for ProxyExecutor class."""
    
    def test_optimize_execution_plan(self):
        """Test that guards are sorted by rejection efficiency."""
        def score_fn(emb):
            return np.ones(len(emb)) * 0.7
        
        guards = [
            ConformalProxy("low_eff", MockClassifier(score_fn), cost=2.0, pass_rate=0.8),
            ConformalProxy("high_eff", MockClassifier(score_fn), cost=0.5, pass_rate=0.2),
            ConformalProxy("mid_eff", MockClassifier(score_fn), cost=1.0, pass_rate=0.5),
        ]
        
        executor = ProxyExecutor(guards=guards)
        
        # Should be sorted: high_eff (1.6), mid_eff (0.5), low_eff (0.1)
        assert executor._execution_plan[0].name == "high_eff"
        assert executor._execution_plan[1].name == "mid_eff"
        assert executor._execution_plan[2].name == "low_eff"
    
    def test_process_batch_all_pass(self, sample_batch):
        """Test batch processing when all guards pass."""
        def high_score_fn(emb):
            return np.ones(len(emb)) * 0.9
        
        guards = [
            ConformalProxy("guard1", MockClassifier(high_score_fn), threshold=0.5),
            ConformalProxy("guard2", MockClassifier(high_score_fn), threshold=0.5),
        ]
        
        executor = ProxyExecutor(guards=guards)
        results, stats = executor.process_batch(sample_batch)
        
        assert stats.total_documents == 10
        assert stats.rejected_by_guards == 0
        assert stats.passed_to_oracle == 10
    
    def test_process_batch_all_reject(self, sample_batch):
        """Test batch processing when first guard rejects all."""
        def low_score_fn(emb):
            return np.ones(len(emb)) * 0.1
        
        guards = [
            ConformalProxy("guard1", MockClassifier(low_score_fn), threshold=0.5),
            ConformalProxy("guard2", MockClassifier(low_score_fn), threshold=0.5),
        ]
        
        executor = ProxyExecutor(guards=guards)
        results, stats = executor.process_batch(sample_batch)
        
        assert stats.total_documents == 10
        assert stats.rejected_by_guards == 10
        assert stats.passed_to_oracle == 0
        
        # Second guard should not have been evaluated
        assert "guard2" not in stats.proxy_stats or stats.proxy_stats["guard2"]["evaluated"] == 0
    
    def test_process_batch_partial_reject(self, sample_batch):
        """Test batch processing with partial rejection."""
        def partial_score_fn(emb):
            # First half passes, second half fails
            scores = np.zeros(len(emb))
            scores[:len(emb)//2] = 0.8
            scores[len(emb)//2:] = 0.2
            return scores
        
        guards = [
            ConformalProxy("guard1", MockClassifier(partial_score_fn), threshold=0.5),
        ]
        
        executor = ProxyExecutor(guards=guards)
        results, stats = executor.process_batch(sample_batch)
        
        assert stats.total_documents == 10
        assert stats.rejected_by_guards == 5
        assert stats.passed_to_oracle == 5
    
    def test_process_batch_with_oracle(self, sample_batch):
        """Test batch processing with LLM oracle."""
        def high_score_fn(emb):
            return np.ones(len(emb)) * 0.9
        
        guards = [
            ConformalProxy("price_guard", MockClassifier(high_score_fn), threshold=0.5),
        ]
        
        oracle = MockLLMOracle(default_values={"price": 150})
        executor = ProxyExecutor(guards=guards, llm_oracle=oracle)
        
        predicates = {"price": lambda x: x > 100}
        
        results, stats = executor.process_batch(
            sample_batch,
            predicates=predicates,
            attributes=["price"]
        )
        
        assert oracle.call_count == 10  # All documents passed to oracle
        assert stats.oracle_accepted == 10  # All should pass predicate
    
    def test_hard_negative_collection(self, sample_batch):
        """Test hard negative collection."""
        def high_score_fn(emb):
            return np.ones(len(emb)) * 0.9
        
        guards = [
            ConformalProxy("price_guard", MockClassifier(high_score_fn), threshold=0.5),
        ]
        
        # Oracle returns price=50 which will fail the price>100 predicate
        oracle = MockLLMOracle(default_values={"price": 50})
        
        config = ProxyRuntimeConfig(collect_hard_negatives=True)
        executor = ProxyExecutor(guards=guards, llm_oracle=oracle, config=config)
        
        predicates = {"price": lambda x: x > 100}
        
        results, stats = executor.process_batch(
            sample_batch,
            predicates=predicates,
            attributes=["price"]
        )
        
        # All should be hard negatives (passed guards but failed predicate)
        assert stats.oracle_rejected == 10
        assert len(executor.get_hard_negatives()) == 10
        
        # Check hard negative attributes
        hard_negs = executor.get_hard_negatives("price")
        assert len(hard_negs) == 10
        assert all(hn.failed_attribute == "price" for hn in hard_negs)
    
    def test_process_single(self, sample_batch):
        """Test single document processing."""
        def high_score_fn(emb):
            return np.ones(len(emb)) * 0.9
        
        guards = [
            ConformalProxy("guard1", MockClassifier(high_score_fn), threshold=0.5),
        ]
        
        executor = ProxyExecutor(guards=guards)
        
        doc_id, document, embedding = sample_batch[0]
        result, proxy_results = executor.process_single(doc_id, document, embedding)
        
        assert result is not None
        assert result["passed_guards"] == True
        assert len(proxy_results) == 1
        assert proxy_results[0].decision == FilterDecision.PASS
    
    def test_short_circuit_on_reject(self, sample_batch):
        """Test that evaluation stops on first rejection."""
        call_counts = {"guard1": 0, "guard2": 0}
        
        def guard1_score_fn(emb):
            call_counts["guard1"] += len(emb)
            return np.ones(len(emb)) * 0.1  # Will reject
        
        def guard2_score_fn(emb):
            call_counts["guard2"] += len(emb)
            return np.ones(len(emb)) * 0.9
        
        guards = [
            ConformalProxy("guard1", MockClassifier(guard1_score_fn), 
                          threshold=0.5, cost=0.5, pass_rate=0.1),  # High efficiency
            ConformalProxy("guard2", MockClassifier(guard2_score_fn), 
                          threshold=0.5, cost=2.0, pass_rate=0.9),  # Low efficiency
        ]
        
        executor = ProxyExecutor(guards=guards)
        
        # Process single document
        doc_id, document, embedding = sample_batch[0]
        result, proxy_results = executor.process_single(doc_id, document, embedding)
        
        # guard1 should be called first (higher efficiency) and reject
        assert call_counts["guard1"] == 1
        assert call_counts["guard2"] == 0  # Short-circuited
        assert result is None
    
    def test_get_hard_negative_training_data(self, sample_batch):
        """Test getting training data from hard negatives."""
        def high_score_fn(emb):
            return np.ones(len(emb)) * 0.9
        
        guards = [
            ConformalProxy("price_guard", MockClassifier(high_score_fn), threshold=0.5),
        ]
        
        oracle = MockLLMOracle(default_values={"price": 50})
        config = ProxyRuntimeConfig(collect_hard_negatives=True)
        executor = ProxyExecutor(guards=guards, llm_oracle=oracle, config=config)
        
        predicates = {"price": lambda x: x > 100}
        executor.process_batch(sample_batch, predicates=predicates)
        
        embeddings, labels = executor.get_hard_negative_training_data("price")
        
        assert len(embeddings) == 10
        assert len(labels) == 10
        assert all(labels == 0)  # All hard negatives get label 0


class TestDocumentBatch:
    """Tests for DocumentBatch class."""
    
    def test_len(self, sample_batch):
        """Test batch length."""
        assert len(sample_batch) == 10
    
    def test_getitem(self, sample_batch):
        """Test indexing."""
        doc_id, document, embedding = sample_batch[0]
        assert doc_id == "doc_0"
        assert document == "Document content 0"
        assert embedding.shape == (128,)
    
    def test_get_embeddings_numpy(self, sample_embeddings):
        """Test numpy embedding extraction."""
        batch = DocumentBatch(
            doc_ids=["doc_0"],
            documents=["test"],
            embeddings=sample_embeddings
        )
        
        np_emb = batch.get_embeddings_numpy()
        assert isinstance(np_emb, np.ndarray)
        assert np.allclose(np_emb, sample_embeddings)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_embeddings_torch(self, sample_embeddings):
        """Test torch embedding extraction."""
        batch = DocumentBatch(
            doc_ids=["doc_0"],
            documents=["test"],
            embeddings=sample_embeddings
        )
        
        torch_emb = batch.get_embeddings_torch()
        assert isinstance(torch_emb, torch.Tensor)
        assert torch_emb.shape == sample_embeddings.shape


class TestExecutionStats:
    """Tests for ExecutionStats class."""
    
    def test_guard_rejection_rate(self):
        """Test rejection rate calculation."""
        stats = ExecutionStats(
            total_documents=100,
            rejected_by_guards=30,
            passed_to_oracle=70
        )
        
        assert stats.proxy_rejection_rate == pytest.approx(0.3, rel=1e-3)
    
    def test_oracle_precision(self):
        """Test oracle precision calculation."""
        stats = ExecutionStats(
            total_documents=100,
            rejected_by_guards=30,
            passed_to_oracle=70,
            oracle_accepted=60,
            oracle_rejected=10
        )
        
        # Precision = 60 / 70
        assert stats.oracle_precision == pytest.approx(60/70, rel=1e-3)
    
    def test_cost_savings(self):
        """Test cost savings calculation."""
        stats = ExecutionStats(
            total_documents=100,
            rejected_by_guards=30
        )
        
        # Saved 30% of oracle calls
        assert stats.cost_savings == pytest.approx(0.3, rel=1e-3)


# ============================================================================
# Integration Tests with PyTorch
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchIntegration:
    """Tests using actual PyTorch classifiers."""
    
    def test_with_linear_classifier(self, sample_embeddings):
        """Test with a simple linear PyTorch classifier."""
        # Create simple linear classifier
        classifier = nn.Sequential(
            nn.Linear(128, 1)
        )
        
        # Initialize weights to give specific behavior
        with torch.no_grad():
            classifier[0].weight.fill_(0.1)
            classifier[0].bias.fill_(0.0)
        
        guard = ConformalProxy(
            name="linear_guard",
            classifier=classifier,
            threshold=0.5
        )
        
        scores = guard.predict(sample_embeddings)
        
        assert len(scores) == len(sample_embeddings)
        assert all(0 <= s <= 1 for s in scores)  # Scores should be probabilities
    
    def test_create_guard_from_classifier(self, sample_embeddings):
        """Test factory function with calibration."""
        classifier = nn.Sequential(
            nn.Linear(128, 1)
        )
        
        # Create calibration data
        labels = np.random.randint(0, 2, len(sample_embeddings)).astype(np.float32)
        
        guard = create_proxy_from_classifier(
            name="calibrated_guard",
            classifier=classifier,
            calibration_data=(sample_embeddings, labels),
            target_recall=0.9
        )
        
        assert guard.name == "calibrated_guard"
        assert guard.threshold != 0.5  # Should have been calibrated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
