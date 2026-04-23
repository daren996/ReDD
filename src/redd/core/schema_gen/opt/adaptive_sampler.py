"""
Adaptive Sampling Algorithm for Schema Generation.

This module implements the two-phase adaptive sampling algorithm that reduces
the number of document processing steps (LLM calls) while maintaining
probabilistic guarantees on schema quality.

The algorithm tracks schema entropy and stops sampling when entropy falls
below a stability threshold for a consecutive number of iterations.

References:
    Nick's Notes July 23, 2025 - Adaptive Sampling
"""

import logging
import math
from typing import Any, Dict, List, Optional, Callable
from .schema_entropy import SchemaEntropyCalculator


class AdaptiveSampler:
    """
    Adaptive sampling controller for schema generation.
    
    Implements the AdaptiveSchemaExtraction algorithm with:
    - Schema entropy tracking
    - Stability streak monitoring
    - Probabilistic early stopping
    - Feature coverage guarantees
    """
    
    def __init__(
        self,
        theta: float = 0.05,
        m: int = 3,
        n_min: int = 5,
        delta: float = 0.1,
        epsilon: float = 0.05,
        entropy_calculator: Optional[SchemaEntropyCalculator] = None,
        enable_probabilistic_stop: bool = True
    ):
        """
        Initialize the adaptive sampler.
        
        Args:
            theta: Entropy threshold for stability (default 0.05)
            m: Stability streak threshold - number of consecutive low entropy
               iterations required (default 3)
            n_min: Minimum samples to process before early stopping (default 5)
            delta: Allowed failure probability for coverage guarantee (default 0.1)
            epsilon: Minimum feature frequency for coverage (default 0.05)
            entropy_calculator: Optional custom entropy calculator
            enable_probabilistic_stop: Whether to use probabilistic stopping
                                      criterion (default True)
        """
        self.theta = theta
        self.m = m
        self.n_min = n_min
        self.delta = delta
        self.epsilon = epsilon
        self.enable_probabilistic_stop = enable_probabilistic_stop
        
        # Initialize entropy calculator
        self.entropy_calculator = entropy_calculator or SchemaEntropyCalculator()
        
        # State tracking
        self.low_entropy_streak = 0
        self.n_processed = 0
        self.should_stop = False
        self.stop_reason = None
        
        logging.info(
            f"[{self.__class__.__name__}:__init__] Initialized with parameters: "
            f"theta={theta}, m={m}, n_min={n_min}, delta={delta}, epsilon={epsilon}, "
            f"probabilistic_stop={enable_probabilistic_stop}"
        )
    
    def should_continue(self, current_schema: Any) -> bool:
        """
        Determine whether to continue processing more documents.
        
        This is the main decision function that implements the stopping criteria
        from the AdaptiveSchemaExtraction algorithm.
        
        Args:
            current_schema: Current state of the schema
            
        Returns:
            True if processing should continue, False if should stop
        """
        # Compute entropy for current iteration
        entropy = self.entropy_calculator.compute_entropy(current_schema)
        
        # Update streak counter
        if entropy < self.theta:
            self.low_entropy_streak += 1
        else:
            self.low_entropy_streak = 0
        
        # Increment processed count
        self.n_processed += 1
        
        logging.info(
            f"[{self.__class__.__name__}:should_continue] "
            f"Iteration {self.n_processed}: entropy={entropy:.4f}, "
            f"streak={self.low_entropy_streak}/{self.m}, "
            f"features={self.entropy_calculator.get_feature_count()}"
        )
        
        # Check stopping criteria
        
        # Criterion 1: Stability streak + minimum samples
        if self.low_entropy_streak >= self.m and self.n_processed >= self.n_min:
            self.should_stop = True
            self.stop_reason = (
                f"stability_streak: streak={self.low_entropy_streak} >= m={self.m}, "
                f"processed={self.n_processed} >= n_min={self.n_min}"
            )
            logging.info(
                f"[{self.__class__.__name__}:should_continue] "
                f"Stopping due to {self.stop_reason}"
            )
            return False
        
        # Criterion 2: Probabilistic stopping (optional)
        if self.enable_probabilistic_stop:
            feature_count = self.entropy_calculator.get_feature_count()
            if self._probabilistic_stop(self.n_processed, feature_count):
                self.should_stop = True
                self.stop_reason = (
                    f"probabilistic_stop: processed={self.n_processed}, "
                    f"features={feature_count}, delta={self.delta}"
                )
                logging.info(
                    f"[{self.__class__.__name__}:should_continue] "
                    f"Stopping due to {self.stop_reason}"
                )
                return False
        
        # Continue processing
        return True
    
    def _probabilistic_stop(self, n_processed: int, feature_est: int) -> bool:
        """
        Probabilistic stopping criterion.
        
        Implements the ProbabilisticStop algorithm:
            If (1 - ε)^n_processed ≤ δ / ℱ_est, return True
        
        This ensures that all features appearing with probability ≥ ε
        are observed with probability ≥ (1 - δ).
        
        Args:
            n_processed: Number of documents processed so far
            feature_est: Estimated feature count seen so far
            
        Returns:
            True if should stop, False otherwise
        """
        if feature_est == 0:
            return False
        
        # Compute (1 - ε)^n_processed
        prob_missing = math.pow(1 - self.epsilon, n_processed)
        
        # Compute threshold δ / ℱ_est
        threshold = self.delta / feature_est
        
        should_stop = prob_missing <= threshold
        
        if should_stop:
            logging.debug(
                f"[{self.__class__.__name__}:_probabilistic_stop] "
                f"Probabilistic stop triggered: "
                f"(1-ε)^n={(1-self.epsilon):.3f}^{n_processed}={prob_missing:.6f} "
                f"<= δ/F={self.delta}/{feature_est}={threshold:.6f}"
            )
        
        return should_stop
    
    def compute_minimum_samples(self, feature_universe_size: int) -> int:
        """
        Compute the minimum number of samples required for feature coverage.
        
        Implements the formula:
            n_min = ⌈log(|ℱ|/δ) / ε⌉
        
        This ensures that all features appearing with probability ≥ ε
        are observed with probability ≥ (1 - δ).
        
        Args:
            feature_universe_size: Size of the feature universe |ℱ|
            
        Returns:
            Minimum number of samples required
        """
        if feature_universe_size <= 0:
            return self.n_min
        
        numerator = math.log(feature_universe_size / self.delta)
        n_min_computed = math.ceil(numerator / self.epsilon)
        
        logging.info(
            f"[{self.__class__.__name__}:compute_minimum_samples] "
            f"Computed n_min={n_min_computed} for |F|={feature_universe_size}, "
            f"δ={self.delta}, ε={self.epsilon}"
        )
        
        return max(n_min_computed, self.n_min)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the adaptive sampling process.
        
        Returns:
            Dictionary containing sampling statistics
        """
        entropy_stats = self.entropy_calculator.get_statistics()
        
        return {
            "n_processed": self.n_processed,
            "low_entropy_streak": self.low_entropy_streak,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
            "parameters": {
                "theta": self.theta,
                "m": self.m,
                "n_min": self.n_min,
                "delta": self.delta,
                "epsilon": self.epsilon,
                "probabilistic_stop_enabled": self.enable_probabilistic_stop
            },
            "entropy_statistics": entropy_stats,
            "entropy_history": self.entropy_calculator.get_entropy_history()
        }
    
    def reset(self):
        """Reset the sampler to initial state."""
        self.low_entropy_streak = 0
        self.n_processed = 0
        self.should_stop = False
        self.stop_reason = None
        self.entropy_calculator.reset()
        
        logging.info(f"[{self.__class__.__name__}:reset] Adaptive sampler reset")
    
    def get_stop_reason(self) -> Optional[str]:
        """
        Get the reason for stopping (if stopped).
        
        Returns:
            String describing the stop reason, or None if not stopped
        """
        return self.stop_reason
    
    def estimate_remaining_documents(self, total_documents: int) -> int:
        """
        Estimate how many more documents need to be processed.
        
        This is a heuristic based on current entropy and streak.
        
        Args:
            total_documents: Total number of documents available
            
        Returns:
            Estimated number of remaining documents to process
        """
        if self.should_stop:
            return 0
        
        # If we haven't reached minimum samples, need at least that many more
        if self.n_processed < self.n_min:
            min_remaining = self.n_min - self.n_processed
        else:
            min_remaining = 0
        
        # If entropy is high, we likely need more documents
        if self.entropy_calculator.entropy_history:
            recent_entropy = self.entropy_calculator.entropy_history[-1]
            if recent_entropy > self.theta:
                # Need at least m more to build streak
                streak_remaining = self.m - self.low_entropy_streak
            else:
                # Already in streak, need to complete it
                streak_remaining = max(0, self.m - self.low_entropy_streak)
        else:
            streak_remaining = self.m
        
        estimated = max(min_remaining, streak_remaining)
        
        # Cap at available documents
        available = total_documents - self.n_processed
        estimated = min(estimated, available)
        
        return estimated

