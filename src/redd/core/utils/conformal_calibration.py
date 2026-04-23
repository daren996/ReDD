"""
Conformal Prediction Calibration Utilities.

This module provides reusable conformal prediction calibration functionality
for computing thresholds with statistical coverage guarantees.

Conformal Prediction Background:
- Given a calibration set of (score, label) pairs where label=1 means "relevant"
- Define nonconformity score: higher score = less conforming (less relevant)
- Compute threshold as the (1-alpha)(n+1)/n quantile of calibration scores
- Guarantee: With probability >= 1-alpha, future relevant items will have score <= threshold

Usage:
    ```python
    from core.utils.conformal_calibration import ConformalCalibrator, ConformalCalibrationResult
    
    # Create calibrator
    calibrator = ConformalCalibrator(target_recall=0.95)
    
    # Calibrate with scores from relevant items
    # Lower scores = more relevant (e.g., negative cosine similarity)
    calibration_scores = [-0.8, -0.7, -0.6, -0.5, -0.4]  # NC scores for relevant items
    result = calibrator.calibrate(calibration_scores)
    
    print(result.threshold)  # e.g., -0.42
    print(result.guarantee)  # "With probability >= 95.00%, ..."
    
    # Use threshold for filtering
    # Keep items where NC_score <= threshold (i.e., similarity >= -threshold)
    ```
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

__all__ = [
    "ConformalCalibrationResult",
    "ConformalCalibrator",
    "cosine_similarity",
    "nonconformity_score_negative_cosine",
]


@dataclass
class ConformalCalibrationResult:
    """
    Result of conformal prediction calibration.
    
    Attributes:
        threshold: The calibrated threshold τ_alpha (in nonconformity score space)
        alpha: The significance level (1 - target_recall)
        target_recall: The target recall rate (1 - alpha)
        guarantee: Statistical guarantee string
        num_calibration_samples: Number of samples used for calibration
        calibration_scores: The nonconformity scores from calibration set
        quantile_level: The actual quantile level used
        metadata: Additional metadata about the calibration
    """
    threshold: float
    alpha: float
    target_recall: float
    guarantee: str
    num_calibration_samples: int
    calibration_scores: List[float] = field(default_factory=list)
    quantile_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"ConformalCalibrationResult("
            f"threshold={self.threshold:.4f}, "
            f"target_recall={self.target_recall:.2%}, "
            f"n_samples={self.num_calibration_samples})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "target_recall": self.target_recall,
            "guarantee": self.guarantee,
            "num_calibration_samples": self.num_calibration_samples,
            "quantile_level": self.quantile_level,
            "metadata": self.metadata,
            # Note: calibration_scores not included by default (can be large)
        }
    
    def save(self, path: Union[str, Path], include_scores: bool = False) -> None:
        """
        Save calibration result to JSON file.
        
        Args:
            path: Path to save file
            include_scores: Whether to include calibration_scores (can be large)
        """
        data = self.to_dict()
        if include_scores:
            data["calibration_scores"] = self.calibration_scores
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"[ConformalCalibrationResult:save] Saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalCalibrationResult":
        """
        Load calibration result from JSON file.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded ConformalCalibrationResult
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(
            threshold=data["threshold"],
            alpha=data["alpha"],
            target_recall=data["target_recall"],
            guarantee=data["guarantee"],
            num_calibration_samples=data["num_calibration_samples"],
            calibration_scores=data.get("calibration_scores", []),
            quantile_level=data.get("quantile_level", 0.0),
            metadata=data.get("metadata", {}),
        )


class ConformalCalibrator:
    """
    Conformal prediction calibrator for computing thresholds with coverage guarantees.
    
    This calibrator computes a threshold τ_alpha such that with probability >= 1-alpha,
    future relevant items will have nonconformity score <= τ_alpha.
    
    The nonconformity score function should be defined such that:
    - Lower scores indicate higher relevance/conformity
    - Higher scores indicate lower relevance/conformity
    
    Common nonconformity functions:
    - Negative cosine similarity: NC(q, d) = -cos(E_q, E_d)
    - Distance: NC(q, d) = ||E_q - E_d||
    - 1 - similarity: NC(q, d) = 1 - sim(E_q, E_d)
    
    Example:
        ```python
        calibrator = ConformalCalibrator(target_recall=0.95)
        
        # Collect NC scores for relevant (query, doc) pairs
        nc_scores = []
        for query_emb, doc_emb in relevant_pairs:
            nc = -cosine_similarity(query_emb, doc_emb)
            nc_scores.append(nc)
        
        # Calibrate
        result = calibrator.calibrate(nc_scores)
        
        # Use: keep docs where NC_score <= result.threshold
        # Equivalent to: keep docs where similarity >= -result.threshold
        ```
    """
    
    DEFAULT_TARGET_RECALL = 0.95
    DEFAULT_THRESHOLD = 0.0  # In NC space, 0 means similarity >= 0
    
    def __init__(
        self,
        target_recall: float = DEFAULT_TARGET_RECALL,
        default_threshold: Optional[float] = None,
    ):
        """
        Initialize the conformal calibrator.
        
        Args:
            target_recall: Target recall rate (1 - alpha), default 0.95
            default_threshold: Default threshold to use when calibration fails
        """
        if not 0 < target_recall < 1:
            raise ValueError(f"target_recall must be in (0, 1), got {target_recall}")
        
        self.target_recall = target_recall
        self.alpha = 1.0 - target_recall
        self.default_threshold = (
            default_threshold if default_threshold is not None 
            else self.DEFAULT_THRESHOLD
        )
        
        logging.info(
            f"[{self.__class__.__name__}:__init__] Initialized with "
            f"target_recall={target_recall:.2%}, alpha={self.alpha:.4f}"
        )
    
    def calibrate(
        self,
        calibration_scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConformalCalibrationResult:
        """
        Calibrate threshold from nonconformity scores of relevant items.
        
        The calibration uses the conformal prediction formula:
        τ_alpha = quantile_{(1-alpha)(n+1)/n}(calibration_scores)
        
        This guarantees that with probability >= 1-alpha, a future relevant item
        will have NC score <= τ_alpha.
        
        Args:
            calibration_scores: List of nonconformity scores for relevant items.
                                These should be computed using a consistent NC function.
            metadata: Optional metadata to include in the result
            
        Returns:
            ConformalCalibrationResult with threshold and guarantee
        """
        if not calibration_scores:
            logging.warning(
                f"[{self.__class__.__name__}:calibrate] Empty calibration set. "
                f"Using default threshold."
            )
            return self._create_default_result(metadata)
        
        n = len(calibration_scores)
        scores_array = np.array(calibration_scores)
        
        # Compute conformal quantile level
        # For coverage guarantee of (1-alpha), use ceil((n+1)(1-alpha))/n quantile
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0
        
        # Compute threshold
        threshold = float(np.quantile(scores_array, quantile_level))
        
        # Create result
        result = ConformalCalibrationResult(
            threshold=threshold,
            alpha=self.alpha,
            target_recall=self.target_recall,
            guarantee=(
                f"With probability >= {self.target_recall:.2%}, relevant items "
                f"(from the same distribution) will be retained."
            ),
            num_calibration_samples=n,
            calibration_scores=list(calibration_scores),
            quantile_level=float(quantile_level),
            metadata=metadata or {},
        )
        
        logging.info(
            f"[{self.__class__.__name__}:calibrate] Calibration complete. "
            f"threshold={threshold:.4f}, quantile_level={quantile_level:.4f}, n={n}"
        )
        
        return result
    
    def calibrate_from_similarities(
        self,
        similarities: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConformalCalibrationResult:
        """
        Calibrate from similarity scores (convenience method).
        
        Converts similarities to nonconformity scores using NC = -similarity,
        then performs calibration.
        
        Args:
            similarities: List of similarity scores for relevant items (higher = more similar)
            metadata: Optional metadata
            
        Returns:
            ConformalCalibrationResult with threshold in similarity space
            (threshold is the minimum similarity to retain)
        """
        # Convert to NC scores
        nc_scores = [-s for s in similarities]
        
        # Calibrate
        result = self.calibrate(nc_scores, metadata)
        
        # Convert threshold back to similarity space
        # NC threshold: keep if NC <= threshold
        # Similarity threshold: keep if sim >= -NC_threshold
        similarity_threshold = -result.threshold
        
        # Create new result with similarity threshold
        return ConformalCalibrationResult(
            threshold=similarity_threshold,
            alpha=result.alpha,
            target_recall=result.target_recall,
            guarantee=result.guarantee,
            num_calibration_samples=result.num_calibration_samples,
            calibration_scores=similarities,  # Store original similarities
            quantile_level=result.quantile_level,
            metadata={
                **(metadata or {}),
                "calibration_space": "similarity",
                "nc_threshold": result.threshold,
            },
        )
    
    def _create_default_result(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConformalCalibrationResult:
        """Create default result when calibration is not possible."""
        return ConformalCalibrationResult(
            threshold=self.default_threshold,
            alpha=self.alpha,
            target_recall=self.target_recall,
            guarantee="No guarantee (using default threshold, not calibrated)",
            num_calibration_samples=0,
            calibration_scores=[],
            quantile_level=0.0,
            metadata={**(metadata or {}), "is_default": True},
        )


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(
    emb1: Union[List[float], np.ndarray],
    emb2: Union[List[float], np.ndarray],
) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Cosine similarity score in [-1, 1]
    """
    arr1 = np.asarray(emb1, dtype=np.float32)
    arr2 = np.asarray(emb2, dtype=np.float32)
    
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def nonconformity_score_negative_cosine(
    emb1: Union[List[float], np.ndarray],
    emb2: Union[List[float], np.ndarray],
) -> float:
    """
    Compute nonconformity score as negative cosine similarity.
    
    NC(e1, e2) = -cos(e1, e2)
    
    Lower NC score means higher similarity (more conforming).
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Nonconformity score (negative cosine similarity)
    """
    return -cosine_similarity(emb1, emb2)
