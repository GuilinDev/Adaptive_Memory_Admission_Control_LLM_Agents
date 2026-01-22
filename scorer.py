"""
Memory Admission Scorer - Core Implementation
Implements the weighted linear scoring model for LLM agent memory admission.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Memory:
    """Represents a candidate memory for admission."""
    content: str
    turn_id: int
    timestamp: float
    metadata: Dict = None


@dataclass
class Features:
    """Container for the five feature scores."""
    utility: float  # U: Future task value
    confidence: float  # C: Factual reliability
    novelty: float  # N: Redundancy avoidance
    recency: float  # R: Temporal relevance
    type_prior: float  # T: Content category importance

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for computation."""
        return np.array([
            self.utility,
            self.confidence,
            self.novelty,
            self.recency,
            self.type_prior
        ])


class MemoryAdmissionScorer:
    """
    Weighted linear scoring model for memory admission control.

    S(m) = w_U * U(m) + w_C * C(m) + w_N * N(m) + w_R * R(m) + w_T * T(m)

    Admits memory if S(m) >= threshold.
    """

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        feature_extractors: Optional[Dict] = None
    ):
        """
        Initialize the scorer.

        Args:
            weights: Weight vector [w_U, w_C, w_N, w_R, w_T].
                    If None, uses equal weights [0.2, 0.2, 0.2, 0.2, 0.2].
            threshold: Admission threshold (default 0.5).
            feature_extractors: Dictionary of feature extractor instances.
        """
        if weights is None:
            self.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        else:
            assert len(weights) == 5, "Must provide 5 weights for U, C, N, R, T"
            assert np.abs(np.sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            assert np.all(weights >= 0), "Weights must be non-negative"
            self.weights = np.array(weights)

        self.threshold = threshold
        self.feature_extractors = feature_extractors or {}

    def compute_features(
        self,
        memory: Memory,
        conversation_history: List[Dict],
        existing_memories: List[Memory]
    ) -> Features:
        """
        Compute all five features for a candidate memory.

        Args:
            memory: Candidate memory to score.
            conversation_history: Full conversation history.
            existing_memories: Currently stored memories.

        Returns:
            Features object with U, C, N, R, T scores.
        """
        # TODO: Implement feature extraction using feature_extractors
        # This is a placeholder - actual implementation will call specific extractors

        utility = self._compute_utility(memory, conversation_history)
        confidence = self._compute_confidence(memory, conversation_history)
        novelty = self._compute_novelty(memory, existing_memories)
        recency = self._compute_recency(memory)
        type_prior = self._compute_type_prior(memory)

        return Features(
            utility=utility,
            confidence=confidence,
            novelty=novelty,
            recency=recency,
            type_prior=type_prior
        )

    def _compute_utility(self, memory: Memory, history: List[Dict]) -> float:
        """
        Utility feature: Future task value (LLM-based).

        TODO: Implement LLM-based utility scoring with temperature=0.
        """
        # Placeholder - will call LLM
        if 'utility' in self.feature_extractors:
            return self.feature_extractors['utility'].score(memory, history)
        return 0.5  # Default placeholder

    def _compute_confidence(self, memory: Memory, history: List[Dict]) -> float:
        """
        Confidence feature: Factual reliability (rule-based, ROUGE-L).

        TODO: Implement evidence span extraction and ROUGE-L scoring.
        """
        if 'confidence' in self.feature_extractors:
            return self.feature_extractors['confidence'].score(memory, history)
        return 0.5  # Default placeholder

    def _compute_novelty(self, memory: Memory, existing: List[Memory]) -> float:
        """
        Novelty feature: Redundancy avoidance (cosine similarity).

        TODO: Implement sentence embedding and cosine similarity computation.
        """
        if 'novelty' in self.feature_extractors:
            return self.feature_extractors['novelty'].score(memory, existing)
        return 0.5  # Default placeholder

    def _compute_recency(self, memory: Memory) -> float:
        """
        Recency feature: Temporal relevance (exponential decay).

        TODO: Implement exponential decay based on memory age.
        """
        if 'recency' in self.feature_extractors:
            return self.feature_extractors['recency'].score(memory)
        return 0.5  # Default placeholder

    def _compute_type_prior(self, memory: Memory) -> float:
        """
        Type Prior feature: Content category importance (rule-based classification).

        TODO: Implement type classification and prior lookup.
        """
        if 'type_prior' in self.feature_extractors:
            return self.feature_extractors['type_prior'].score(memory)
        return 0.5  # Default placeholder

    def score(self, features: Features) -> float:
        """
        Compute weighted linear score from features.

        Args:
            features: Features object with U, C, N, R, T scores.

        Returns:
            Admission score in [0, 1].
        """
        feature_array = features.to_array()
        score = np.dot(self.weights, feature_array)
        return float(score)

    def should_admit(self, score: float) -> bool:
        """
        Determine whether to admit a memory based on its score.

        Args:
            score: Admission score.

        Returns:
            True if score >= threshold, False otherwise.
        """
        return score >= self.threshold

    def admit(
        self,
        memory: Memory,
        conversation_history: List[Dict],
        existing_memories: List[Memory]
    ) -> Tuple[bool, float, Features]:
        """
        Full admission pipeline: compute features, score, and decide.

        Args:
            memory: Candidate memory.
            conversation_history: Full conversation history.
            existing_memories: Currently stored memories.

        Returns:
            Tuple of (should_admit, score, features).
        """
        features = self.compute_features(memory, conversation_history, existing_memories)
        score = self.score(features)
        decision = self.should_admit(score)

        return decision, score, features

    def set_weights(self, weights: np.ndarray):
        """Update scorer weights."""
        assert len(weights) == 5
        assert np.abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights >= 0)
        self.weights = np.array(weights)

    def set_threshold(self, threshold: float):
        """Update admission threshold."""
        self.threshold = threshold


if __name__ == "__main__":
    # Simple test
    scorer = MemoryAdmissionScorer()

    test_memory = Memory(
        content="User prefers dark mode",
        turn_id=5,
        timestamp=1234567890.0,
        metadata={"type": "preference"}
    )

    history = [{"turn": 1, "text": "..."}, {"turn": 2, "text": "..."}]
    existing = []

    decision, score, features = scorer.admit(test_memory, history, existing)

    print(f"Decision: {decision}")
    print(f"Score: {score:.3f}")
    print(f"Features: {features}")
