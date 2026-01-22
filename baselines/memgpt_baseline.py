"""
MemGPT-like Baseline
Uses recency + LLM importance scoring (similar to MemGPT approach).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.recency import RecencyExtractor
from features.utility import UtilityExtractor
from typing import List, Dict


class MemGPTBaseline:
    """
    MemGPT-like baseline using recency + LLM importance.

    MemGPT combines temporal recency with LLM-based importance scoring.
    This is a simplified version based on the MemGPT paper approach.
    """

    def __init__(
        self,
        recency_weight: float = 0.5,
        importance_weight: float = 0.5,
        threshold: float = 0.5,
        model_name: str = "qwen2.5:latest"
    ):
        """
        Initialize MemGPT baseline.

        Args:
            recency_weight: Weight for recency score (default: 0.5).
            importance_weight: Weight for importance score (default: 0.5).
            threshold: Admission threshold (default: 0.5).
            model_name: LLM model for importance scoring.
        """
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.threshold = threshold

        # Initialize feature extractors
        self.recency_extractor = RecencyExtractor(decay_rate=0.01)
        self.utility_extractor = UtilityExtractor(model_name=model_name)

    def score(
        self,
        memory,
        conversation_history: List[Dict] = None,
        current_time: float = None
    ) -> float:
        """
        Compute MemGPT-style score combining recency + importance.

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context for importance.
            current_time: Current timestamp for recency.

        Returns:
            Combined score in [0, 1].
        """
        # Recency score
        if current_time is not None:
            recency_score = self.recency_extractor.score(memory, current_time)
        else:
            recency_score = 1.0  # Default to max if no timestamp

        # Importance score (using LLM)
        if conversation_history is not None:
            importance_score = self.utility_extractor.score(memory, conversation_history)
        else:
            importance_score = 0.5  # Neutral if no context

        # Weighted combination
        combined_score = (
            self.recency_weight * recency_score +
            self.importance_weight * importance_score
        )

        return combined_score

    def should_admit(
        self,
        memory,
        conversation_history: List[Dict] = None,
        current_time: float = None
    ) -> bool:
        """
        Decide whether to admit a memory.

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context.
            current_time: Current timestamp.

        Returns:
            True if memory should be admitted.
        """
        score = self.score(memory, conversation_history, current_time)
        return score >= self.threshold


if __name__ == "__main__":
    # Test the MemGPT baseline
    import time
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str
        timestamp: float

    baseline = MemGPTBaseline(
        recency_weight=0.5,
        importance_weight=0.5,
        threshold=0.5
    )

    current = time.time()

    # Test memory
    memory = TestMemory(
        content="User prefers Python for data analysis",
        timestamp=current - 3600  # 1 hour ago
    )

    history = [
        {"turn_id": 1, "text": "What tools do you use?"},
        {"turn_id": 2, "text": "I mainly use Python for data analysis."},
    ]

    print("MemGPT Baseline Test:")
    print(f"  Memory: '{memory.content}'")
    print(f"  Recency weight: {baseline.recency_weight}")
    print(f"  Importance weight: {baseline.importance_weight}")
    print("\nScoring (this may take 10-30s with LLM)...")

    score = baseline.score(memory, history, current)
    decision = baseline.should_admit(memory, history, current)

    print(f"  Combined score: {score:.3f}")
    print(f"  Threshold: {baseline.threshold}")
    print(f"  Decision: {'ADMIT' if decision else 'REJECT'}")
