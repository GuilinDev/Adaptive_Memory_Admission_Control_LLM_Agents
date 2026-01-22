"""
Simple Threshold Baseline
Uses a single feature (recency) with threshold.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.recency import RecencyExtractor


class ThresholdBaseline:
    """
    Simple threshold baseline using only recency.

    Admits memories if recency score exceeds threshold.
    """

    def __init__(self, threshold: float = 0.5, decay_rate: float = 0.01):
        """
        Initialize threshold baseline.

        Args:
            threshold: Admission threshold (default: 0.5).
            decay_rate: Recency decay rate per hour.
        """
        self.threshold = threshold
        self.recency_extractor = RecencyExtractor(decay_rate=decay_rate)

    def score(self, memory, current_time: float = None) -> float:
        """
        Compute recency score only.

        Args:
            memory: Candidate memory.
            current_time: Current timestamp.

        Returns:
            Recency score in [0, 1].
        """
        if current_time is not None:
            return self.recency_extractor.score(memory, current_time)
        else:
            return 1.0  # Default to max if no timestamp

    def should_admit(self, memory, current_time: float = None) -> bool:
        """
        Decide whether to admit based on recency threshold.

        Args:
            memory: Candidate memory.
            current_time: Current timestamp.

        Returns:
            True if recency score >= threshold.
        """
        score = self.score(memory, current_time)
        return score >= self.threshold


if __name__ == "__main__":
    # Test the threshold baseline
    import time
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str
        timestamp: float

    baseline = ThresholdBaseline(threshold=0.8)

    current = time.time()

    test_cases = [
        ("Very recent (1 min ago)", current - 60),
        ("1 hour ago", current - 3600),
        ("1 day ago", current - 86400),
        ("1 week ago", current - 604800),
    ]

    print("Threshold Baseline Test:")
    print(f"  Threshold: {baseline.threshold}")
    print(f"  Decay rate: {baseline.recency_extractor.decay_rate}/hour\n")

    for desc, timestamp in test_cases:
        memory = TestMemory(content=f"Memory: {desc}", timestamp=timestamp)
        score = baseline.score(memory, current)
        decision = baseline.should_admit(memory, current)

        print(f"  {desc}:")
        print(f"    Recency: {score:.3f}")
        print(f"    Decision: {'ADMIT' if decision else 'REJECT'}")
