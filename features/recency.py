"""
Recency Feature Extractor (R)
Measures temporal relevance through exponential decay.
"""

import time
import math


class RecencyExtractor:
    """
    Recency (R): Temporal Relevance

    Models information value decay over time using exponential decay.

    R(m) = exp(-lambda * age(m))

    where age(m) is time elapsed since the information was mentioned.
    """

    def __init__(self, decay_rate: float = 0.01):
        """
        Initialize the recency extractor.

        Args:
            decay_rate: Lambda parameter for exponential decay (default: 0.01 per hour).
                       Higher values = faster decay.
        """
        self.decay_rate = decay_rate

    def score(self, memory, current_time: float = None) -> float:
        """
        Compute recency score for a memory.

        Args:
            memory: Memory object with timestamp attribute.
            current_time: Current timestamp (default: time.time()).

        Returns:
            Recency score in [0, 1].
            1.0 = very recent
            approaches 0.0 as memory ages
        """
        if current_time is None:
            current_time = time.time()

        # Calculate age in hours
        age_seconds = current_time - memory.timestamp
        age_hours = age_seconds / 3600.0

        # Exponential decay
        recency = math.exp(-self.decay_rate * age_hours)

        return max(0.0, min(1.0, recency))  # Clamp to [0, 1]

    def get_half_life_hours(self) -> float:
        """
        Get the half-life (time for recency to drop to 0.5) in hours.

        Returns:
            Half-life in hours.
        """
        # exp(-lambda * t_half) = 0.5
        # -lambda * t_half = ln(0.5)
        # t_half = -ln(0.5) / lambda = ln(2) / lambda
        return math.log(2) / self.decay_rate

    def set_half_life(self, hours: float):
        """
        Set decay rate based on desired half-life.

        Args:
            hours: Desired half-life in hours.
        """
        self.decay_rate = math.log(2) / hours

    def get_ttl(self, threshold: float = 0.1) -> float:
        """
        Get time-to-live (when recency drops below threshold) in hours.

        Args:
            threshold: Recency threshold (default: 0.1).

        Returns:
            TTL in hours.
        """
        # exp(-lambda * t_ttl) = threshold
        # -lambda * t_ttl = ln(threshold)
        # t_ttl = -ln(threshold) / lambda
        return -math.log(threshold) / self.decay_rate


class AdaptiveRecencyExtractor(RecencyExtractor):
    """
    Adaptive recency extractor that adjusts decay based on memory type.

    Different information types have different persistence requirements:
    - Preferences: slow decay (long TTL)
    - Facts: moderate decay
    - Plans: faster decay
    - Temporary states: rapid decay
    """

    def __init__(self):
        """Initialize with default decay rate."""
        super().__init__(decay_rate=0.01)

        # Type-specific decay rates (per hour)
        self.type_decay_rates = {
            'preference': 0.001,   # Very slow decay (~29 days half-life)
            'identity': 0.001,     # Very slow decay
            'fact': 0.01,          # Moderate decay (~2.9 days half-life)
            'plan': 0.05,          # Faster decay (~14 hours half-life)
            'goal': 0.05,
            'temporary': 0.2,      # Rapid decay (~3.5 hours half-life)
            'ephemeral': 0.5,      # Very rapid decay (~1.4 hours half-life)
        }

    def score(self, memory, current_time: float = None) -> float:
        """
        Compute type-adaptive recency score.

        Args:
            memory: Memory object with timestamp and optional type.
            current_time: Current timestamp.

        Returns:
            Recency score in [0, 1].
        """
        # Get memory type if available
        memory_type = None
        if hasattr(memory, 'metadata') and memory.metadata:
            memory_type = memory.metadata.get('type')
        elif hasattr(memory, 'type'):
            memory_type = memory.type

        # Use type-specific decay rate if available
        if memory_type and memory_type in self.type_decay_rates:
            original_rate = self.decay_rate
            self.decay_rate = self.type_decay_rates[memory_type]
            score = super().score(memory, current_time)
            self.decay_rate = original_rate  # Restore default
            return score
        else:
            # Fall back to default decay rate
            return super().score(memory, current_time)


if __name__ == "__main__":
    # Test the recency extractor
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str
        timestamp: float
        metadata: dict = None
        type: str = None

    extractor = RecencyExtractor(decay_rate=0.01)

    current_time = time.time()

    # Test case 1: Very recent memory (1 minute ago)
    memory1 = TestMemory(
        content="User just asked about Python",
        timestamp=current_time - 60
    )
    score1 = extractor.score(memory1, current_time)
    print(f"Test 1 - Very recent (1 min ago): {score1:.3f}")

    # Test case 2: Recent memory (1 hour ago)
    memory2 = TestMemory(
        content="User mentioned preferring Python",
        timestamp=current_time - 3600
    )
    score2 = extractor.score(memory2, current_time)
    print(f"Test 2 - Recent (1 hour ago): {score2:.3f}")

    # Test case 3: Old memory (1 week ago)
    memory3 = TestMemory(
        content="User asked about JavaScript",
        timestamp=current_time - (7 * 24 * 3600)
    )
    score3 = extractor.score(memory3, current_time)
    print(f"Test 3 - Old (1 week ago): {score3:.3f}")

    print(f"\nHalf-life: {extractor.get_half_life_hours():.1f} hours")
    print(f"TTL (threshold=0.1): {extractor.get_ttl(0.1):.1f} hours")

    # Test adaptive extractor
    print("\n--- Adaptive Recency Extractor ---")
    adaptive = AdaptiveRecencyExtractor()

    memory_pref = TestMemory(
        content="User prefers dark mode",
        timestamp=current_time - (7 * 24 * 3600),  # 1 week ago
        metadata={'type': 'preference'}
    )
    score_pref = adaptive.score(memory_pref, current_time)
    print(f"Preference (1 week ago): {score_pref:.3f}")

    memory_temp = TestMemory(
        content="User is currently in a meeting",
        timestamp=current_time - (7 * 24 * 3600),  # 1 week ago
        metadata={'type': 'temporary'}
    )
    score_temp = adaptive.score(memory_temp, current_time)
    print(f"Temporary (1 week ago): {score_temp:.3f}")
