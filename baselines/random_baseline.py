"""
Random Baseline
Randomly admits memories with a fixed probability.
"""

import random
from typing import List


class RandomBaseline:
    """
    Random baseline that admits memories with probability p.

    This serves as the simplest baseline to compare against.
    """

    def __init__(self, admission_probability: float = 0.5, seed: int = 42):
        """
        Initialize random baseline.

        Args:
            admission_probability: Probability of admitting a memory (default: 0.5).
            seed: Random seed for reproducibility.
        """
        self.admission_probability = admission_probability
        self.seed = seed
        random.seed(seed)

    def should_admit(self, memory, **kwargs) -> bool:
        """
        Decide whether to admit a memory randomly.

        Args:
            memory: Candidate memory (ignored).
            **kwargs: Additional context (ignored).

        Returns:
            True if memory should be admitted (random).
        """
        return random.random() < self.admission_probability

    def score(self, memory, **kwargs) -> float:
        """
        Return random score for compatibility.

        Args:
            memory: Candidate memory (ignored).
            **kwargs: Additional context (ignored).

        Returns:
            Random score in [0, 1].
        """
        return random.random()


if __name__ == "__main__":
    # Test the random baseline
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str

    baseline = RandomBaseline(admission_probability=0.5, seed=42)

    # Test multiple memories
    test_memories = [
        TestMemory(content=f"Memory {i}")
        for i in range(10)
    ]

    admitted = sum(1 for mem in test_memories if baseline.should_admit(mem))

    print(f"Random Baseline Test:")
    print(f"  Admission probability: {baseline.admission_probability}")
    print(f"  Memories tested: {len(test_memories)}")
    print(f"  Admitted: {admitted}")
    print(f"  Admission rate: {admitted/len(test_memories):.2f}")
