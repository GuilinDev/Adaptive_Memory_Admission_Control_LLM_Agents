"""
MemoryBank Baseline
Uses recency + relevance + importance scoring (based on MemoryBank paper).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.recency import RecencyExtractor
from features.utility import UtilityExtractor
from features.novelty import NoveltyExtractor
from typing import List, Dict
import numpy as np


class MemoryBankBaseline:
    """
    MemoryBank baseline using recency + relevance + importance.

    Based on: "MemoryBank: Enhancing Large Language Models with Long-Term Memory"

    Score = w_rec * recency + w_rel * relevance + w_imp * importance
    Default weights from paper: [0.3, 0.4, 0.3]
    """

    def __init__(
        self,
        w_recency: float = 0.3,
        w_relevance: float = 0.4,
        w_importance: float = 0.3,
        threshold: float = 0.5,
        model_name: str = "qwen2.5:latest"
    ):
        """
        Initialize MemoryBank baseline.

        Args:
            w_recency: Weight for recency score.
            w_relevance: Weight for relevance score.
            w_importance: Weight for importance score.
            threshold: Admission threshold.
            model_name: LLM model for importance scoring.
        """
        self.w_recency = w_recency
        self.w_relevance = w_relevance
        self.w_importance = w_importance
        self.threshold = threshold

        # Initialize extractors
        self.recency_extractor = RecencyExtractor(decay_rate=0.01)
        self.novelty_extractor = NoveltyExtractor(model_name='all-MiniLM-L6-v2')
        self.utility_extractor = UtilityExtractor(model_name=model_name)

    def score(
        self,
        memory,
        conversation_history: List[Dict] = None,
        current_time: float = None,
        existing_memories: List = None
    ) -> float:
        """
        Compute MemoryBank-style score.

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context.
            current_time: Current timestamp.
            existing_memories: Existing memory store for relevance.

        Returns:
            Combined score in [0, 1].
        """
        # Recency score
        if current_time is not None:
            recency_score = self.recency_extractor.score(memory, current_time)
        else:
            recency_score = 1.0

        # Relevance score (similarity to recent conversation context)
        # We approximate this by computing similarity to recent turns
        if conversation_history and len(conversation_history) > 0:
            relevance_score = self._compute_relevance(memory, conversation_history)
        else:
            relevance_score = 0.5

        # Importance score (using LLM)
        if conversation_history is not None:
            try:
                importance_score = self.utility_extractor.score(memory, conversation_history)
            except Exception:
                importance_score = 0.5
        else:
            importance_score = 0.5

        # Weighted combination
        combined_score = (
            self.w_recency * recency_score +
            self.w_relevance * relevance_score +
            self.w_importance * importance_score
        )

        return combined_score

    def _compute_relevance(self, memory, conversation_history: List[Dict]) -> float:
        """
        Compute relevance score based on similarity to recent conversation.

        Args:
            memory: Candidate memory.
            conversation_history: Recent conversation turns.

        Returns:
            Relevance score in [0, 1].
        """
        # Combine recent turns into context
        recent_turns = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        context_text = " ".join([t.get('text', '') for t in recent_turns])

        if not context_text.strip():
            return 0.5

        # Get embeddings
        memory_emb = self.novelty_extractor._get_embedding(memory.content)
        context_emb = self.novelty_extractor._get_embedding(context_text)

        # Compute similarity
        similarity = self.novelty_extractor._cosine_similarity(memory_emb, context_emb)

        # Map from [-1, 1] to [0, 1]
        relevance = (similarity + 1) / 2

        return relevance

    def should_admit(
        self,
        memory,
        conversation_history: List[Dict] = None,
        current_time: float = None,
        existing_memories: List = None
    ) -> bool:
        """
        Decide whether to admit a memory.

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context.
            current_time: Current timestamp.
            existing_memories: Existing memories.

        Returns:
            True if memory should be admitted.
        """
        score = self.score(memory, conversation_history, current_time, existing_memories)
        return score >= self.threshold


if __name__ == "__main__":
    # Test the MemoryBank baseline
    import time
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str
        timestamp: float

    baseline = MemoryBankBaseline(
        w_recency=0.3,
        w_relevance=0.4,
        w_importance=0.3,
        threshold=0.5
    )

    current = time.time()

    memory = TestMemory(
        content="User prefers Python for data analysis",
        timestamp=current - 3600
    )

    history = [
        {"turn_id": 1, "text": "What tools do you use for data analysis?"},
        {"turn_id": 2, "text": "I mainly use Python with pandas and numpy."},
    ]

    print("MemoryBank Baseline Test:")
    print(f"  Memory: '{memory.content}'")
    print(f"  Weights: rec={baseline.w_recency}, rel={baseline.w_relevance}, imp={baseline.w_importance}")
    print("\nScoring...")

    score = baseline.score(memory, history, current)
    decision = baseline.should_admit(memory, history, current)

    print(f"  Combined score: {score:.3f}")
    print(f"  Threshold: {baseline.threshold}")
    print(f"  Decision: {'ADMIT' if decision else 'REJECT'}")
