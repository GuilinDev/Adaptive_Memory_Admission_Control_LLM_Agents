"""
Novelty Feature Extractor (N)
Measures redundancy avoidance through semantic similarity.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class NoveltyExtractor:
    """
    Novelty (N): Redundancy Avoidance

    Computes semantic similarity between candidate memory and existing memories
    using sentence embeddings. High novelty = distinct new information.

    N(m) = 1 - max_similarity(m, existing_memories)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the novelty extractor.

        Args:
            model_name: Sentence-BERT model to use for embeddings.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}

    def score(self, memory, existing_memories: List) -> float:
        """
        Compute novelty score for a candidate memory.

        Args:
            memory: Candidate memory object.
            existing_memories: List of currently stored memories.

        Returns:
            Novelty score in [0, 1].
            1.0 = completely novel (no overlap with existing memories)
            0.0 = completely redundant (duplicate of existing memory)
        """
        if not existing_memories:
            # No existing memories = completely novel
            return 1.0

        # Get embedding for candidate memory
        candidate_embedding = self._get_embedding(memory.content)

        # Compute cosine similarities with all existing memories
        similarities = []
        for existing_mem in existing_memories:
            existing_embedding = self._get_embedding(existing_mem.content)
            similarity = self._cosine_similarity(candidate_embedding, existing_embedding)
            similarities.append(similarity)

        # Novelty = 1 - max_similarity
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity

        return max(0.0, min(1.0, novelty))  # Clamp to [0, 1]

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence embedding for text.

        Args:
            text: Input text.

        Returns:
            Embedding vector as numpy array.
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            # Use actual sentence transformer
            embedding = self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            print(f"Warning: Sentence embedding failed ({str(e)}), using fallback")
            # Fallback: Random embedding for testing
            np.random.seed(hash(text) % (2**32))  # Deterministic for same text
            embedding = np.random.randn(384)  # MiniLM has 384 dimensions
            embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Cache the embedding
        self.embedding_cache[text] = embedding

        return embedding

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Compute dot product
        similarity = np.dot(vec1_norm, vec2_norm)

        return float(similarity)

    def find_similar_memories(
        self,
        memory,
        existing_memories: List,
        threshold: float = 0.8
    ) -> List[tuple]:
        """
        Find existing memories similar to the candidate.

        Args:
            memory: Candidate memory.
            existing_memories: List of existing memories.
            threshold: Similarity threshold (default 0.8).

        Returns:
            List of (memory, similarity_score) tuples for memories above threshold.
        """
        if not existing_memories:
            return []

        candidate_embedding = self._get_embedding(memory.content)

        similar_memories = []
        for existing_mem in existing_memories:
            existing_embedding = self._get_embedding(existing_mem.content)
            similarity = self._cosine_similarity(candidate_embedding, existing_embedding)

            if similarity >= threshold:
                similar_memories.append((existing_mem, similarity))

        # Sort by similarity (descending)
        similar_memories.sort(key=lambda x: x[1], reverse=True)

        return similar_memories

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()


if __name__ == "__main__":
    # Test the novelty extractor
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str

    extractor = NoveltyExtractor()

    # Test case 1: Novel memory (no existing memories)
    memory1 = TestMemory(content="User prefers dark mode")
    existing1 = []

    score1 = extractor.score(memory1, existing1)
    print(f"Test 1 - Novel (no existing): {score1:.3f}")

    # Test case 2: Somewhat redundant memory
    memory2 = TestMemory(content="User likes dark theme for UI")
    existing2 = [
        TestMemory(content="User prefers dark mode"),
        TestMemory(content="User is located in California"),
    ]

    score2 = extractor.score(memory2, existing2)
    print(f"\nTest 2 - Somewhat redundant: {score2:.3f}")

    # Find similar memories
    similar = extractor.find_similar_memories(memory2, existing2, threshold=0.5)
    print(f"Similar memories found: {len(similar)}")
    for mem, sim in similar:
        print(f"  - '{mem.content}' (similarity: {sim:.3f})")

    # Test case 3: Completely different memory
    memory3 = TestMemory(content="Python is great for machine learning")
    existing3 = [
        TestMemory(content="User prefers dark mode"),
        TestMemory(content="User is located in California"),
    ]

    score3 = extractor.score(memory3, existing3)
    print(f"\nTest 3 - Novel (different topic): {score3:.3f}")
