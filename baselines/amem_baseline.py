"""
A-mem Baseline
Uses LLM-generated structured attributes with cosine similarity matching.
Based on: "A-mem: Agentic Memory for LLM Agents"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.novelty import NoveltyExtractor
from typing import List, Dict, Optional
import subprocess
import json
import re
import numpy as np


class AMemBaseline:
    """
    A-mem baseline using LLM-generated structured attributes.

    A-mem generates:
    1. Context: When this memory is relevant
    2. Keywords: Key terms for retrieval
    3. Tags: Category labels
    4. Importance: 1-5 scale importance

    Admission is based on:
    - Importance threshold
    - Novelty compared to existing memories (via embedding similarity)
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:latest",
        importance_threshold: float = 2.5,
        novelty_threshold: float = 0.7,
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize A-mem baseline.

        Args:
            model_name: LLM for attribute generation.
            importance_threshold: Minimum importance (1-5 scale) for admission.
            novelty_threshold: Minimum novelty for admission.
            embedding_model: Model for computing similarity.
        """
        self.model_name = model_name
        self.importance_threshold = importance_threshold
        self.novelty_threshold = novelty_threshold
        self.novelty_extractor = NoveltyExtractor(model_name=embedding_model)

        # Cache for generated attributes
        self.attribute_cache = {}

    def generate_attributes(self, memory, conversation_history: List[Dict]) -> Dict:
        """
        Generate structured attributes for a memory using LLM.

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context.

        Returns:
            Dictionary with context, keywords, tags, importance.
        """
        # Check cache
        cache_key = memory.content
        if cache_key in self.attribute_cache:
            return self.attribute_cache[cache_key]

        prompt = self._construct_attribute_prompt(memory, conversation_history)

        try:
            response = self._call_ollama(prompt)
            attributes = self._parse_attributes(response)
        except Exception as e:
            print(f"Warning: Attribute generation failed ({str(e)}), using defaults")
            attributes = {
                'context': 'general conversation',
                'keywords': [memory.content.split()[0]] if memory.content else ['unknown'],
                'tags': ['general'],
                'importance': 3.0
            }

        self.attribute_cache[cache_key] = attributes
        return attributes

    def _construct_attribute_prompt(self, memory, conversation_history: List[Dict]) -> str:
        """Construct prompt for attribute generation."""
        context_window = 3
        recent_turns = conversation_history[-context_window:] if len(conversation_history) > context_window else conversation_history

        context_text = "\n".join([
            f"Turn {t.get('turn_id', i)}: {t.get('text', '')}"
            for i, t in enumerate(recent_turns)
        ])

        prompt = f"""Analyze this memory and generate structured attributes.

Conversation context:
{context_text}

Memory to analyze:
"{memory.content}"

Generate the following attributes in JSON format:
1. context: When this memory would be useful (1 sentence)
2. keywords: 3-5 key terms for retrieval
3. tags: 1-3 category labels (e.g., "preference", "fact", "plan", "emotion")
4. importance: Score from 1-5 (1=trivial, 5=critical)

Respond ONLY with valid JSON:
{{"context": "...", "keywords": ["...", "..."], "tags": ["..."], "importance": N}}"""

        return prompt

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama for LLM inference."""
        cmd = ["ollama", "run", self.model_name, prompt]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise RuntimeError(f"Ollama error: {result.stderr}")

    def _parse_attributes(self, response: str) -> Dict:
        """Parse LLM response to extract attributes."""
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)

        if json_match:
            try:
                attrs = json.loads(json_match.group())
                # Validate and normalize
                return {
                    'context': str(attrs.get('context', 'general')),
                    'keywords': list(attrs.get('keywords', []))[:5],
                    'tags': list(attrs.get('tags', ['general']))[:3],
                    'importance': float(attrs.get('importance', 3))
                }
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback parsing
        importance = 3.0
        numbers = re.findall(r'\b([1-5])\b', response)
        if numbers:
            importance = float(numbers[-1])

        return {
            'context': 'general',
            'keywords': [],
            'tags': ['general'],
            'importance': importance
        }

    def score(
        self,
        memory,
        conversation_history: List[Dict] = None,
        existing_memories: List = None
    ) -> float:
        """
        Compute A-mem style admission score.

        Combines importance and novelty.

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context.
            existing_memories: Existing memory store.

        Returns:
            Score in [0, 1].
        """
        # Generate attributes
        if conversation_history is None:
            conversation_history = []

        attributes = self.generate_attributes(memory, conversation_history)

        # Normalize importance from [1,5] to [0,1]
        importance_normalized = (attributes['importance'] - 1) / 4.0

        # Compute novelty
        if existing_memories:
            novelty = self.novelty_extractor.score(memory, existing_memories)
        else:
            novelty = 1.0

        # A-mem primarily uses importance, with novelty as secondary
        # Weight: 70% importance, 30% novelty
        score = 0.7 * importance_normalized + 0.3 * novelty

        return score

    def should_admit(
        self,
        memory,
        conversation_history: List[Dict] = None,
        existing_memories: List = None
    ) -> bool:
        """
        Decide whether to admit a memory using A-mem logic.

        A memory is admitted if:
        1. Importance >= threshold (normalized)
        2. Novelty >= novelty_threshold (not too similar to existing)

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context.
            existing_memories: Existing memories.

        Returns:
            True if memory should be admitted.
        """
        if conversation_history is None:
            conversation_history = []

        attributes = self.generate_attributes(memory, conversation_history)

        # Check importance threshold
        if attributes['importance'] < self.importance_threshold:
            return False

        # Check novelty threshold
        if existing_memories:
            novelty = self.novelty_extractor.score(memory, existing_memories)
            if novelty < self.novelty_threshold:
                return False

        return True

    def get_admission_details(
        self,
        memory,
        conversation_history: List[Dict] = None,
        existing_memories: List = None
    ) -> Dict:
        """
        Get detailed admission decision with attributes.

        Returns:
            Dictionary with attributes, scores, and decision.
        """
        if conversation_history is None:
            conversation_history = []

        attributes = self.generate_attributes(memory, conversation_history)

        novelty = 1.0
        if existing_memories:
            novelty = self.novelty_extractor.score(memory, existing_memories)

        score = self.score(memory, conversation_history, existing_memories)
        decision = self.should_admit(memory, conversation_history, existing_memories)

        return {
            'attributes': attributes,
            'novelty': novelty,
            'score': score,
            'decision': decision
        }


if __name__ == "__main__":
    # Test the A-mem baseline
    import time
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str
        timestamp: float

    baseline = AMemBaseline(
        model_name="qwen2.5:latest",
        importance_threshold=2.5,
        novelty_threshold=0.7
    )

    current = time.time()

    memory = TestMemory(
        content="User is a vegetarian and prefers plant-based meals",
        timestamp=current
    )

    history = [
        {"turn_id": 1, "text": "Can you recommend a restaurant?"},
        {"turn_id": 2, "text": "Sure! What kind of food do you prefer?"},
        {"turn_id": 3, "text": "I'm vegetarian, so plant-based options would be great."},
    ]

    print("A-mem Baseline Test:")
    print(f"  Memory: '{memory.content}'")
    print("\nGenerating attributes (this may take 30-60s)...")

    details = baseline.get_admission_details(memory, history, [])

    print(f"\nGenerated Attributes:")
    print(f"  Context: {details['attributes']['context']}")
    print(f"  Keywords: {details['attributes']['keywords']}")
    print(f"  Tags: {details['attributes']['tags']}")
    print(f"  Importance: {details['attributes']['importance']}")
    print(f"\nNovelty: {details['novelty']:.3f}")
    print(f"Combined Score: {details['score']:.3f}")
    print(f"Decision: {'ADMIT' if details['decision'] else 'REJECT'}")
