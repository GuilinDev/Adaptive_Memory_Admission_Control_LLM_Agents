"""
Type Prior Feature Extractor (T)
Assigns importance based on memory content category.
"""

from typing import Dict, Set
import re


class TypePriorExtractor:
    """
    Type Prior (T): Content Category Importance

    Classifies memories into categories and assigns priors based on typical
    persistence requirements.

    Categories and priors:
    - Preference/Identity: 0.9 (long-term storage)
    - Fact: 0.7 (moderate persistence)
    - Plan/Goal: 0.5 (medium-term storage)
    - Temporary: 0.2 (short-term storage)
    """

    def __init__(self, custom_priors: Dict[str, float] = None):
        """
        Initialize the type prior extractor.

        Args:
            custom_priors: Optional custom prior scores by type.
        """
        # Default prior scores
        self.type_priors = {
            'preference': 0.9,
            'identity': 0.9,
            'belief': 0.9,
            'value': 0.9,

            'fact': 0.7,
            'knowledge': 0.7,
            'information': 0.7,

            'plan': 0.5,
            'goal': 0.5,
            'intention': 0.5,
            'task': 0.5,

            'temporary': 0.2,
            'ephemeral': 0.2,
            'transient': 0.2,
            'state': 0.3,

            'unknown': 0.5,  # Default for unclassified
        }

        # Override with custom priors if provided
        if custom_priors:
            self.type_priors.update(custom_priors)

        # Keywords for rule-based classification
        self.type_keywords = {
            'preference': {
                'prefer', 'like', 'dislike', 'hate', 'love', 'favorite',
                'enjoy', 'appreciate', 'avoid', 'want', 'wish'
            },
            'identity': {
                'i am', 'i\'m', 'my name is', 'called', 'live in', 'from',
                'work as', 'job', 'profession', 'nationality', 'age'
            },
            'fact': {
                'is', 'are', 'was', 'were', 'always', 'never', 'true',
                'false', 'known', 'established', 'defined'
            },
            'plan': {
                'will', 'going to', 'plan to', 'intend to', 'scheduled',
                'planning', 'upcoming', 'future', 'tomorrow', 'next'
            },
            'goal': {
                'want to', 'hope to', 'aim to', 'goal', 'objective',
                'target', 'aspire', 'strive'
            },
            'temporary': {
                'currently', 'right now', 'at the moment', 'today',
                'this morning', 'this afternoon', 'temporarily'
            },
        }

    def score(self, memory) -> float:
        """
        Compute type prior score for a memory.

        Args:
            memory: Memory object with content.

        Returns:
            Type prior score in [0, 1].
        """
        # Classify the memory type
        memory_type = self.classify(memory)

        # Return the prior for this type
        return self.type_priors.get(memory_type, self.type_priors['unknown'])

    def classify(self, memory) -> str:
        """
        Classify memory into a type category.

        Args:
            memory: Memory object with content.

        Returns:
            Type classification string.
        """
        content = memory.content.lower()

        # Check for explicit type in metadata
        if hasattr(memory, 'metadata') and memory.metadata:
            if 'type' in memory.metadata:
                return memory.metadata['type']

        if hasattr(memory, 'type') and memory.type:
            return memory.type

        # Rule-based classification using keywords
        type_scores = {}

        for mem_type, keywords in self.type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in content:
                    score += 1
            if score > 0:
                type_scores[mem_type] = score

        if type_scores:
            # Return type with highest keyword match count
            best_type = max(type_scores.items(), key=lambda x: x[1])[0]
            return best_type

        # Default to unknown if no classification possible
        return 'unknown'

    def add_type(self, type_name: str, prior: float, keywords: Set[str]):
        """
        Add a new type category.

        Args:
            type_name: Name of the type.
            prior: Prior score for this type.
            keywords: Set of keywords for rule-based matching.
        """
        self.type_priors[type_name] = prior
        self.type_keywords[type_name] = keywords

    def get_all_types(self) -> Dict[str, float]:
        """
        Get all defined types and their priors.

        Returns:
            Dictionary mapping type names to prior scores.
        """
        return self.type_priors.copy()


class AdvancedTypePriorExtractor(TypePriorExtractor):
    """
    Advanced type prior extractor using pattern matching and NER.

    TODO: Could integrate with spaCy NER or other NLP libraries
    for more sophisticated type classification.
    """

    def __init__(self):
        super().__init__()

        # Regex patterns for advanced classification
        self.patterns = {
            'identity': [
                r'\b(my name is|i am|i\'m)\s+\w+',
                r'\b(live in|from|based in)\s+[A-Z]\w+',
            ],
            'preference': [
                r'\b(prefer|like|love|favorite)\s+\w+',
                r'\b(don\'t like|dislike|hate)\s+\w+',
            ],
            'plan': [
                r'\b(will|going to|planning to)\s+\w+',
                r'\b(scheduled for|appointment)\s+',
            ],
        }

    def classify(self, memory) -> str:
        """
        Classify using both keywords and regex patterns.

        Args:
            memory: Memory object.

        Returns:
            Type classification.
        """
        content = memory.content

        # Try pattern matching first (more precise)
        for mem_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return mem_type

        # Fall back to keyword-based classification
        return super().classify(memory)


if __name__ == "__main__":
    # Test the type prior extractor
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str
        metadata: dict = None
        type: str = None

    extractor = TypePriorExtractor()

    # Test case 1: Preference
    memory1 = TestMemory(content="I prefer Python over JavaScript")
    type1 = extractor.classify(memory1)
    score1 = extractor.score(memory1)
    print(f"Test 1: '{memory1.content}'")
    print(f"  Type: {type1}, Score: {score1:.2f}\n")

    # Test case 2: Identity
    memory2 = TestMemory(content="My name is Alice and I live in Boston")
    type2 = extractor.classify(memory2)
    score2 = extractor.score(memory2)
    print(f"Test 2: '{memory2.content}'")
    print(f"  Type: {type2}, Score: {score2:.2f}\n")

    # Test case 3: Plan
    memory3 = TestMemory(content="I will go to the meeting tomorrow")
    type3 = extractor.classify(memory3)
    score3 = extractor.score(memory3)
    print(f"Test 3: '{memory3.content}'")
    print(f"  Type: {type3}, Score: {score3:.2f}\n")

    # Test case 4: Temporary
    memory4 = TestMemory(content="I am currently in a meeting")
    type4 = extractor.classify(memory4)
    score4 = extractor.score(memory4)
    print(f"Test 4: '{memory4.content}'")
    print(f"  Type: {type4}, Score: {score4:.2f}\n")

    # Test case 5: Explicit type in metadata
    memory5 = TestMemory(
        content="Some content",
        metadata={'type': 'fact'}
    )
    type5 = extractor.classify(memory5)
    score5 = extractor.score(memory5)
    print(f"Test 5: Explicit type in metadata")
    print(f"  Type: {type5}, Score: {score5:.2f}\n")

    # Test advanced extractor
    print("--- Advanced Extractor ---")
    advanced = AdvancedTypePriorExtractor()

    memory6 = TestMemory(content="I'm planning to visit Paris next month")
    type6 = advanced.classify(memory6)
    score6 = advanced.score(memory6)
    print(f"Test 6: '{memory6.content}'")
    print(f"  Type: {type6}, Score: {score6:.2f}")
