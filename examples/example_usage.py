"""
Example Usage: Memory Admission Control for LLM Agents

This script demonstrates how to use the memory admission system
to decide which conversation turns should be stored in long-term memory.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorer import MemoryAdmissionScorer
from data_loader import MemoryCandidate, ConversationTurn
from features.utility import UtilityExtractor
from features.confidence import ConfidenceExtractor
from features.novelty import NoveltyExtractor
from features.recency import RecencyExtractor
from features.type_prior import TypePriorExtractor


def example_1_basic_scoring():
    """Example 1: Basic memory admission scoring"""
    print("=" * 60)
    print("Example 1: Basic Memory Admission Scoring")
    print("=" * 60)

    # Initialize scorer with optimized weights from paper
    # Weights: [Utility, Confidence, Novelty, Recency, Type Prior]
    scorer = MemoryAdmissionScorer(
        weights=[0.221, 0.115, 0.230, 0.229, 0.205],
        threshold=0.3
    )

    # Create a sample conversation history
    history = [
        ConversationTurn(
            speaker="User",
            text="Hi, I'm planning a trip to Japan next month.",
            timestamp="2026-01-01T10:00:00"
        ),
        ConversationTurn(
            speaker="Agent",
            text="That sounds exciting! What cities are you planning to visit?",
            timestamp="2026-01-01T10:00:30"
        ),
        ConversationTurn(
            speaker="User",
            text="I'm thinking Tokyo and Kyoto. I love Japanese food.",
            timestamp="2026-01-01T10:01:00"
        ),
    ]

    # Test different candidate memories
    test_cases = [
        {
            "text": "My birthday is March 15th.",
            "label": "SHOULD ADMIT (important personal fact)"
        },
        {
            "text": "The weather is nice today.",
            "label": "SHOULD REJECT (ephemeral information)"
        },
        {
            "text": "I'm allergic to peanuts.",
            "label": "SHOULD ADMIT (critical health information)"
        },
        {
            "text": "I prefer window seats on flights.",
            "label": "SHOULD ADMIT (useful preference)"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        candidate_turn = ConversationTurn(
            speaker="User",
            text=test["text"],
            timestamp="2026-01-01T10:05:00"
        )

        candidate = MemoryCandidate(
            turn=candidate_turn,
            conversation_history=history,
            existing_memories=[]
        )

        score = scorer.score(candidate)
        decision = scorer.should_admit(candidate)

        print(f"\nTest Case {i}:")
        print(f"  Text: \"{test['text']}\"")
        print(f"  Score: {score:.3f}")
        print(f"  Decision: {'✓ ADMIT' if decision else '✗ REJECT'}")
        print(f"  Expected: {test['label']}")


def example_2_feature_breakdown():
    """Example 2: Examine individual feature contributions"""
    print("\n" + "=" * 60)
    print("Example 2: Feature Contribution Analysis")
    print("=" * 60)

    # Initialize individual feature extractors
    extractors = {
        "Utility": UtilityExtractor(),
        "Confidence": ConfidenceExtractor(),
        "Novelty": NoveltyExtractor(),
        "Recency": RecencyExtractor(),
        "Type Prior": TypePriorExtractor()
    }

    # Create a sample candidate
    history = [
        ConversationTurn(
            speaker="User",
            text="I work as a software engineer at a tech company.",
            timestamp="2026-01-01T09:00:00"
        ),
    ]

    candidate_turn = ConversationTurn(
        speaker="User",
        text="I'm fluent in Python and Java programming languages.",
        timestamp="2026-01-01T10:00:00"
    )

    candidate = MemoryCandidate(
        turn=candidate_turn,
        conversation_history=history,
        existing_memories=[]
    )

    print(f"\nCandidate: \"{candidate_turn.text}\"")
    print("\nFeature Scores:")
    print("-" * 50)

    weights = [0.221, 0.115, 0.230, 0.229, 0.205]
    feature_names = ["Utility", "Confidence", "Novelty", "Recency", "Type Prior"]

    total_score = 0.0
    for i, (name, extractor) in enumerate(extractors.items()):
        try:
            feature_score = extractor.extract(candidate)
            weighted_score = weights[i] * feature_score
            total_score += weighted_score

            print(f"{name:12} | Raw: {feature_score:.3f} | "
                  f"Weight: {weights[i]:.3f} | "
                  f"Contribution: {weighted_score:.3f}")
        except Exception as e:
            print(f"{name:12} | Error: {str(e)}")

    print("-" * 50)
    print(f"{'Total Score':12} | {total_score:.3f}")
    print(f"{'Decision':12} | {'✓ ADMIT' if total_score > 0.3 else '✗ REJECT'}")


def example_3_temporal_decay():
    """Example 3: Demonstrate temporal decay (Recency feature)"""
    print("\n" + "=" * 60)
    print("Example 3: Temporal Decay Analysis")
    print("=" * 60)

    recency_extractor = RecencyExtractor(
        decay_rate=0.1,  # Decay rate per day
        max_age_days=90
    )

    base_time = datetime.now()

    print("\nRecency scores for the same memory at different ages:")
    print("-" * 50)

    test_ages = [0, 1, 7, 30, 60, 90]
    for days_ago in test_ages:
        timestamp = (base_time - timedelta(days=days_ago)).isoformat()

        candidate_turn = ConversationTurn(
            speaker="User",
            text="I love classical music.",
            timestamp=timestamp
        )

        candidate = MemoryCandidate(
            turn=candidate_turn,
            conversation_history=[],
            existing_memories=[]
        )

        score = recency_extractor.extract(candidate)
        print(f"Age: {days_ago:3d} days | Recency Score: {score:.3f}")


def example_4_novelty_detection():
    """Example 4: Demonstrate novelty detection"""
    print("\n" + "=" * 60)
    print("Example 4: Novelty Detection")
    print("=" * 60)

    novelty_extractor = NoveltyExtractor(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.85
    )

    # Existing memories about user preferences
    existing = [
        ConversationTurn(
            speaker="User",
            text="I love Italian cuisine, especially pasta.",
            timestamp="2026-01-01T09:00:00"
        ),
        ConversationTurn(
            speaker="User",
            text="I work as a data scientist.",
            timestamp="2026-01-01T09:10:00"
        ),
    ]

    # Test candidates with varying novelty
    test_cases = [
        ("I enjoy eating pasta and pizza.", "LOW novelty (redundant with memory 1)"),
        ("I'm allergic to shellfish.", "HIGH novelty (new health info)"),
        ("I'm a data scientist at a tech company.", "LOW novelty (similar to memory 2)"),
        ("My favorite color is blue.", "HIGH novelty (new preference)"),
    ]

    print("\nTesting novelty with existing memories:")
    print("Existing memories:")
    for i, mem in enumerate(existing, 1):
        print(f"  {i}. \"{mem.text}\"")

    print("\n" + "-" * 50)

    for text, expected in test_cases:
        candidate_turn = ConversationTurn(
            speaker="User",
            text=text,
            timestamp="2026-01-01T10:00:00"
        )

        candidate = MemoryCandidate(
            turn=candidate_turn,
            conversation_history=[],
            existing_memories=existing
        )

        novelty_score = novelty_extractor.extract(candidate)

        print(f"\nCandidate: \"{text}\"")
        print(f"Novelty Score: {novelty_score:.3f}")
        print(f"Expected: {expected}")


def example_5_content_type_classification():
    """Example 5: Content type classification and priors"""
    print("\n" + "=" * 60)
    print("Example 5: Content Type Classification")
    print("=" * 60)

    type_extractor = TypePriorExtractor()

    test_cases = [
        ("My birthday is March 15th.", "fact"),
        ("I prefer tea over coffee.", "preference"),
        ("I attended the conference last week.", "event"),
        ("What's the weather like today?", "question"),
        ("I think AI will change the world.", "opinion"),
    ]

    print("\nClassifying different content types:")
    print("-" * 50)

    for text, expected_type in test_cases:
        candidate_turn = ConversationTurn(
            speaker="User",
            text=text,
            timestamp="2026-01-01T10:00:00"
        )

        candidate = MemoryCandidate(
            turn=candidate_turn,
            conversation_history=[],
            existing_memories=[]
        )

        type_score = type_extractor.extract(candidate)
        detected_type = type_extractor.classify_type(text)

        print(f"\nText: \"{text}\"")
        print(f"Detected Type: {detected_type}")
        print(f"Expected Type: {expected_type}")
        print(f"Type Prior Score: {type_score:.3f}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  Memory Admission Control - Usage Examples              ║")
    print("╚" + "=" * 58 + "╝")

    try:
        example_1_basic_scoring()
        example_2_feature_breakdown()
        example_3_temporal_decay()
        example_4_novelty_detection()
        example_5_content_type_classification()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
