"""
Confidence Feature Extractor (C)
Measures factual reliability through evidence-based verification.
"""

from typing import List, Dict
from rouge_score import rouge_scorer


class ConfidenceExtractor:
    """
    Confidence (C): Factual Reliability

    Extracts supporting evidence spans from conversation history and
    computes alignment scores using ROUGE-L to verify factual correctness.

    High confidence = memory well-supported by conversation evidence.
    Low confidence = potential hallucination or unsupported inference.
    """

    def __init__(self, rouge_metric: str = 'rougeL'):
        """
        Initialize the confidence extractor.

        Args:
            rouge_metric: ROUGE metric to use (default: rougeL).
        """
        self.rouge_metric = rouge_metric
        self.scorer = rouge_scorer.RougeScorer([rouge_metric], use_stemmer=True)

    def score(self, memory, conversation_history: List[Dict]) -> float:
        """
        Compute confidence score for a candidate memory.

        Args:
            memory: Memory object with content.
            conversation_history: List of conversation turns.

        Returns:
            Confidence score in [0, 1].
        """
        memory_text = memory.content

        # Extract candidate supporting spans from conversation history
        supporting_spans = self._extract_supporting_spans(
            memory_text,
            conversation_history
        )

        if not supporting_spans:
            # No supporting evidence found - low confidence
            return 0.0

        # Compute ROUGE-L scores against all supporting spans
        alignment_scores = []
        for span in supporting_spans:
            score = self._compute_alignment(memory_text, span)
            alignment_scores.append(score)

        # Return maximum alignment score (best evidence match)
        return max(alignment_scores)

    def _extract_supporting_spans(
        self,
        memory_text: str,
        conversation_history: List[Dict]
    ) -> List[str]:
        """
        Extract relevant spans from conversation that could support the memory.

        Args:
            memory_text: Memory content to find support for.
            conversation_history: Full conversation history.

        Returns:
            List of text spans that might support the memory.
        """
        # TODO: Implement more sophisticated span extraction
        # For now, use simple approach: all conversation turns are candidates

        spans = []
        for turn in conversation_history:
            text = turn.get('text', '')
            if text and len(text) > 0:
                # Simple filtering: only include turns with overlapping words
                memory_words = set(memory_text.lower().split())
                turn_words = set(text.lower().split())

                overlap = memory_words.intersection(turn_words)
                if len(overlap) > 0:
                    spans.append(text)

        return spans

    def _compute_alignment(self, memory_text: str, evidence_span: str) -> float:
        """
        Compute ROUGE-L alignment score between memory and evidence.

        Args:
            memory_text: Candidate memory content.
            evidence_span: Supporting evidence from conversation.

        Returns:
            ROUGE-L F1 score in [0, 1].
        """
        try:
            # Use ROUGE scorer to compute alignment
            scores = self.scorer.score(evidence_span, memory_text)
            return scores[self.rouge_metric].fmeasure
        except Exception as e:
            # Fallback to simple word overlap if ROUGE fails
            print(f"Warning: ROUGE scoring failed ({str(e)}), using word overlap fallback")

            memory_words = set(memory_text.lower().split())
            evidence_words = set(evidence_span.lower().split())

            if not memory_words or not evidence_words:
                return 0.0

            overlap = memory_words.intersection(evidence_words)
            precision = len(overlap) / len(memory_words) if memory_words else 0
            recall = len(overlap) / len(evidence_words) if evidence_words else 0

            if precision + recall == 0:
                return 0.0

            f1 = 2 * (precision * recall) / (precision + recall)
            return f1


    def detect_hallucination(self, memory, conversation_history: List[Dict]) -> bool:
        """
        Detect if a memory is likely a hallucination.

        Args:
            memory: Memory object.
            conversation_history: Conversation history.

        Returns:
            True if likely hallucination (low confidence), False otherwise.
        """
        confidence = self.score(memory, conversation_history)

        # Threshold: confidence < 0.3 suggests hallucination
        hallucination_threshold = 0.3
        return confidence < hallucination_threshold


if __name__ == "__main__":
    # Test the confidence extractor
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str

    extractor = ConfidenceExtractor()

    # Test case 1: Well-supported memory
    memory1 = TestMemory(content="User prefers Python for readability")
    history1 = [
        {"turn_id": 1, "text": "What programming language do you prefer?"},
        {"turn_id": 2, "text": "I prefer Python because it's very readable."},
        {"turn_id": 3, "text": "That makes sense!"},
    ]

    score1 = extractor.score(memory1, history1)
    print(f"Test 1 - Well-supported memory: {score1:.3f}")
    print(f"  Hallucination? {extractor.detect_hallucination(memory1, history1)}")

    # Test case 2: Unsupported memory (potential hallucination)
    memory2 = TestMemory(content="User lives in Tokyo and has three cats")
    history2 = [
        {"turn_id": 1, "text": "What's the weather like today?"},
        {"turn_id": 2, "text": "It's sunny and warm."},
    ]

    score2 = extractor.score(memory2, history2)
    print(f"\nTest 2 - Unsupported memory: {score2:.3f}")
    print(f"  Hallucination? {extractor.detect_hallucination(memory2, history2)}")
