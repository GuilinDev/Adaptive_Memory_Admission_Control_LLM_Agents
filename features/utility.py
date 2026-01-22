"""
Utility Feature Extractor (U)
Estimates future task value using LLM-based scoring.
"""

from typing import List, Dict
import time
import subprocess
import json
import re


class UtilityExtractor:
    """
    Utility (U): Future Task Value

    Uses LLM prompting with temperature=0 to estimate the likelihood
    that a memory will be useful for future tasks or queries.
    """

    def __init__(self, model_name: str = "qwen2.5:latest", temperature: float = 0.0):
        """
        Initialize the utility extractor.

        Args:
            model_name: Ollama model to use for scoring (default: qwen2.5:latest).
            temperature: Sampling temperature (0 for deterministic).
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_ollama = True  # Using Ollama for local LLM inference

    def score(self, memory, conversation_history: List[Dict]) -> float:
        """
        Compute utility score for a candidate memory.

        Args:
            memory: Memory object with content, timestamp, etc.
            conversation_history: List of conversation turns.

        Returns:
            Utility score in [0, 1].
        """
        prompt = self._construct_prompt(memory, conversation_history)

        try:
            if self.use_ollama:
                response_text = self._call_ollama(prompt)
            else:
                # Fallback to placeholder if Ollama is not available
                response_text = "0.5"

            score = self._parse_response(response_text)
            return score

        except Exception as e:
            print(f"Warning: LLM scoring failed ({str(e)}), using fallback score 0.5")
            return 0.5

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama CLI to get LLM response.

        Args:
            prompt: The prompt to send to the model.

        Returns:
            The model's response text.
        """
        try:
            # Prepare the generate request
            cmd = [
                "ollama",
                "run",
                self.model_name,
                prompt
            ]

            # Call ollama
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise RuntimeError(f"Ollama returned error: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama request timed out")
        except FileNotFoundError:
            raise RuntimeError("Ollama not found. Please install ollama.")

    def _construct_prompt(self, memory, conversation_history: List[Dict]) -> str:
        """
        Construct the LLM prompt for utility scoring.

        Args:
            memory: Candidate memory.
            conversation_history: Conversation context.

        Returns:
            Prompt string.
        """
        # Build context from recent conversation turns
        context_window = 5  # Last 5 turns
        recent_turns = conversation_history[-context_window:] if len(conversation_history) > context_window else conversation_history

        context_text = "\n".join([
            f"Turn {t.get('turn_id', i)}: {t.get('text', '')}"
            for i, t in enumerate(recent_turns)
        ])

        prompt = f"""Given the following conversation context:

{context_text}

Candidate memory to evaluate:
"{memory.content}"

Rate the likelihood that this memory will be useful for continuing the conversation or completing future tasks. Provide a score between 0.0 (completely useless) and 1.0 (extremely useful).

Your response should be a single floating-point number between 0.0 and 1.0.

Score:"""

        return prompt

    def _parse_response(self, response_text: str) -> float:
        """
        Parse LLM response to extract utility score.

        Args:
            response_text: Raw LLM response.

        Returns:
            Parsed score in [0, 1].
        """
        try:
            # Try to extract a float from the response
            score = float(response_text.strip())
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            # If parsing fails, try to extract numbers
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response_text)
            if numbers:
                return float(numbers[0])
            # Fallback to neutral score
            return 0.5


if __name__ == "__main__":
    # Test the utility extractor
    from dataclasses import dataclass

    @dataclass
    class TestMemory:
        content: str
        turn_id: int
        timestamp: float

    extractor = UtilityExtractor()

    test_memory = TestMemory(
        content="User prefers Python over JavaScript",
        turn_id=5,
        timestamp=time.time()
    )

    test_history = [
        {"turn_id": 1, "text": "What programming languages do you know?"},
        {"turn_id": 2, "text": "I know Python, JavaScript, and Go."},
        {"turn_id": 3, "text": "Which one do you prefer?"},
        {"turn_id": 4, "text": "I prefer Python for its readability."},
        {"turn_id": 5, "text": "Great! I'll keep that in mind."},
    ]

    score = extractor.score(test_memory, test_history)
    print(f"Utility score: {score:.3f}")
