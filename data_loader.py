"""
LoCoMo Dataset Loader for Memory Admission Experiments

This module loads the LoCoMo benchmark data and prepares it for memory admission experiments.
It extracts dialogue turns as candidate memories and derives ground truth labels from QA pairs.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from datetime import datetime, timedelta


@dataclass
class DialogueTurn:
    """Represents a single dialogue turn."""
    speaker: str
    dia_id: str  # e.g., "D1:3"
    text: str
    session_id: int
    turn_id: int
    timestamp: float  # Unix timestamp


@dataclass
class QAPair:
    """Represents a question-answer pair with evidence."""
    question: str
    answer: str
    evidence: List[str]  # List of dia_ids like ["D1:3", "D2:8"]
    category: int  # 1=fact, 2=temporal, 3=inference


@dataclass
class Conversation:
    """Represents a complete conversation with multiple sessions."""
    sample_id: str
    speaker_a: str
    speaker_b: str
    dialogue_turns: List[DialogueTurn]
    qa_pairs: List[QAPair]
    turn_index: Dict[str, DialogueTurn]  # dia_id -> turn mapping


@dataclass
class MemoryCandidate:
    """A candidate memory for admission decision."""
    dialogue_turn: DialogueTurn
    conversation_id: str
    is_referenced: bool  # Whether this turn is referenced in any QA evidence
    reference_count: int  # How many QA pairs reference this turn
    conversation_history: List[DialogueTurn]  # All turns before this one


class LoCoMoDataLoader:
    """Loads and processes the LoCoMo benchmark dataset."""

    def __init__(self, data_path: str = "data/locomo/data/locomo10.json"):
        """
        Initialize the data loader.

        Args:
            data_path: Path to locomo10.json file
        """
        self.data_path = data_path
        self.conversations: List[Conversation] = []
        self._load_data()

    def _load_data(self):
        """Load the LoCoMo JSON data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"LoCoMo data not found at {self.data_path}")

        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)

        for conv_data in raw_data:
            self.conversations.append(self._parse_conversation(conv_data))

    def _parse_conversation(self, conv_data: Dict) -> Conversation:
        """Parse a single conversation from raw JSON."""
        speaker_a = conv_data['conversation']['speaker_a']
        speaker_b = conv_data['conversation']['speaker_b']
        sample_id = conv_data['sample_id']

        # Extract all dialogue turns from all sessions
        dialogue_turns = []
        session_dates = {}

        # Find all sessions
        session_keys = [k for k in conv_data['conversation'].keys() if k.startswith('session_')]
        session_nums = set()
        for key in session_keys:
            if key.endswith('_date_time'):
                session_num = int(key.split('_')[1])
                session_nums.add(session_num)
                session_dates[session_num] = conv_data['conversation'][key]

        # Parse each session
        for session_num in sorted(session_nums):
            session_key = f'session_{session_num}'
            if session_key not in conv_data['conversation']:
                continue

            session_turns = conv_data['conversation'][session_key]

            # Parse session date/time to generate timestamps
            date_str = session_dates.get(session_num, "2023-05-01 10:00:00")
            try:
                session_start = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except:
                # Fallback if date format is different
                session_start = datetime(2023, 5, 1, 10, 0, 0) + timedelta(days=session_num*7)

            for turn_idx, turn_data in enumerate(session_turns):
                # Simulate timestamps: each turn is ~2 minutes apart
                turn_timestamp = (session_start + timedelta(minutes=turn_idx*2)).timestamp()

                turn = DialogueTurn(
                    speaker=turn_data['speaker'],
                    dia_id=turn_data['dia_id'],
                    text=turn_data['text'],
                    session_id=session_num,
                    turn_id=turn_idx,
                    timestamp=turn_timestamp
                )
                dialogue_turns.append(turn)

        # Build dia_id index
        turn_index = {turn.dia_id: turn for turn in dialogue_turns}

        # Parse QA pairs
        qa_pairs = []
        for qa_data in conv_data['qa']:
            # Some QA pairs have 'adversarial_answer' instead of 'answer'
            if 'answer' in qa_data:
                answer = qa_data['answer'] if isinstance(qa_data['answer'], str) else str(qa_data['answer'])
            elif 'adversarial_answer' in qa_data:
                answer = qa_data['adversarial_answer']
            else:
                answer = ""  # Fallback for missing answer

            qa = QAPair(
                question=qa_data['question'],
                answer=answer,
                evidence=qa_data['evidence'],
                category=qa_data['category']
            )
            qa_pairs.append(qa)

        return Conversation(
            sample_id=sample_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            dialogue_turns=dialogue_turns,
            qa_pairs=qa_pairs,
            turn_index=turn_index
        )

    def get_memory_candidates(self, conversation: Conversation) -> List[MemoryCandidate]:
        """
        Extract memory candidates from a conversation.

        Each dialogue turn is a candidate memory. We derive ground truth labels
        from QA evidence: turns referenced in evidence should be admitted.

        Args:
            conversation: A parsed conversation

        Returns:
            List of memory candidates with ground truth labels
        """
        # Count references for each dialogue turn
        reference_counts = {}
        referenced_turns = set()

        for qa in conversation.qa_pairs:
            for dia_id in qa.evidence:
                referenced_turns.add(dia_id)
                reference_counts[dia_id] = reference_counts.get(dia_id, 0) + 1

        # Create memory candidates
        candidates = []
        for i, turn in enumerate(conversation.dialogue_turns):
            # Conversation history = all turns before this one
            conversation_history = conversation.dialogue_turns[:i]

            candidate = MemoryCandidate(
                dialogue_turn=turn,
                conversation_id=conversation.sample_id,
                is_referenced=turn.dia_id in referenced_turns,
                reference_count=reference_counts.get(turn.dia_id, 0),
                conversation_history=conversation_history
            )
            candidates.append(candidate)

        return candidates

    def get_all_memory_candidates(self) -> List[MemoryCandidate]:
        """Get all memory candidates from all conversations."""
        all_candidates = []
        for conversation in self.conversations:
            candidates = self.get_memory_candidates(conversation)
            all_candidates.extend(candidates)
        return all_candidates

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[MemoryCandidate], List[MemoryCandidate], List[MemoryCandidate]]:
        """
        Split data into train/val/test sets.

        We split at the conversation level to avoid data leakage.

        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility

        Returns:
            (train_candidates, val_candidates, test_candidates)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        random.seed(seed)
        conversations = self.conversations.copy()
        random.shuffle(conversations)

        n = len(conversations)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_convs = conversations[:train_end]
        val_convs = conversations[train_end:val_end]
        test_convs = conversations[val_end:]

        train_candidates = []
        for conv in train_convs:
            train_candidates.extend(self.get_memory_candidates(conv))

        val_candidates = []
        for conv in val_convs:
            val_candidates.extend(self.get_memory_candidates(conv))

        test_candidates = []
        for conv in test_convs:
            test_candidates.extend(self.get_memory_candidates(conv))

        return train_candidates, val_candidates, test_candidates

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        all_candidates = self.get_all_memory_candidates()
        referenced = sum(1 for c in all_candidates if c.is_referenced)

        total_qa = sum(len(conv.qa_pairs) for conv in self.conversations)
        total_turns = sum(len(conv.dialogue_turns) for conv in self.conversations)

        stats = {
            'num_conversations': len(self.conversations),
            'total_dialogue_turns': total_turns,
            'total_qa_pairs': total_qa,
            'total_memory_candidates': len(all_candidates),
            'referenced_turns': referenced,
            'unreferenced_turns': len(all_candidates) - referenced,
            'reference_ratio': referenced / len(all_candidates) if all_candidates else 0,
            'avg_turns_per_conversation': total_turns / len(self.conversations),
            'avg_qa_per_conversation': total_qa / len(self.conversations)
        }

        return stats


if __name__ == "__main__":
    # Test the data loader
    print("Loading LoCoMo dataset...")
    loader = LoCoMoDataLoader()

    print("\nDataset Statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nSample conversation:")
    conv = loader.conversations[0]
    print(f"  ID: {conv.sample_id}")
    print(f"  Speakers: {conv.speaker_a} & {conv.speaker_b}")
    print(f"  Dialogue turns: {len(conv.dialogue_turns)}")
    print(f"  QA pairs: {len(conv.qa_pairs)}")

    print("\nFirst 3 dialogue turns:")
    for turn in conv.dialogue_turns[:3]:
        print(f"  [{turn.dia_id}] {turn.speaker}: {turn.text}")

    print("\nFirst QA pair:")
    qa = conv.qa_pairs[0]
    print(f"  Q: {qa.question}")
    print(f"  A: {qa.answer}")
    print(f"  Evidence: {qa.evidence}")

    print("\nTrain/Val/Test split:")
    train, val, test = loader.train_val_test_split()
    print(f"  Train: {len(train)} candidates")
    print(f"  Val: {len(val)} candidates")
    print(f"  Test: {len(test)} candidates")

    print(f"\n  Train referenced: {sum(1 for c in train if c.is_referenced)}/{len(train)}")
    print(f"  Val referenced: {sum(1 for c in val if c.is_referenced)}/{len(val)}")
    print(f"  Test referenced: {sum(1 for c in test if c.is_referenced)}/{len(test)}")
