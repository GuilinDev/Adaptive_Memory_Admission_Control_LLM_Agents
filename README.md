# Adaptive Memory Admission Control for LLM Agents

This repository contains the reference implementation for the paper:

> **Adaptive Memory Admission Control for LLM Agents Using Weighted Feature Scoring**
> Anonymous Authors
> Under review at ICLR 2026

## Overview

Long-term memory management is critical for LLM-based conversational agents. Our system decides which conversational turns should be admitted to persistent memory using a weighted combination of five interpretable features:

- **Utility (U)**: Future usefulness assessed via LLM
- **Confidence (C)**: Factual reliability measured by information consistency
- **Novelty (N)**: Information uniqueness compared to existing memories
- **Recency (R)**: Temporal freshness with exponential decay
- **Type Prior (T)**: Content type importance (preferences, facts, states, etc.)

The admission score for a candidate memory `m` is computed as:

```
S(m) = w_U·U(m) + w_C·C(m) + w_N·N(m) + w_R·R(m) + w_T·T(m)
```

where weights `w` are optimized via 5-fold cross-validated grid search to maximize F1 score.

## Key Results

| Method | Precision | Recall | F1 | Latency (ms) |
|--------|-----------|--------|-----|--------------|
| **Ours** | **0.417** | 0.972 | **0.583** | **2644** |
| A-mem | 0.371 | 1.000 | 0.541 | 3831 |
| Equal Weights | 0.362 | 0.694 | 0.476 | 2916 |
| MemoryBank | 0.368 | 0.583 | 0.452 | 2843 |
| MemGPT | 0.316 | 0.333 | 0.324 | 2765 |
| Random | 0.278 | 0.278 | 0.278 | <1 |

Key findings:
- **7.8% F1 improvement** over A-mem (0.583 vs 0.541)
- **97.2% recall** with only 2.8% reduction compared to A-mem
- **31% faster** than A-mem (2644ms vs 3831ms)
- **Type Prior** is the dominant feature (weight 0.60), suggesting content category is the strongest signal for admission decisions

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- Sentence-BERT
- ROUGE
- scikit-learn

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd adaptive-memory-admission

# Install dependencies
pip install -r requirements.txt

# Download Sentence-BERT model (used for Novelty feature)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Quick Start

### Basic Usage

```python
from scorer import MemoryAdmissionScorer
from data_loader import MemoryCandidate, ConversationTurn

# Initialize the scorer with learned weights (optimized via cross-validation)
scorer = MemoryAdmissionScorer(
    weights=[0.1, 0.1, 0.1, 0.1, 0.6],  # [U, C, N, R, T]
    threshold=0.55
)

# Create a candidate memory from a conversation turn
turn = ConversationTurn(
    speaker="User",
    text="My birthday is on March 15th.",
    timestamp="2026-01-01T10:00:00"
)
candidate = MemoryCandidate(
    turn=turn,
    conversation_history=[...],  # Previous turns for context
    existing_memories=[...]       # Already stored memories
)

# Score and decide admission
score = scorer.score(candidate)
should_admit = scorer.should_admit(candidate)

print(f"Admission score: {score:.3f}")
print(f"Decision: {'ADMIT' if should_admit else 'REJECT'}")
```

### Learning Custom Weights

```python
from weight_optimizer import WeightOptimizerCV
from data_loader import load_locomo_dataset

# Load training data (LoCoMo or your own labeled dataset)
train_data = load_locomo_dataset(split="train")

# Initialize optimizer with cross-validation
optimizer = WeightOptimizerCV(
    n_folds=5,
    random_state=42
)

# Optimize weights to maximize F1 score
best_weights, best_threshold = optimizer.optimize(train_data)

print(f"Optimized weights: {best_weights}")
print(f"Optimized threshold: {best_threshold}")
```

## Feature Descriptions

### 1. Utility (U) - `features/utility.py`

Measures future usefulness via LLM prompting:

```python
from features.utility import UtilityExtractor

extractor = UtilityExtractor(model_name="qwen2.5:latest")
utility_score = extractor.score(memory, conversation_history)
```

### 2. Confidence (C) - `features/confidence.py`

Assesses factual reliability by measuring consistency between the candidate statement and surrounding context using ROUGE-L:

```python
from features.confidence import ConfidenceExtractor

extractor = ConfidenceExtractor(rouge_metric="rougeL")
confidence_score = extractor.score(memory, conversation_history)
```

High confidence indicates the information is well-supported by context, reducing hallucination risk.

### 3. Novelty (N) - `features/novelty.py`

Quantifies information uniqueness using semantic similarity (Sentence-BERT) between the candidate and existing memories:

```python
from features.novelty import NoveltyExtractor

extractor = NoveltyExtractor(model_name="all-MiniLM-L6-v2")
novelty_score = extractor.score(memory, existing_memories)
```

Higher novelty means less redundancy with stored memories.

### 4. Recency (R) - `features/recency.py`

Applies exponential temporal decay to prioritize recent information:

```python
from features.recency import RecencyExtractor

extractor = RecencyExtractor(decay_rate=0.01)
recency_score = extractor.score(memory, current_time)
```

### 5. Type Prior (T) - `features/type_prior.py`

Assigns importance scores to different content types using rule-based classification:

```python
from features.type_prior import TypePriorExtractor

extractor = TypePriorExtractor()
type_score = extractor.score(memory)
# Returns 0.9 for preferences, 0.7 for facts, 0.5 for plans, 0.2 for temporary states
```

## Running Experiments

### Full Baseline Comparison

```bash
cd experiments
python run_all_baselines.py --medium  # 100 test samples with LLM
python run_all_baselines.py --no-llm --small  # 30 samples without LLM (faster)
```

### Weight Optimization with Cross-Validation

```bash
python optimize_weights_cv.py
```

Results are saved to `results/optimized_weights_cv.json`.

## Project Structure

```
code_release/
├── features/               # Feature extractors
│   ├── __init__.py
│   ├── utility.py         # U: Future usefulness (LLM-based)
│   ├── confidence.py      # C: Factual reliability (ROUGE-L)
│   ├── novelty.py         # N: Information uniqueness (SBERT)
│   ├── recency.py         # R: Temporal freshness (decay)
│   └── type_prior.py      # T: Content type importance (rules)
├── baselines/             # Baseline methods
│   ├── random_baseline.py
│   ├── memgpt_baseline.py
│   ├── memorybank_baseline.py
│   └── amem_baseline.py
├── scorer.py              # Main admission scorer
├── weight_optimizer.py    # Weight learning via cross-validation
├── data_loader.py         # LoCoMo dataset utilities
├── run_all_baselines.py   # Full experiment runner
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── LICENSE                # MIT License
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the creators of the LoCoMo benchmark for providing evaluation data.
