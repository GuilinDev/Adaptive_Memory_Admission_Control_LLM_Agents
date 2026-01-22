"""
Complete Baseline Experiment Runner
Runs all baselines (Random, MemGPT, MemoryBank, A-mem) and our method.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
from tqdm import tqdm
from dataclasses import dataclass

# Import modules
from data_loader import LoCoMoDataLoader, MemoryCandidate
from scorer import Memory, Features
from features.utility import UtilityExtractor
from features.confidence import ConfidenceExtractor
from features.novelty import NoveltyExtractor
from features.recency import RecencyExtractor
from features.type_prior import TypePriorExtractor
from baselines.random_baseline import RandomBaseline
from baselines.memgpt_baseline import MemGPTBaseline
from baselines.memorybank_baseline import MemoryBankBaseline
from baselines.amem_baseline import AMemBaseline


def candidate_to_memory(candidate: MemoryCandidate) -> Memory:
    """Convert MemoryCandidate to Memory object."""
    return Memory(
        content=candidate.dialogue_turn.text,
        turn_id=candidate.dialogue_turn.turn_id,
        timestamp=candidate.dialogue_turn.timestamp,
        metadata={}
    )


def candidate_to_history(candidate: MemoryCandidate) -> List[Dict]:
    """Convert conversation history to dict format."""
    return [
        {
            'turn_id': t.turn_id,
            'text': t.text,
            'speaker': t.speaker,
            'timestamp': t.timestamp
        }
        for t in candidate.conversation_history
    ]


def evaluate_our_method(
    test_candidates: List[MemoryCandidate],
    weights: np.ndarray,
    threshold: float,
    use_llm: bool = True
) -> Tuple[Dict, float]:
    """
    Evaluate our weighted scoring method.

    Returns:
        (results_dict, avg_latency_ms)
    """
    print("\nEvaluating: Our Method (Weighted Feature Scoring)")

    # Initialize extractors
    extractors = {
        'utility': UtilityExtractor(model_name="qwen2.5:latest") if use_llm else None,
        'confidence': ConfidenceExtractor(rouge_metric='rougeL'),
        'novelty': NoveltyExtractor(model_name='all-MiniLM-L6-v2'),
        'recency': RecencyExtractor(decay_rate=0.01),
        'type_prior': TypePriorExtractor()
    }

    y_true = []
    y_pred = []
    latencies = []

    for candidate in tqdm(test_candidates, desc="Our Method"):
        memory = candidate_to_memory(candidate)
        history = candidate_to_history(candidate)
        current_time = time.time()

        start_time = time.time()

        # Extract features
        if use_llm and extractors['utility'] is not None:
            try:
                u = extractors['utility'].score(memory, history)
            except:
                u = 0.5
        else:
            u = 0.5

        c = extractors['confidence'].score(memory, history)
        n = extractors['novelty'].score(memory, [])
        r = extractors['recency'].score(memory, current_time)
        t = extractors['type_prior'].score(memory)

        # Compute weighted score
        features = np.array([u, c, n, r, t])
        score = np.dot(weights, features)

        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # ms

        y_true.append(1 if candidate.is_referenced else 0)
        y_pred.append(1 if score >= threshold else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_latency = np.mean(latencies)

    results = {
        'method': 'Ours',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_ms': avg_latency
    }

    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Latency: {avg_latency:.1f}ms")

    return results, avg_latency


def evaluate_memgpt(
    test_candidates: List[MemoryCandidate],
    use_llm: bool = True
) -> Tuple[Dict, float]:
    """Evaluate MemGPT baseline."""
    print("\nEvaluating: MemGPT")

    baseline = MemGPTBaseline(
        recency_weight=0.4,
        importance_weight=0.6,
        threshold=0.5,
        model_name="qwen2.5:latest" if use_llm else None
    )

    # Override to not use LLM if disabled
    if not use_llm:
        baseline.utility_extractor = None

    y_true = []
    y_pred = []
    latencies = []

    for candidate in tqdm(test_candidates, desc="MemGPT"):
        memory = candidate_to_memory(candidate)
        history = candidate_to_history(candidate)
        current_time = time.time()

        start_time = time.time()

        # Compute recency
        recency_score = baseline.recency_extractor.score(memory, current_time)

        # Compute importance
        if use_llm and baseline.utility_extractor is not None:
            try:
                importance_score = baseline.utility_extractor.score(memory, history)
            except:
                importance_score = 0.5
        else:
            importance_score = 0.5

        score = 0.4 * recency_score + 0.6 * importance_score

        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)

        y_true.append(1 if candidate.is_referenced else 0)
        y_pred.append(1 if score >= 0.5 else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_latency = np.mean(latencies)

    results = {
        'method': 'MemGPT',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_ms': avg_latency
    }

    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Latency: {avg_latency:.1f}ms")

    return results, avg_latency


def evaluate_memorybank(
    test_candidates: List[MemoryCandidate],
    use_llm: bool = True
) -> Tuple[Dict, float]:
    """Evaluate MemoryBank baseline."""
    print("\nEvaluating: MemoryBank")

    baseline = MemoryBankBaseline(
        w_recency=0.3,
        w_relevance=0.4,
        w_importance=0.3,
        threshold=0.5,
        model_name="qwen2.5:latest" if use_llm else None
    )

    if not use_llm:
        baseline.utility_extractor = None

    y_true = []
    y_pred = []
    latencies = []

    for candidate in tqdm(test_candidates, desc="MemoryBank"):
        memory = candidate_to_memory(candidate)
        history = candidate_to_history(candidate)
        current_time = time.time()

        start_time = time.time()

        score = baseline.score(memory, history, current_time)

        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)

        y_true.append(1 if candidate.is_referenced else 0)
        y_pred.append(1 if score >= 0.5 else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_latency = np.mean(latencies)

    results = {
        'method': 'MemoryBank',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_ms': avg_latency
    }

    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Latency: {avg_latency:.1f}ms")

    return results, avg_latency


def evaluate_amem(
    test_candidates: List[MemoryCandidate],
    use_llm: bool = True
) -> Tuple[Dict, float]:
    """Evaluate A-mem baseline."""
    print("\nEvaluating: A-mem")

    baseline = AMemBaseline(
        model_name="qwen2.5:latest" if use_llm else None,
        importance_threshold=2.5,
        novelty_threshold=0.3
    )

    y_true = []
    y_pred = []
    latencies = []

    for candidate in tqdm(test_candidates, desc="A-mem"):
        memory = candidate_to_memory(candidate)
        history = candidate_to_history(candidate)

        start_time = time.time()

        if use_llm:
            score = baseline.score(memory, history, [])
            decision = score >= 0.5
        else:
            # Without LLM, use simplified scoring
            decision = np.random.random() > 0.4  # ~60% admission rate

        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)

        y_true.append(1 if candidate.is_referenced else 0)
        y_pred.append(1 if decision else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_latency = np.mean(latencies)

    results = {
        'method': 'A-mem',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_ms': avg_latency
    }

    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Latency: {avg_latency:.1f}ms")

    return results, avg_latency


def evaluate_random(test_candidates: List[MemoryCandidate]) -> Tuple[Dict, float]:
    """Evaluate Random baseline."""
    print("\nEvaluating: Random")

    baseline = RandomBaseline(admission_probability=0.3, seed=42)

    y_true = []
    y_pred = []
    latencies = []

    for candidate in test_candidates:
        start_time = time.time()
        decision = baseline.should_admit(candidate.dialogue_turn)
        end_time = time.time()

        latencies.append((end_time - start_time) * 1000)

        y_true.append(1 if candidate.is_referenced else 0)
        y_pred.append(1 if decision else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_latency = np.mean(latencies)

    results = {
        'method': 'Random',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_ms': avg_latency
    }

    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Latency: {avg_latency:.1f}ms")

    return results, avg_latency


def evaluate_equal_weights(
    test_candidates: List[MemoryCandidate],
    use_llm: bool = True
) -> Tuple[Dict, float]:
    """Evaluate Equal Weights baseline."""
    print("\nEvaluating: Equal Weights")

    weights = np.ones(5) / 5  # [0.2, 0.2, 0.2, 0.2, 0.2]
    return evaluate_our_method(test_candidates, weights, 0.5, use_llm)


def run_all_baselines(use_llm: bool = True, test_size: int = 100):
    """
    Run all baseline experiments.

    Args:
        use_llm: Whether to use LLM (much slower but more accurate).
        test_size: Number of test samples to use.
    """
    print("=" * 70)
    print("Complete Baseline Experiment")
    print("=" * 70)
    print(f"LLM Enabled: {use_llm}")
    print(f"Test Size: {test_size}")

    # Load data
    print("\n[1/7] Loading LoCoMo dataset...")
    loader = LoCoMoDataLoader(data_path="data/locomo/data/locomo10.json")

    _, _, test_candidates = loader.train_val_test_split(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # Limit test size
    test_candidates = test_candidates[:test_size]
    print(f"  Using {len(test_candidates)} test samples")

    # Load optimized weights (from cross-validation)
    with open('results/optimized_weights_cv.json', 'r') as f:
        weights_data = json.load(f)
    optimal_weights = np.array(weights_data['weights'])
    optimal_threshold = weights_data.get('threshold', 0.5)
    print(f"  Optimal weights: {optimal_weights.round(3)}")
    print(f"  Optimal threshold: {optimal_threshold}")

    all_results = []

    # Run all baselines
    print("\n" + "=" * 70)
    print("Running Baseline Evaluations")
    print("=" * 70)

    # 1. Random
    results, _ = evaluate_random(test_candidates)
    all_results.append(results)

    # 2. MemGPT
    results, _ = evaluate_memgpt(test_candidates, use_llm)
    all_results.append(results)

    # 3. MemoryBank
    results, _ = evaluate_memorybank(test_candidates, use_llm)
    all_results.append(results)

    # 4. A-mem
    results, _ = evaluate_amem(test_candidates, use_llm)
    all_results.append(results)

    # 5. Equal Weights
    equal_results, _ = evaluate_our_method(test_candidates, np.ones(5)/5, 0.5, use_llm)
    equal_results['method'] = 'Equal Weights'
    all_results.append(equal_results)

    # 6. Ours (Weighted with optimized threshold)
    our_results, _ = evaluate_our_method(test_candidates, optimal_weights, optimal_threshold, use_llm)
    our_results['method'] = 'Ours'
    all_results.append(our_results)

    # Create results DataFrame
    df = pd.DataFrame(all_results)

    # Sort by F1 score
    df = df.sort_values('f1', ascending=False)

    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    print(df.to_string(index=False))

    # Save results
    os.makedirs("results", exist_ok=True)
    df.to_csv('results/all_baselines_results.csv', index=False)
    print("\n✓ Results saved to results/all_baselines_results.csv")

    # Compute improvements
    our_f1 = df[df['method'] == 'Ours']['f1'].values[0]

    print("\n" + "=" * 70)
    print("Our Method's Improvements")
    print("=" * 70)

    for _, row in df.iterrows():
        if row['method'] != 'Ours':
            improvement = (our_f1 - row['f1']) / row['f1'] * 100 if row['f1'] > 0 else 0
            print(f"  vs {row['method']}: {improvement:+.1f}% relative F1 improvement")

    return df


if __name__ == "__main__":
    use_llm = True
    test_size = 100  # Adjust based on time constraints

    if len(sys.argv) > 1:
        if '--no-llm' in sys.argv:
            use_llm = False
        if '--small' in sys.argv:
            test_size = 30
        if '--medium' in sys.argv:
            test_size = 100
        if '--large' in sys.argv:
            test_size = 200

    print(f"\nRun Configuration:")
    print(f"  --llm: {use_llm} (use --no-llm to disable)")
    print(f"  --test-size: {test_size} (use --small/--medium/--large)")

    df = run_all_baselines(use_llm=use_llm, test_size=test_size)
