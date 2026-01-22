"""
Improved Weight Optimizer with Cross-Validation and Threshold Optimization.
Fixes overfitting issue by using k-fold cross-validation.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import json
import os
import time
from tqdm import tqdm

# Import modules
from data_loader import LoCoMoDataLoader, MemoryCandidate
from scorer import Memory
from features.utility import UtilityExtractor
from features.confidence import ConfidenceExtractor
from features.novelty import NoveltyExtractor
from features.recency import RecencyExtractor
from features.type_prior import TypePriorExtractor


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


def extract_features_for_candidates(
    candidates: List[MemoryCandidate],
    extractors: Dict,
    use_llm: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-extract all features for candidates.

    Returns:
        (features_matrix, labels)
    """
    features_list = []
    labels = []

    for candidate in tqdm(candidates, desc="Extracting features"):
        memory = candidate_to_memory(candidate)
        history = candidate_to_history(candidate)
        current_time = time.time()

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

        features_list.append([u, c, n, r, t])
        labels.append(1 if candidate.is_referenced else 0)

    return np.array(features_list), np.array(labels)


def evaluate_weights_threshold(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    threshold: float
) -> Dict:
    """
    Evaluate given weights and threshold.

    Returns:
        Dictionary with precision, recall, f1
    """
    # Compute scores
    scores = features @ weights

    # Predict
    y_pred = (scores >= threshold).astype(int)

    # Metrics
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'admitted_ratio': y_pred.mean()
    }


def optimize_weights_and_threshold_cv(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    n_random_starts: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, float, Dict]:
    """
    Optimize weights and threshold using cross-validation.

    Uses random search + local optimization to find best weights.

    Args:
        features: (N, 5) feature matrix
        labels: (N,) binary labels
        n_folds: Number of CV folds
        n_random_starts: Number of random weight initializations
        seed: Random seed

    Returns:
        (best_weights, best_threshold, cv_results)
    """
    np.random.seed(seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    best_cv_f1 = 0.0
    best_weights = np.ones(5) / 5  # Start with equal weights
    best_threshold = 0.5

    # Try different weight combinations
    weight_candidates = []

    # 1. Equal weights
    weight_candidates.append(np.ones(5) / 5)

    # 2. Random weights
    for _ in range(n_random_starts):
        w = np.random.uniform(0.05, 1.0, size=5)
        w = w / w.sum()
        weight_candidates.append(w)

    # 3. Feature-emphasized weights (give more weight to one feature)
    for i in range(5):
        w = np.ones(5) * 0.1
        w[i] = 0.6
        w = w / w.sum()
        weight_candidates.append(w)

    # Try different thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    print(f"Optimizing with {n_folds}-fold CV...")
    print(f"Testing {len(weight_candidates)} weight combinations x {len(thresholds)} thresholds")

    results_log = []

    for weights in tqdm(weight_candidates, desc="Weight search"):
        for threshold in thresholds:
            # Cross-validation
            fold_f1s = []

            for train_idx, val_idx in kf.split(features):
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                # Evaluate on validation fold
                result = evaluate_weights_threshold(X_val, y_val, weights, threshold)
                fold_f1s.append(result['f1'])

            mean_f1 = np.mean(fold_f1s)
            std_f1 = np.std(fold_f1s)

            results_log.append({
                'weights': weights.tolist(),
                'threshold': threshold,
                'cv_f1_mean': mean_f1,
                'cv_f1_std': std_f1
            })

            if mean_f1 > best_cv_f1:
                best_cv_f1 = mean_f1
                best_weights = weights.copy()
                best_threshold = threshold

    # Final evaluation on full data
    final_result = evaluate_weights_threshold(features, labels, best_weights, best_threshold)

    cv_results = {
        'best_cv_f1': best_cv_f1,
        'best_weights': best_weights.tolist(),
        'best_threshold': best_threshold,
        'final_f1': final_result['f1'],
        'final_precision': final_result['precision'],
        'final_recall': final_result['recall'],
        'n_folds': n_folds,
        'n_candidates_tested': len(weight_candidates) * len(thresholds)
    }

    return best_weights, best_threshold, cv_results


def main():
    print("=" * 70)
    print("Improved Weight Optimization with Cross-Validation")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading LoCoMo dataset...")
    loader = LoCoMoDataLoader(data_path="data/locomo/data/locomo10.json")

    train_candidates, val_candidates, test_candidates = loader.train_val_test_split(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # Combine train and val for CV optimization
    optimization_candidates = train_candidates + val_candidates

    print(f"  Optimization set: {len(optimization_candidates)} samples")
    print(f"  Test set: {len(test_candidates)} samples")

    # Initialize extractors
    print("\n[2/4] Initializing feature extractors...")
    extractors = {
        'utility': UtilityExtractor(model_name="qwen2.5:latest"),
        'confidence': ConfidenceExtractor(rouge_metric='rougeL'),
        'novelty': NoveltyExtractor(model_name='all-MiniLM-L6-v2'),
        'recency': RecencyExtractor(decay_rate=0.01),
        'type_prior': TypePriorExtractor()
    }

    # Extract features (this takes time due to LLM calls)
    print("\n[3/4] Extracting features (this may take a while with LLM)...")

    # Limit samples for faster optimization
    max_opt_samples = 150  # Use more samples for better optimization
    max_test_samples = 100

    opt_candidates = optimization_candidates[:max_opt_samples]
    test_candidates = test_candidates[:max_test_samples]

    print(f"  Using {len(opt_candidates)} samples for optimization")
    print(f"  Using {len(test_candidates)} samples for testing")

    # Extract features
    opt_features, opt_labels = extract_features_for_candidates(
        opt_candidates, extractors, use_llm=True
    )

    test_features, test_labels = extract_features_for_candidates(
        test_candidates, extractors, use_llm=True
    )

    # Save features for later use
    np.save('results/opt_features.npy', opt_features)
    np.save('results/opt_labels.npy', opt_labels)
    np.save('results/test_features.npy', test_features)
    np.save('results/test_labels.npy', test_labels)
    print("  Features saved to results/")

    # Optimize weights with CV
    print("\n[4/4] Optimizing weights with cross-validation...")
    best_weights, best_threshold, cv_results = optimize_weights_and_threshold_cv(
        opt_features, opt_labels,
        n_folds=5,
        n_random_starts=200,
        seed=42
    )

    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print(f"Best CV F1: {cv_results['best_cv_f1']:.4f}")
    print(f"Best threshold: {best_threshold}")
    print(f"Best weights:")
    print(f"  Utility:    {best_weights[0]:.4f}")
    print(f"  Confidence: {best_weights[1]:.4f}")
    print(f"  Novelty:    {best_weights[2]:.4f}")
    print(f"  Recency:    {best_weights[3]:.4f}")
    print(f"  TypePrior:  {best_weights[4]:.4f}")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)

    # Our optimized weights
    our_result = evaluate_weights_threshold(test_features, test_labels, best_weights, best_threshold)
    print(f"\nOurs (Optimized):")
    print(f"  Precision: {our_result['precision']:.4f}")
    print(f"  Recall: {our_result['recall']:.4f}")
    print(f"  F1: {our_result['f1']:.4f}")

    # Equal weights baseline
    equal_weights = np.ones(5) / 5
    equal_result = evaluate_weights_threshold(test_features, test_labels, equal_weights, 0.5)
    print(f"\nEqual Weights (threshold=0.5):")
    print(f"  Precision: {equal_result['precision']:.4f}")
    print(f"  Recall: {equal_result['recall']:.4f}")
    print(f"  F1: {equal_result['f1']:.4f}")

    # Equal weights with optimized threshold
    equal_opt_result = evaluate_weights_threshold(test_features, test_labels, equal_weights, best_threshold)
    print(f"\nEqual Weights (threshold={best_threshold}):")
    print(f"  Precision: {equal_opt_result['precision']:.4f}")
    print(f"  Recall: {equal_opt_result['recall']:.4f}")
    print(f"  F1: {equal_opt_result['f1']:.4f}")

    # Improvement
    improvement = (our_result['f1'] - equal_result['f1']) / equal_result['f1'] * 100 if equal_result['f1'] > 0 else 0
    print(f"\nImprovement over Equal Weights: {improvement:+.1f}%")

    # Save results
    results = {
        'weights': best_weights.tolist(),
        'names': ['Utility', 'Confidence', 'Novelty', 'Recency', 'TypePrior'],
        'threshold': best_threshold,
        'cv_f1': cv_results['best_cv_f1'],
        'test_f1': our_result['f1'],
        'test_precision': our_result['precision'],
        'test_recall': our_result['recall'],
        'equal_weights_f1': equal_result['f1'],
        'improvement_percent': improvement
    }

    with open('results/optimized_weights_cv.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/optimized_weights_cv.json")

    return best_weights, best_threshold, our_result


if __name__ == "__main__":
    main()
