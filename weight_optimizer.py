"""
Weight Optimizer using Bayesian Optimization
Learns optimal weights for the 5 features (U, C, N, R, T).
"""

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import List, Dict, Tuple
import json
import os


class WeightOptimizer:
    """
    Bayesian optimizer for learning feature weights.

    Optimizes weights w = [w_U, w_C, w_N, w_R, w_T] to maximize F1 score.
    """

    def __init__(
        self,
        scorer,
        feature_extractors: Dict,
        n_iter: int = 50,
        init_points: int = 10,
        random_state: int = 42
    ):
        """
        Initialize weight optimizer.

        Args:
            scorer: MemoryAdmissionScorer instance.
            feature_extractors: Dictionary of feature extractors.
            n_iter: Number of optimization iterations.
            init_points: Number of random initialization points.
            random_state: Random seed.
        """
        self.scorer = scorer
        self.feature_extractors = feature_extractors
        self.n_iter = n_iter
        self.init_points = init_points
        self.random_state = random_state

        # Bayesian optimizer
        self.optimizer = None
        self.best_weights = None
        self.best_f1 = 0.0

    def optimize(
        self,
        train_data: List,
        val_data: List = None,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize weights using Bayesian optimization.

        Args:
            train_data: Training data (list of MemoryCandidate).
            val_data: Validation data (optional, uses train if None).
            threshold: Admission threshold.

        Returns:
            (best_weights, best_f1_score)
        """
        if val_data is None:
            val_data = train_data

        print(f"Optimizing weights on {len(val_data)} validation samples...")
        print(f"  Optimization iterations: {self.n_iter}")
        print(f"  Random init points: {self.init_points}")

        # Define objective function
        def objective(**weights_dict):
            """
            Objective function to maximize: F1 score.

            Args:
                **weights_dict: Dictionary of weight parameters.

            Returns:
                F1 score (to be maximized).
            """
            # Extract weights
            w_U = weights_dict['w_U']
            w_C = weights_dict['w_C']
            w_N = weights_dict['w_N']
            w_R = weights_dict['w_R']
            w_T = weights_dict['w_T']

            weights = np.array([w_U, w_C, w_N, w_R, w_T])

            # Normalize weights to sum to 1
            weights = weights / weights.sum()

            # Update scorer weights
            self.scorer.weights = weights
            self.scorer.threshold = threshold

            # Evaluate on validation data
            y_true = []
            y_pred = []

            for candidate in val_data:
                # Extract features
                try:
                    features = self.scorer.compute_features(
                        candidate.dialogue_turn,
                        candidate.conversation_history,
                        []  # No existing memories for simplicity
                    )

                    # Score and decide
                    score = self.scorer.score(features)
                    decision = self.scorer.should_admit(score)

                    y_true.append(1 if candidate.is_referenced else 0)
                    y_pred.append(1 if decision else 0)

                except Exception as e:
                    # Skip candidates that fail (e.g., LLM timeout)
                    continue

            # Compute F1 score
            if len(y_true) == 0 or sum(y_pred) == 0:
                return 0.0  # Avoid division by zero

            f1 = f1_score(y_true, y_pred, zero_division=0)

            return f1

        # Define parameter bounds
        pbounds = {
            'w_U': (0.0, 1.0),
            'w_C': (0.0, 1.0),
            'w_N': (0.0, 1.0),
            'w_R': (0.0, 1.0),
            'w_T': (0.0, 1.0),
        }

        # Initialize optimizer
        self.optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=self.random_state,
            verbose=2
        )

        # Optional: Add logger
        log_path = "results/weight_optimization.json"
        os.makedirs("results", exist_ok=True)
        logger = JSONLogger(path=log_path)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Run optimization
        self.optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter
        )

        # Extract best weights
        best_params = self.optimizer.max['params']
        raw_weights = np.array([
            best_params['w_U'],
            best_params['w_C'],
            best_params['w_N'],
            best_params['w_R'],
            best_params['w_T']
        ])

        # Normalize
        self.best_weights = raw_weights / raw_weights.sum()
        self.best_f1 = self.optimizer.max['target']

        print(f"\nOptimization complete!")
        print(f"  Best F1: {self.best_f1:.4f}")
        print(f"  Best weights (normalized):")
        print(f"    w_U: {self.best_weights[0]:.4f}")
        print(f"    w_C: {self.best_weights[1]:.4f}")
        print(f"    w_N: {self.best_weights[2]:.4f}")
        print(f"    w_R: {self.best_weights[3]:.4f}")
        print(f"    w_T: {self.best_weights[4]:.4f}")

        return self.best_weights, self.best_f1

    def grid_search(
        self,
        val_data: List,
        threshold: float = 0.5,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Alternative: Random grid search for weights.

        Faster than Bayesian optimization but less efficient.

        Args:
            val_data: Validation data.
            threshold: Admission threshold.
            n_samples: Number of random weight combinations to try.

        Returns:
            (best_weights, best_f1_score)
        """
        print(f"Grid search with {n_samples} random samples...")

        best_weights = None
        best_f1 = 0.0

        np.random.seed(self.random_state)

        for i in range(n_samples):
            # Generate random weights
            weights = np.random.uniform(0, 1, size=5)
            weights = weights / weights.sum()

            # Update scorer
            self.scorer.weights = weights
            self.scorer.threshold = threshold

            # Evaluate
            y_true = []
            y_pred = []

            for candidate in val_data:
                try:
                    features = self.scorer.compute_features(
                        candidate.dialogue_turn,
                        candidate.conversation_history,
                        []
                    )

                    score = self.scorer.score(features)
                    decision = self.scorer.should_admit(score)

                    y_true.append(1 if candidate.is_referenced else 0)
                    y_pred.append(1 if decision else 0)

                except Exception:
                    continue

            if len(y_true) == 0 or sum(y_pred) == 0:
                continue

            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights.copy()

            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{n_samples}, Best F1: {best_f1:.4f}")

        self.best_weights = best_weights
        self.best_f1 = best_f1

        print(f"\nGrid search complete!")
        print(f"  Best F1: {self.best_f1:.4f}")
        print(f"  Best weights:")
        print(f"    w_U: {self.best_weights[0]:.4f}")
        print(f"    w_C: {self.best_weights[1]:.4f}")
        print(f"    w_N: {self.best_weights[2]:.4f}")
        print(f"    w_R: {self.best_weights[3]:.4f}")
        print(f"    w_T: {self.best_weights[4]:.4f}")

        return self.best_weights, self.best_f1

    def save_weights(self, filepath: str):
        """Save optimized weights to JSON file."""
        if self.best_weights is None:
            raise ValueError("No optimized weights to save. Run optimize() first.")

        weights_dict = {
            'weights': self.best_weights.tolist(),
            'f1_score': float(self.best_f1),
            'names': ['w_U', 'w_C', 'w_N', 'w_R', 'w_T']
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(weights_dict, f, indent=2)

        print(f"Weights saved to {filepath}")

    @staticmethod
    def load_weights(filepath: str) -> np.ndarray:
        """Load weights from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return np.array(data['weights'])


if __name__ == "__main__":
    print("Weight Optimizer Test")
    print("=" * 60)
    print("Note: This is a standalone test with dummy data.")
    print("For real optimization, use run_experiments.py")
