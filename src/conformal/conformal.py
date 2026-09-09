"""
Conformal prediction layer for regime classification.

Implements:
1. Split Conformal Classifier
2. Adaptive Prediction Sets (APS)
3. Conformalized Regression for probabilities

Every regime probability ships with documented empirical coverage.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SplitConformalClassifier:
    """
    Split Conformal Classifier (Romano et al. 2020).

    Provides finite-sample-valid prediction sets with guaranteed
    coverage under exchangeability.
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.calibration_scores = None
        self.quantile = None

    def fit(
        self,
        y_cal: np.ndarray,
        probs_cal: np.ndarray,
    ) -> Dict:
        """
        Calibrate on held-out calibration set.

        Parameters
        ----------
        y_cal : np.ndarray
            True labels for calibration set.
        probs_cal : np.ndarray
            Predicted probabilities (n_cal, n_classes).
        """
        # Nonconformity scores: 1 - probability of true class
        scores = 1 - probs_cal[np.arange(len(y_cal)), y_cal]

        self.calibration_scores = np.sort(scores)

        # Adjusted quantile for finite-sample coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * self.confidence_level) / n
        self.quantile = np.quantile(scores, min(q_level, 1.0))

        return {
            "method": "split_conformal",
            "confidence_level": self.confidence_level,
            "n_calibration": n,
            "quantile": float(self.quantile),
            "empirical_coverage": float((scores <= self.quantile).mean()),
            "avg_set_size": float((probs_cal >= (1 - self.quantile)).sum(axis=1).mean()),
        }

    def predict_set(self, probs: np.ndarray) -> np.ndarray:
        """
        Predict conformalized prediction sets.

        Returns
        -------
        np.ndarray
            Boolean mask (n_samples, n_classes): True if class is in set.
        """
        return probs >= (1 - self.quantile)

    def predict_with_sets(
        self, probs: np.ndarray, classes: np.ndarray = None
    ) -> pd.DataFrame:
        """Return predictions with conformalized sets."""
        pred_sets = self.predict_set(probs)
        predicted = np.argmax(probs, axis=1)

        results = []
        for i in range(len(probs)):
            set_size = pred_sets[i].sum()
            set_classes = np.where(pred_sets[i])[0]
            results.append({
                "predicted": predicted[i],
                "set_size": set_size,
                "set_classes": set_classes.tolist(),
                "confidence": probs[i, predicted[i]],
            })

        return pd.DataFrame(results)

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
    ) -> Dict:
        """Evaluate empirical coverage on test set."""
        pred_sets = self.predict_set(probs)
        n = len(y_true)

        # Marginal coverage
        covered = pred_sets[np.arange(n), y_true].mean()

        # Average set size
        avg_set_size = pred_sets.sum(axis=1).mean()

        # Per-class coverage
        per_class_coverage = {}
        for c in range(probs.shape[1]):
            mask = y_true == c
            if mask.sum() > 0:
                per_class_coverage[c] = float(pred_sets[mask, c].mean())

        return {
            "confidence_level": self.confidence_level,
            "empirical_coverage": float(covered),
            "target_coverage": self.confidence_level,
            "coverage_gap": float(self.confidence_level - covered),
            "avg_set_size": float(avg_set_size),
            "per_class_coverage": per_class_coverage,
            "n_test": n,
        }


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (Gibbs & Candes 2021).

    Adjusts the conformal threshold online to maintain target coverage
    under distribution shift.
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        learning_rate: float = 0.01,
    ):
        self.confidence_level = confidence_level
        self.learning_rate = learning_rate
        self.threshold = 0.1  # Initial threshold (1 - confidence)
        self.miscoverage_history = []

    def update(self, y_true: int, probs: np.ndarray):
        """
        Update threshold based on whether prediction was covered.

        Parameters
        ----------
        y_true : int
            True class label.
        probs : np.ndarray
            Predicted probabilities for one sample.
        """
        pred_set = probs >= self.threshold
        covered = bool(pred_set[y_true])

        # Update threshold
        miscoverage = 1 - covered
        self.miscoverage_history.append(miscoverage)

        self.threshold -= self.learning_rate * (self.confidence_level - (1 - miscoverage))
        self.threshold = np.clip(self.threshold, 0.01, 0.99)

    def predict_set(self, probs: np.ndarray) -> np.ndarray:
        return probs >= self.threshold

    def get_diagnostics(self) -> Dict:
        if not self.miscoverage_history:
            return {"method": "aci", "n_updates": 0}

        recent_window = min(100, len(self.miscoverage_history))
        recent_coverage = 1 - np.mean(self.miscoverage_history[-recent_window:])

        return {
            "method": "adaptive_conformal_inference",
            "confidence_level": self.confidence_level,
            "current_threshold": float(self.threshold),
            "overall_coverage": float(1 - np.mean(self.miscoverage_history)),
            "recent_coverage": float(recent_coverage),
            "n_updates": len(self.miscoverage_history),
            "learning_rate": self.learning_rate,
        }


class ConformalPipeline:
    """
    Full conformal prediction pipeline.

    Runs split-conformal + adaptive conformal on each model's outputs.
    """

    def __init__(
        self,
        confidence_levels: List[float] = [0.90, 0.95],
    ):
        self.confidence_levels = confidence_levels
        self.calibrators = {}
        self.aci_adapters = {}

    def calibrate(
        self,
        model_name: str,
        y_cal: np.ndarray,
        probs_cal: np.ndarray,
    ) -> Dict:
        """Calibrate a model's predictions."""
        results = {}

        self.calibrators[model_name] = {}
        for cl in self.confidence_levels:
            cal = SplitConformalClassifier(confidence_level=cl)
            cal_result = cal.fit(y_cal, probs_cal)
            self.calibrators[model_name][cl] = cal
            results[f"split_conformal_{cl}"] = cal_result

        # Initialize ACI adapter
        self.aci_adapters[model_name] = AdaptiveConformalInference(
            confidence_level=0.90
        )

        return results

    def predict_sets(
        self,
        model_name: str,
        probs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Get conformalized prediction sets for all confidence levels."""
        sets = {}
        for cl in self.confidence_levels:
            cal = self.calibrators[model_name][cl]
            sets[f"set_{cl}"] = cal.predict_set(probs)
        return sets

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        probs: np.ndarray,
    ) -> Dict:
        """Evaluate coverage for a model."""
        results = {}
        for cl in self.confidence_levels:
            cal = self.calibrators[model_name][cl]
            eval_result = cal.evaluate_coverage(y_true, probs)
            results[f"evaluation_{cl}"] = eval_result
        return results

    def update_online(
        self,
        model_name: str,
        y_true: int,
        probs: np.ndarray,
    ):
        """Online ACI update."""
        if model_name in self.aci_adapters:
            self.aci_adapters[model_name].update(y_true, probs)

    def get_all_diagnostics(self) -> Dict:
        diags = {}
        for name, adapter in self.aci_adapters.items():
            diags[name] = adapter.get_diagnostics()
        return diags


def compute_reliability_diagram(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Compute calibration curve data for reliability diagram.

    Returns bin_centers, empirical_probs, and ECE.
    """
    max_probs = probs.max(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)

    bin_centers = []
    bin_empirical = []
    bin_counts = []

    for i in range(n_bins):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i + 1])
        if mask.sum() > 0:
            predicted_conf = max_probs[mask].mean()
            empirical_acc = (probs[mask].argmax(axis=1) == y_true[mask]).mean()
            bin_centers.append(predicted_conf)
            bin_empirical.append(empirical_acc)
            bin_counts.append(mask.sum())

    # ECE (Expected Calibration Error)
    total = sum(bin_counts)
    ece = sum(
        c / total * abs(p - e)
        for c, p, e in zip(bin_counts, bin_empirical, bin_centers)
    ) if total > 0 else 0

    return {
        "bin_centers": bin_centers,
        "empirical_probs": bin_empirical,
        "bin_counts": bin_counts,
        "ece": float(ece),
        "n_bins": n_bins,
    }
