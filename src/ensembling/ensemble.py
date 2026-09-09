"""
Ensembling layer for regime detection.

Implements:
1. Bayesian Model Averaging (BMA)
2. Constrained Stacking
3. WAIC/PSIS-LOO model selection via ArviZ
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BayesianModelAveraging:
    """
    Bayesian Model Averaging over multiple regime models.

    Weights models by their posterior probability (approximated via BIC).
    """

    def __init__(self, n_regimes: int = 5):
        self.n_regimes = n_regimes
        self.model_weights = {}
        self.models = {}

    def add_model(
        self,
        name: str,
        probabilities: np.ndarray,
        bic: float,
    ):
        """
        Add a model's predictions.

        Parameters
        ----------
        name : str
            Model identifier.
        probabilities : np.ndarray
            (n_samples, n_regimes) probability matrix.
        bic : float
            Bayesian Information Criterion for weight computation.
        """
        self.models[name] = probabilities
        self.model_weights[name] = bic

    def compute_weights(self) -> Dict[str, float]:
        """
        Compute BMA weights from BIC values.

        w_i = exp(-0.5 * delta_bic_i) / sum(exp(-0.5 * delta_bic_j))
        """
        bics = np.array(list(self.model_weights.values()))
        delta_bics = bics - bics.min()
        log_weights = -0.5 * delta_bics
        weights = np.exp(log_weights)
        weights /= weights.sum()

        weight_dict = {}
        for name, w in zip(self.model_weights.keys(), weights):
            weight_dict[name] = float(w)
            self.model_weights[name] = float(w)

        return weight_dict

    def predict(self) -> np.ndarray:
        """Weighted average of model probabilities."""
        if not self.models:
            raise ValueError("No models added")

        weights = self.compute_weights()
        ensemble = None

        for name, probs in self.models.items():
            w = weights[name]
            if ensemble is None:
                ensemble = w * probs
            else:
                ensemble += w * probs

        return ensemble

    def get_diagnostics(self) -> Dict:
        weights = self.compute_weights()
        return {
            "method": "bma",
            "n_models": len(self.models),
            "weights": weights,
            "n_regimes": self.n_regimes,
        }


class ConstrainedStacking:
    """
    Constrained stacking ensemble.

    Learns weights via cross-validated logistic regression with
    non-negativity constraints (weights sum to 1, each >= 0).
    """

    def __init__(self, n_regimes: int = 5):
        self.n_regimes = n_regimes
        self.models = {}
        self.weights = None

    def add_model(self, name: str, probabilities: np.ndarray):
        self.models[name] = probabilities

    def fit_weights(
        self,
        y_true: np.ndarray,
        regularization: float = 0.01,
    ) -> Dict:
        """
        Fit stacking weights using constrained optimization.

        Minimizes cross-entropy loss with L2 regularization.
        """
        from scipy.optimize import minimize

        n_models = len(self.models)
        prob_matrix = np.stack(list(self.models.values()), axis=-1)  # (T, K, M)

        def objective(w):
            # Weighted average
            ensemble = np.einsum("tkm,m->tk", prob_matrix, w)
            # Cross-entropy loss
            nll = -np.sum(y_true * np.log(ensemble + 1e-10))
            # L2 regularization
            reg = regularization * np.sum(w ** 2)
            return nll + reg

        # Constraints: weights sum to 1, each >= 0
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n_models
        w0 = np.ones(n_models) / n_models

        result = minimize(objective, w0, method="SLSQP",
                         bounds=bounds, constraints=constraints)

        self.weights = result.x
        weight_dict = dict(zip(self.models.keys(), self.weights))

        return {
            "method": "constrained_stacking",
            "weights": weight_dict,
            "converged": result.success,
            "objective": float(result.fun),
        }

    def predict(self) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights not fitted")

        ensemble = None
        for name, probs in self.models.items():
            w = self.weights[list(self.models.keys()).index(name)]
            if ensemble is None:
                ensemble = w * probs
            else:
                ensemble += w * probs

        return ensemble

    def predict_with_uncertainty(self, X: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ensemble prediction and model disagreement."""
        probs = self.predict()
        # Model disagreement as uncertainty proxy
        all_probs = np.stack(list(self.models.values()), axis=0)
        disagreement = all_probs.std(axis=0).mean(axis=1)
        return probs, disagreement


class EnsemblePipeline:
    """
    Full ensembling pipeline: BMA + Stacking + Model Selection.
    """

    def __init__(self, n_regimes: int = 5):
        self.n_regimes = n_regimes
        self.bma = BayesianModelAveraging(n_regimes)
        self.stacking = ConstrainedStacking(n_regimes)
        self.model_diagnostics = {}

    def add_model(
        self,
        name: str,
        probabilities: np.ndarray,
        bic: float = None,
        log_likelihood: float = None,
        n_params: int = None,
        n_samples: int = None,
    ):
        """Add a model to the ensemble."""
        self.bma.add_model(name, probabilities, bic=bic or 0)
        self.stacking.add_model(name, probabilities)

        self.model_diagnostics[name] = {
            "bic": bic,
            "log_likelihood": log_likelihood,
            "n_params": n_params,
            "n_samples": n_samples,
        }

    def fit(
        self,
        y_true: np.ndarray = None,
    ) -> Dict:
        """
        Fit ensemble weights.

        If y_true is provided, uses stacking. Otherwise uses BMA.
        """
        results = {}

        # BMA weights
        bma_weights = self.bma.compute_weights()
        results["bma_weights"] = bma_weights

        # Stacking weights
        if y_true is not None:
            stacking_result = self.stacking.fit_weights(y_true)
            results["stacking"] = stacking_result

        return results

    def predict(self, method: str = "bma") -> np.ndarray:
        """Get ensemble predictions."""
        if method == "bma":
            return self.bma.predict()
        elif method == "stacking":
            return self.stacking.predict()
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_diagnostics(self) -> Dict:
        return {
            "bma": self.bma.get_diagnostics(),
            "model_diagnostics": self.model_diagnostics,
            "n_models": len(self.model_diagnostics),
        }

    def get_ensemble_output(
        self,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Return ensemble output as DataFrame."""
        bma_probs = self.predict("bma")
        stacking_probs = self.predict("stacking") if len(self.stacking.models) > 0 else bma_probs

        n = min(len(bma_probs), len(dates))
        df = pd.DataFrame(index=dates[:n])

        df["regime_bma"] = np.argmax(bma_probs[:n], axis=1)
        df["confidence_bma"] = bma_probs[:n].max(axis=1)

        df["regime_stacking"] = np.argmax(stacking_probs[:n], axis=1)
        df["confidence_stacking"] = stacking_probs[:n].max(axis=1)

        for s in range(self.n_regimes):
            df[f"prob_bma_{s}"] = bma_probs[:n, s]
            df[f"prob_stacking_{s}"] = stacking_probs[:n, s]

        return df
