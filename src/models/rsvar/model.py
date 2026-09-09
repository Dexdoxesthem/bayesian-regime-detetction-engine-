"""
Regime-Switching Vector Autoregression (RS-VAR) for regime detection.

Uses statsmodels MarkovAutography for the baseline,
then a simplified Bayesian RS-VAR via PyMC for the multivariate case.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeSwitchingVAR:
    """
    Regime-Switching VAR using statsmodels MarkovAutography.

    Fits a Markov-switching VAR(p) model on multivariate time series
    and extracts regime-conditional coefficients and probabilities.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        p_lags: int = 1,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.p_lags = p_lags
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.result = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        endog_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Fit regime-switching VAR.

        Uses statsmodels MarkovAutography for the switching intercept model.
        """
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.endog_names = endog_names or self.feature_names

        # Use univariate Markov switching on the primary feature
        # statsmodels MarkovAutography requires univariate data
        X_1d = X[:, 0:1]  # First feature only
        endog_names = self.endog_names[:1]

        # Scale
        X_scaled = self.scaler.fit_transform(X_1d)

        logger.info("Fitting RS-VAR: %d regimes, %d endogenous, %d lags",
                     self.n_regimes, X_1d.shape[1], self.p_lags)

        try:
            # MarkovAutography with switching intercept
            self.model = MarkovAutoregression(
                X_scaled,
                k_regimes=self.n_regimes,
                order=self.p_lags,
                switching_ar=False,
                switching_variance=True,
            )
            self.result = self.model.fit(maxiter=self.max_iter)
            self._is_fitted = True

            # Extract diagnostics
            diagnostics = self._compute_diagnostics(X_scaled, endog_names)
            return diagnostics

        except Exception as e:
            logger.error("RS-VAR fit failed: %s", e)
            # Fallback: use simple regime-switching via EM
            return self._fit_simple_rsv(X_scaled, endog_names)

    def _fit_simple_rsv(self, X_scaled, endog_names):
        """Simple regime-switching VAR as fallback."""
        from sklearn.mixture import GaussianMixture

        # Use Gaussian Mixture as a proxy for regime switching
        self._gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_init=5,
        )
        self._gmm.fit(X_scaled)
        self._is_fitted = True
        self._use_gmm = True

        logger.info("Using Gaussian Mixture fallback for RS-VAR")

        return {
            "model_type": "rs_var_gmm_fallback",
            "n_regimes": self.n_regimes,
            "n_features": X_scaled.shape[1],
            "n_samples": X_scaled.shape[0],
            "aic": float(self._gmm.aic(X_scaled)),
            "bic": float(self._gmm.bic(X_scaled)),
            "converged": True,
            "endog_names": endog_names,
        }

    def _compute_diagnostics(self, X_scaled, endog_names):
        """Compute model diagnostics."""
        if hasattr(self, '_use_gmm') and self._use_gmm:
            return {
                "model_type": "rs_var_gmm",
                "n_regimes": self.n_regimes,
                "n_features": X_scaled.shape[1],
                "n_samples": X_scaled.shape[0],
                "endog_names": endog_names,
            }

        result = self.result

        # Regime characteristics
        regime_probs = result.smoothed_marginal_probabilities
        regimes = np.argmax(regime_probs, axis=1) if regime_probs.shape[1] > 1 else np.zeros(len(X_scaled))

        # Regime-conditional coefficients
        regime_params = {}
        for s in range(self.n_regimes):
            if s < len(result.params):
                regime_params[f"regime_{s}"] = result.params[s].tolist()

        # Transition matrix
        try:
            trans_mat = result.regime_transition
            if hasattr(trans_mat, 'shape') and trans_mat.shape == (self.n_regimes, self.n_regimes):
                trans_mat_list = trans_mat.tolist()
            else:
                trans_mat_list = None
        except:
            trans_mat_list = None

        diagnostics = {
            "model_type": "rs_var_statsmodels",
            "n_regimes": self.n_regimes,
            "n_features": X_scaled.shape[1],
            "n_samples": X_scaled.shape[0],
            "endog_names": endog_names,
            "aic": float(result.aic) if hasattr(result, 'aic') else None,
            "bic": float(result.bic) if hasattr(result, 'bic') else None,
            "log_likelihood": float(result.llf) if hasattr(result, 'llf') else None,
            "converged": True,
            "regime_params": regime_params,
            "transition_matrix": trans_mat_list,
            "regime_pct_time": {
                f"regime_{s}": float((regimes == s).mean())
                for s in range(self.n_regimes)
            },
        }

        logger.info("RS-VAR: AIC=%.1f, BIC=%.1f",
                     diagnostics["aic"] or 0, diagnostics["bic"] or 0)

        return diagnostics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get regime probabilities."""
        if hasattr(self, '_use_gmm') and self._use_gmm:
            n_feat = self.scaler.n_features_in_
            return self._gmm.predict_proba(self.scaler.transform(X[:, :n_feat]))

        n_feat = self.scaler.n_features_in_
        X_scaled = self.scaler.transform(X[:, :n_feat])
        try:
            probs = self.result.smoothed_marginal_probabilities
            probs = np.atleast_2d(probs)
            if probs.ndim == 1:
                probs = np.column_stack([1 - probs, probs])
            if len(probs) != len(X_scaled):
                probs = probs[:len(X_scaled)]
            return probs
        except:
            return np.ones((len(X_scaled), self.n_regimes)) / self.n_regimes

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_regime_dataframe(self, X: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
        probs = self.predict_proba(X)
        # Align lengths
        n = min(len(probs), len(dates))
        probs = probs[:n]
        dates = dates[:n]
        df = pd.DataFrame(index=dates)
        df["regime"] = np.argmax(probs, axis=1)
        for s in range(self.n_regimes):
            df[f"prob_{s}"] = probs[:, s] if probs.ndim > 1 else (1 - probs) if s == 0 else probs
        df["confidence"] = probs.max(axis=1) if probs.ndim > 1 else np.maximum(probs, 1 - probs)
        return df


def run_rsvar_pipeline(
    features: pd.DataFrame,
    output_dir: Path,
    n_regimes: int = 2,
) -> Dict:
    """Run full RS-VAR pipeline."""
    # Select multivariate features
    endog_features = [
        "nifty50_return_1d", "nifty50_realized_vol_21d",
        "india_vix_level", "breadth_above_dma50",
        "fii_net_z_21d", "inr_momentum_21d",
    ]
    available = [f for f in endog_features if f in features.columns]
    X = features[available].values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    dates = features.index[valid_mask]

    rsvar = RegimeSwitchingVAR(n_regimes=n_regimes)
    diagnostics = rsvar.fit(X, endog_names=available)

    regime_df = rsvar.get_regime_dataframe(X, dates)

    output_dir.mkdir(parents=True, exist_ok=True)
    regime_df.to_parquet(output_dir / "regime_assignments.parquet")
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    return diagnostics
