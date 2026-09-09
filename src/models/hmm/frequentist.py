"""
Frequentist Hidden Markov Model for regime detection.

Pure Python/NumPy implementation (no hmmlearn dependency).
Uses Baum-Welch (EM) for parameter estimation and
Viterbi/forward-backward for state inference.
"""
import logging
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class GaussianHMM:
    """
    Gaussian Hidden Markov Model with full covariance matrices.

    Pure NumPy implementation using Baum-Welch EM algorithm.
    """

    def __init__(
        self,
        n_components: int = 5,
        n_iter: int = 100,
        tol: float = 1e-4,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.monitor_ = type("Monitor", (), {"converged": False, "iter": 0, "history": []})()

    def _init_params(self, X: np.ndarray):
        """Initialize parameters using K-means."""
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        K = self.n_components

        # K-means initialization
        km = KMeans(n_clusters=K, random_state=self.random_state, n_init=5)
        labels = km.fit_predict(X)

        # Initial means from K-means centers
        self.means_ = km.cluster_centers_.copy()

        # Initial covariances from cluster covariances
        self.covars_ = np.zeros((K, n_features, n_features))
        for k in range(K):
            cluster_data = X[labels == k]
            if len(cluster_data) > 1:
                self.covars_[k] = np.cov(cluster_data.T) + 1e-6 * np.eye(n_features)
            else:
                self.covars_[k] = np.eye(n_features)

        # Initial transition matrix (with slight diagonal bias for stickiness)
        self.transmat_ = np.ones((K, K)) / K
        for k in range(K):
            self.transmat_[k, k] = 0.5 + rng.random() * 0.3
            self.transmat_[k] /= self.transmat_[k].sum()

        # Initial start probabilities (uniform)
        self.startprob_ = np.ones(K) / K

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log-likelihood of each observation under each state."""
        K = self.n_components
        T = X.shape[0]
        log_lik = np.zeros((T, K))

        for k in range(K):
            try:
                mean = self.means_[k]
                cov = self.covars_[k]
                # Regularize covariance
                cov = cov + 1e-6 * np.eye(cov.shape[0])
                rv = sp_stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                log_lik[:, k] = rv.logpdf(X)
            except Exception:
                log_lik[:, k] = -1e10

        return log_lik

    def _forward(self, log_lik: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm. Returns log-alpha and log-likelihood."""
        T, K = log_lik.shape
        log_alpha = np.full((T, K), -np.inf)

        # Initialize
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_lik[0]

        # Forward pass
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = (
                    np.logaddexp.reduce(log_alpha[t - 1] + np.log(self.transmat_[:, j] + 1e-300))
                    + log_lik[t, j]
                )

        log_prob = np.logaddexp.reduce(log_alpha[-1])
        return log_alpha, log_prob

    def _backward(self, log_lik: np.ndarray) -> np.ndarray:
        """Backward algorithm. Returns log-beta."""
        T, K = log_lik.shape
        log_beta = np.full((T, K), -np.inf)
        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = np.logaddexp.reduce(
                    np.log(self.transmat_[i, :] + 1e-300)
                    + log_lik[t + 1, :]
                    + log_beta[t + 1, :]
                )

        return log_beta

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E-step: compute responsibilities."""
        log_lik = self._compute_log_likelihood(X)
        log_alpha, log_prob = self._forward(log_lik)
        log_beta = self._backward(log_lik)

        T, K = log_lik.shape

        # Responsibilities (gamma)
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi (transition responsibilities)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            log_xi_t = (
                log_alpha[t, :, None]
                + np.log(self.transmat_ + 1e-300)
                + log_lik[t + 1, None, :]
                + log_beta[t + 1, None, :]
            )
            log_xi_t -= np.logaddexp.reduce(log_xi_t.ravel())
            xi[t] = np.exp(log_xi_t)

        return gamma, xi, log_prob

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: update parameters."""
        T, K = gamma.shape
        N = X.shape[1]

        # Start probabilities
        self.startprob_ = gamma[0] / gamma[0].sum()

        # Transition matrix
        xi_sum = xi.sum(axis=0)
        self.transmat_ = xi_sum / xi_sum.sum(axis=1, keepdims=True)

        # Means
        for k in range(K):
            wk = gamma[:, k]
            wk_sum = wk.sum()
            if wk_sum > 1e-10:
                self.means_[k] = wk @ X / wk_sum
            else:
                self.means_[k] = X.mean(axis=0)

        # Covariances
        for k in range(K):
            wk = gamma[:, k]
            wk_sum = wk.sum()
            if wk_sum > 1e-10:
                diff = X - self.means_[k]
                self.covars_[k] = (diff.T * wk) @ diff / wk_sum
                # Regularize
                self.covars_[k] += 1e-6 * np.eye(N)
            else:
                self.covars_[k] = np.eye(N)

    def fit(self, X: np.ndarray) -> float:
        """Fit HMM using Baum-Welch EM."""
        self._init_params(X)
        prev_log_prob = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            gamma, xi, log_prob = self._e_step(X)

            # M-step
            self._m_step(X, gamma, xi)

            self.monitor_.history.append(log_prob)

            # Check convergence
            if abs(log_prob - prev_log_prob) < self.tol:
                self.monitor_.converged = True
                self.monitor_.iter = iteration + 1
                break

            prev_log_prob = log_prob

        if not self.monitor_.converged:
            self.monitor_.iter = self.n_iter

        return log_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding for most likely state sequence."""
        log_lik = self._compute_log_likelihood(X)
        T, K = log_lik.shape

        # Viterbi
        log_delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        log_delta[0] = np.log(self.startprob_ + 1e-300) + log_lik[0]

        for t in range(1, T):
            for j in range(K):
                candidates = log_delta[t - 1] + np.log(self.transmat_[:, j] + 1e-300)
                psi[t, j] = np.argmax(candidates)
                log_delta[t, j] = candidates[psi[t, j]] + log_lik[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed state probabilities (forward-backward)."""
        log_lik = self._compute_log_likelihood(X)
        log_alpha, _ = self._forward(log_lik)
        log_beta = self._backward(log_lik)

        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def score(self, X: np.ndarray) -> float:
        """Compute log-likelihood."""
        log_lik = self._compute_log_likelihood(X)
        _, log_prob = self._forward(log_lik)
        return log_prob


class FrequentistHMM:
    """
    Wrapper around GaussianHMM with preprocessing and diagnostics.
    """

    REGIME_LABELS = {
        0: "Risk-On",
        1: "Risk-Off",
        2: "Transitional",
        3: "Late-Cycle",
        4: "Post-Shock",
    }

    def __init__(
        self,
        n_regimes: int = 5,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            n_iter=self.n_iter,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        log_prob = self.model.fit(X_scaled)
        self._is_fitted = True

        n_params = self._count_parameters()
        n_samples = X.shape[0]
        bic = -2 * log_prob + n_params * np.log(n_samples)
        aic = -2 * log_prob + 2 * n_params

        states = self.predict(X)
        regime_stats = self._compute_regime_stats(states, X)

        diagnostics = {
            "model_type": "frequentist_hmm",
            "n_regimes": self.n_regimes,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_params": n_params,
            "log_likelihood": float(log_prob),
            "aic": float(aic),
            "bic": float(bic),
            "converged": bool(self.model.monitor_.converged),
            "n_iter_used": int(self.model.monitor_.iter),
            "regime_stats": regime_stats,
            "transition_matrix": self.model.transmat_.tolist(),
        }

        logger.info(
            "HMM(%d states): LL=%.1f, BIC=%.1f, converged=%s",
            self.n_regimes, log_prob, bic, self.model.monitor_.converged,
        )
        return diagnostics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def _count_parameters(self) -> int:
        n = self.n_regimes
        nf = self.model.means_.shape[1]
        p = (n - 1) + n * (n - 1) + n * nf  # start + trans + means
        if self.covariance_type == "full":
            p += n * nf * (nf + 1) // 2
        elif self.covariance_type == "diag":
            p += n * nf
        elif self.covariance_type == "spherical":
            p += n
        return int(p)

    def _compute_regime_stats(self, states: np.ndarray, X: np.ndarray) -> Dict:
        stats_dict = {}
        for s in range(self.n_regimes):
            mask = states == s
            n_days = mask.sum()
            if n_days == 0:
                continue
            blocks = self._find_contiguous_blocks(mask)
            avg_dur = np.mean([b[1] - b[0] + 1 for b in blocks]) if blocks else 0
            stats_dict[f"regime_{s}"] = {
                "label": self.REGIME_LABELS.get(s, f"State {s}"),
                "n_days": int(n_days),
                "pct_time": float(n_days / len(states)),
                "avg_duration_days": float(avg_dur),
                "n_occurrences": len(blocks),
                "mean_features": X[mask].mean(axis=0).tolist(),
            }
        return stats_dict

    def _find_contiguous_blocks(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        blocks = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                blocks.append((start, i - 1))
                start = None
        if start is not None:
            blocks.append((start, len(mask) - 1))
        return blocks

    def get_regime_dataframe(self, X: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
        states = self.predict(X)
        probs = self.predict_proba(X)
        df = pd.DataFrame({"regime": states}, index=dates)
        for s in range(self.n_regimes):
            df[f"prob_{s}"] = probs[:, s]
        df["confidence"] = probs.max(axis=1)
        return df


def select_best_n_regimes(
    X: np.ndarray,
    candidate_regimes: List[int] = [3, 5, 7],
    n_features_subset: Optional[int] = None,
) -> Tuple[int, Dict]:
    if n_features_subset is not None:
        variances = X.var(axis=0)
        top_idx = np.argsort(variances)[-n_features_subset:]
        X = X[:, top_idx]

    results = {}
    for n in candidate_regimes:
        try:
            hmm = FrequentistHMM(n_regimes=n, n_iter=80)
            diag = hmm.fit(X)
            results[n] = {
                "bic": diag["bic"],
                "aic": diag["aic"],
                "log_likelihood": diag["log_likelihood"],
                "converged": diag["converged"],
            }
        except Exception as e:
            logger.error("HMM(%d states) failed: %s", n, e)
            results[n] = {"bic": np.inf, "aic": np.inf, "error": str(e)}

    best_n = min(results, key=lambda k: results[k].get("bic", np.inf))
    logger.info("Best model: %d states (BIC=%.1f)", best_n, results[best_n]["bic"])
    return best_n, results
