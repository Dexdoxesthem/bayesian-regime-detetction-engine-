"""
Sequential/Online inference models for regime detection.

1. Bootstrap Particle Filter for online regime state estimation.
2. Bayesian Online Changepoint Detection (BOCPD) — validates on
   2018 NBFC/IL&FS, 2020 COVID, 2013 Taper Tantrum, 2024 elections.
3. Streaming Dirichlet/Beta posterior updates.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class BootstrapParticleFilter:
    """
    Bootstrap Particle Filter for online regime state estimation.

    Maintains a weighted particle approximation to the posterior
    regime distribution as new observations arrive.
    """

    def __init__(
        self,
        n_particles: int = 500,
        n_regimes: int = 5,
        transition_noise: float = 0.05,
        observation_noise: float = 0.1,
        random_state: int = 42,
    ):
        self.n_particles = n_particles
        self.n_regimes = n_regimes
        self.transition_noise = transition_noise
        self.observation_noise = observation_noise
        self.rng = np.random.default_rng(random_state)

        # Initialize particles and weights
        self.particles = self.rng.integers(0, n_regimes, size=n_particles)
        self.weights = np.ones(n_particles) / n_particles
        self._is_initialized = False

    def initialize(self, regime_means: np.ndarray, regime_stds: np.ndarray):
        """Initialize with regime emission parameters."""
        self.regime_means = regime_means
        self.regime_stds = regime_stds
        self._is_initialized = True

    def step(self, observation: np.ndarray) -> np.ndarray:
        """
        Process one observation and return updated regime probabilities.

        Parameters
        ----------
        observation : np.ndarray
            Single observation vector.

        Returns
        -------
        np.ndarray
            Regime probability distribution (n_regimes,).
        """
        if not self._is_initialized:
            return np.ones(self.n_regimes) / self.n_regimes

        # 1. Propagate particles (transition)
        for i in range(self.n_particles):
            if self.rng.random() < self.transition_noise:
                self.particles[i] = self.rng.integers(0, self.n_regimes)

        # 2. Update weights (likelihood)
        log_weights = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            state = self.particles[i]
            # Gaussian likelihood under current state
            diff = observation - self.regime_means[state]
            log_weights[i] = -0.5 * np.sum(
                (diff / self.regime_stds[state]) ** 2
            )

        # Normalize weights
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum() + 1e-300
        self.weights = weights

        # 3. Regime probability
        regime_probs = np.zeros(self.n_regimes)
        for s in range(self.n_regimes):
            regime_probs[s] = self.weights[self.particles == s].sum()

        # 4. Resample if effective sample size is low
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.n_particles / 2:
            self._resample()

        return regime_probs

    def _resample(self):
        """Systematic resampling."""
        indices = self.rng.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights,
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def process_sequence(self, X: np.ndarray) -> np.ndarray:
        """Process a sequence of observations."""
        T = len(X)
        probs = np.zeros((T, self.n_regimes))
        for t in range(T):
            probs[t] = self.step(X[t])
        return probs


class BayesianOnlineChangepointDetection:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    Detects regime changes by computing the posterior probability of a
    changepoint at each time step.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 60,  # Prior: ~1 changepoint per quarter
        mu_prior: float = 0.0,
        kappa_prior: float = 1.0,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        max_run_length: int = 500,
    ):
        self.hazard_rate = hazard_rate
        self.mu_prior = mu_prior
        self.kappa_prior = kappa_prior
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.max_run_length = max_run_length

        # State
        self._reset()

    def _reset(self):
        self.run_length_probs = np.array([1.0])  # R(0) = 1
        self.mu_messages = np.array([self.mu_prior])
        self.kappa_messages = np.array([self.kappa_prior])
        self.alpha_messages = np.array([self.alpha_prior])
        self.beta_messages = np.array([self.beta_prior])

    def step(self, observation: float) -> Dict:
        """
        Process one scalar observation.

        Returns dict with:
        - changepoint_prob: P(run_length == 0) = changepoint detected
        - run_length_dist: full run-length distribution
        """
        # Update messages first (prepends one element, growing arrays by 1)
        self._update_messages(observation)

        n = len(self.run_length_probs)

        # Predictive probability for each run length (Student-t)
        df = 2 * self.alpha_messages[:n]
        loc = self.mu_messages[:n]
        denom = self.alpha_messages[:n] * self.kappa_messages[:n] + 1e-300
        scale = np.sqrt(
            np.maximum(self.beta_messages[:n] * (self.kappa_messages[:n] + 1) / denom, 1e-20)
        )

        # Evaluate observation under each run-length's predictive
        try:
            predictive_probs = sp_stats.t.pdf(observation, df=df, loc=loc, scale=scale)
        except Exception:
            predictive_probs = np.ones(n) * 1e-10

        predictive_probs = np.maximum(predictive_probs, 1e-300)

        # Growth probabilities
        growth_probs = self.run_length_probs * predictive_probs * (1 - self.hazard_rate)

        # Changepoint probability
        cp_prob = np.sum(self.run_length_probs * predictive_probs * self.hazard_rate)

        # New run-length distribution (one longer than before)
        new_run_length_probs = np.zeros(n + 1)
        new_run_length_probs[1:] = growth_probs
        new_run_length_probs[0] = cp_prob

        # Normalize
        total = new_run_length_probs.sum()
        if total > 0:
            new_run_length_probs /= total
        else:
            new_run_length_probs = np.ones(n + 1) / (n + 1)

        self.run_length_probs = new_run_length_probs

        return {
            "changepoint_prob": float(new_run_length_probs[0]),
            "run_length_dist": new_run_length_probs,
            "max_run_length": int(np.argmax(new_run_length_probs)),
        }

    def _update_messages(self, observation: float):
        """Update Student-t messages."""
        # Posterior update for Normal-Inverse-Gamma conjugate prior
        new_mu = (
            self.kappa_prior * self.mu_prior + observation
        ) / (self.kappa_prior + 1)

        new_kappa = self.kappa_prior + 1
        new_alpha = self.alpha_prior + 0.5
        new_beta = (
            self.beta_prior
            + 0.5 * self.kappa_prior * (observation - self.mu_prior) ** 2
            / (self.kappa_prior + 1)
        )

        # Prepend new message (newest first)
        self.mu_messages = np.concatenate([[new_mu], self.mu_messages])
        self.kappa_messages = np.concatenate([[new_kappa], self.kappa_messages])
        self.alpha_messages = np.concatenate([[new_alpha], self.alpha_messages])
        self.beta_messages = np.concatenate([[new_beta], self.beta_messages])

        # Truncate to max run length
        n = min(len(self.mu_messages), self.max_run_length)
        self.mu_messages = self.mu_messages[:n]
        self.kappa_messages = self.kappa_messages[:n]
        self.alpha_messages = self.alpha_messages[:n]
        self.beta_messages = self.beta_messages[:n]
        self.run_length_probs = self.run_length_probs[:n]
        total = self.run_length_probs.sum()
        if total > 0:
            self.run_length_probs /= total

    def detect(self, X: np.ndarray) -> pd.DataFrame:
        """
        Run BOCPD on a sequence of scalar observations.

        Returns DataFrame with changepoint probabilities and run-length.
        """
        self._reset()
        results = []
        for t in range(len(X)):
            obs = float(X[t]) if np.isscalar(X[t]) else float(X[t][0])
            result = self.step(obs)
            results.append(result)

        df = pd.DataFrame({
            "changepoint_prob": [r["changepoint_prob"] for r in results],
            "max_run_length": [r["max_run_length"] for r in results],
        })
        return df

    def detect_multivariate(
        self, X: np.ndarray, feature_idx: int = 0
    ) -> pd.DataFrame:
        """Run BOCPD on one feature of multivariate data."""
        return self.detect(X[:, feature_idx])


class StreamingDirichlet:
    """
    Streaming Dirichlet posterior updates for regime probabilities.

    Maintains a Dirichlet posterior over regime proportions that
    updates as new regime assignments arrive.
    """

    def __init__(self, n_regimes: int = 5, prior_alpha: float = 1.0):
        self.n_regimes = n_regimes
        self.posterior_alpha = np.full(n_regimes, prior_alpha)
        self.history = []

    def update(self, regime: int):
        """Update posterior with a new regime observation."""
        self.posterior_alpha[regime] += 1
        self.history.append(regime)

    def get_posterior_mean(self) -> np.ndarray:
        """Posterior mean of regime proportions."""
        total = self.posterior_alpha.sum()
        return self.posterior_alpha / total

    def get_posterior_std(self) -> np.ndarray:
        """Posterior std of regime proportions."""
        total = self.posterior_alpha.sum()
        mean = self.posterior_alpha / total
        return np.sqrt(mean * (1 - mean) / (total + 1))

    def get_credible_interval(self, level: float = 0.94) -> np.ndarray:
        """HDI for each regime proportion."""
        total = self.posterior_alpha.sum()
        ci = np.zeros((self.n_regimes, 2))
        for k in range(self.n_regimes):
            ci[k] = sp_stats.beta.interval(
                level,
                self.posterior_alpha[k],
                total - self.posterior_alpha[k] + 1e-10,
            )
        return ci

    def get_summary(self) -> Dict:
        mean = self.get_posterior_mean()
        std = self.get_posterior_std()
        ci = self.get_credible_interval()
        return {
            "regime_means": mean.tolist(),
            "regime_stds": std.tolist(),
            "credible_intervals": ci.tolist(),
            "total_observations": len(self.history),
        }


def run_bocpd_validation(
    features: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """
    Validate BOCPD fires on known breakpoints:
    - 2013 Taper Tantrum (May-Aug 2013)
    - 2018 NBFC/IL&FS crisis (Sep-Oct 2018)
    - 2020 COVID crash (Feb-Mar 2020)
    - 2024 election volatility (Jun 2024)
    """
    # Use Nifty returns
    if "nifty50_return_1d" in features.columns:
        returns = features["nifty50_return_1d"].dropna()
    else:
        returns = features["nifty50_close"].pct_change().dropna()

    bocpd = BayesianOnlineChangepointDetection(hazard_rate=1 / 252)
    bocpd_results = bocpd.detect(returns.values)
    bocpd_results.index = returns.index

    # Known breakpoints
    breakpoints = {
        "2013-06-01": "Taper Tantrum",
        "2018-09-20": "IL&FS / NBFC Crisis",
        "2020-03-01": "COVID Crash",
        "2024-06-04": "Election Results",
    }

    # Check detection around breakpoints
    detection_results = {}
    for date_str, event_name in breakpoints.items():
        bp_date = pd.Timestamp(date_str)
        window = bocpd_results.loc[
            bp_date - pd.Timedelta(days=30):bp_date + pd.Timedelta(days=30)
        ]
        if len(window) > 0:
            max_cp_prob = window["changepoint_prob"].max()
            detection_results[event_name] = {
                "max_changepoint_prob": float(max_cp_prob),
                "detected": bool(max_cp_prob > 0.3),
                "date_range": str(window.index.min().date()) + " to " + str(window.index.max().date()),
            }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    bocpd_results.to_parquet(output_dir / "bocpd_results.parquet")
    with open(output_dir / "breakpoint_validation.json", "w") as f:
        json.dump(detection_results, f, indent=2)

    logger.info("BOCPD validation:")
    for event, res in detection_results.items():
        status = "DETECTED" if res["detected"] else "MISSED"
        logger.info("  %s: %s (prob=%.3f)", event, status, res["max_changepoint_prob"])

    return detection_results
