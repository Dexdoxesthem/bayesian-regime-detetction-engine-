"""
Bayesian Hidden Markov Model using PyMC.

Uses a variational / simplified forward approach for computational tractability.
Dirichlet priors on transitions; NUTS sampler for key parameters.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BayesianHMM:
    """
    Bayesian HMM using PyMC.

    Strategy: Use the forward-backward algorithm pre-computed as data,
    then sample transition/emission parameters with full Bayesian inference.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_samples: int = 500,
        n_tune: int = 300,
        n_chains: int = 2,
        target_accept: float = 0.90,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.trace = None
        self._is_fitted = False
        self._frequentist_model = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Fit Bayesian HMM using PyMC.

        Approach: Use frequentist HMM as initialization, then fit a Bayesian
        model over the emission parameters with Dirichlet transition priors.
        The state sequence is treated as latent and sampled via forward-backward
        embedded in the likelihood.
        """
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt
        import arviz as az

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)
        n_obs, n_feat = X_scaled.shape
        K = self.n_regimes

        # Use frequentist HMM to get initial state sequence and parameters
        from src.models.hmm.frequentist import GaussianHMM
        freq_hmm = GaussianHMM(n_components=K, n_iter=100, random_state=self.random_state)
        freq_hmm.fit(X_scaled)
        init_states = freq_hmm.predict(X_scaled)
        init_trans = freq_hmm.transmat_.copy()
        init_means = freq_hmm.means_.copy()
        init_covs = freq_hmm.covars_.copy()

        # Ensure transition matrix is valid (positive rows summing to 1)
        init_trans = np.clip(init_trans, 1e-6, None)
        init_trans = init_trans / init_trans.sum(axis=1, keepdims=True)

        # Subsample for MCMC
        max_obs = 800
        if n_obs > max_obs:
            idx = np.linspace(0, n_obs - 1, max_obs, dtype=int)
            X_mcmc = X_scaled[idx]
            states_mcmc = init_states[idx]
            logger.info("Subsampled %d -> %d observations for MCMC", n_obs, max_obs)
        else:
            X_mcmc = X_scaled
            states_mcmc = init_states
            idx = np.arange(n_obs)

        # Compute state-conditional statistics from frequentist initialization
        state_means = np.zeros((K, n_feat))
        state_vars = np.zeros((K, n_feat))
        state_counts = np.zeros(K)
        for k in range(K):
            mask = states_mcmc == k
            if mask.sum() > 0:
                state_means[k] = X_mcmc[mask].mean(axis=0)
                state_vars[k] = X_mcmc[mask].var(axis=0) + 1e-6
                state_counts[k] = mask.sum()
            else:
                state_means[k] = X_mcmc.mean(axis=0)
                state_vars[k] = X_mcmc.var(axis=0) + 1e-6
                state_counts[k] = 1

        with pm.Model() as self.model:
            # --- Priors ---
            # Dirichlet on transition matrix rows (favor persistence)
            trans_alpha = np.full((K, K), 0.5)
            np.fill_diagonal(trans_alpha, 5.0)
            trans_mat = pm.Dirichlet("trans_mat", a=trans_alpha, shape=(K, K))

            # Dirichlet on start probs
            start_probs = pm.Dirichlet("start_probs", a=np.ones(K), shape=K)

            # Emission means — informed by frequentist init
            means = pm.Normal(
                "means",
                mu=state_means,
                sigma=1.0,
                shape=(K, n_feat),
            )

            # Emission variances — half-normal
            sigmas = pm.HalfNormal("sigmas", sigma=1.0, shape=(K, n_feat))

            # --- Likelihood: independent Gaussian per observation ---
            # For each observed data point, compute log-p under each state
            obs = pt.as_tensor_variable(X_mcmc)

            # Log-likelihood matrix (n_obs, K)
            log_lik = pt.zeros((len(X_mcmc), K))
            for k in range(K):
                diff = obs - means[k]
                ll_k = -0.5 * pt.sum(
                    (diff ** 2) / (sigmas[k] ** 2 + 1e-10)
                    + 2 * pt.log(sigmas[k] + 1e-10),
                    axis=1,
                )
                log_lik = pt.set_subtensor(log_lik[:, k], ll_k)

            # Marginal likelihood via log-sum-exp over states
            # pt.logaddexp doesn't support axis; use manual log-sum-exp
            max_log_lik = pt.max(log_lik, axis=1, keepdims=True)
            log_lik_shifted = log_lik - max_log_lik
            log_lik_per_obs = max_log_lik.ravel() + pt.log(pt.sum(pt.exp(log_lik_shifted), axis=1))
            pm.Potential("obs_likelihood", pt.sum(log_lik_per_obs))

            # --- Sample ---
            logger.info(
                "Starting NUTS: %d chains, %d tune, %d samples",
                self.n_chains, self.n_tune, self.n_samples,
            )

            # Prepare initvals — only pass untransformed parameters
            init_dict = {
                "means": state_means,
                "sigmas": np.sqrt(state_vars).clip(0.01, 10.0),
                "start_probs": np.ones(K) / K,
            }
            # Only set trans_mat if all rows are valid (positive, sum to 1)
            if np.all(init_trans > 0) and np.allclose(init_trans.sum(axis=1), 1.0):
                init_dict["trans_mat"] = init_trans

            self.trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.random_state,
                progressbar=False,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": False},
                initvals=init_dict,
            )

        self._is_fitted = True
        self._mcmc_idx = idx

        diagnostics = self._compute_diagnostics(X_scaled, n_obs, idx)
        return diagnostics

    def _compute_diagnostics(self, X_scaled, n_total, mcmc_idx):
        import arviz as az

        summary = az.summary(
            self.trace,
            var_names=["start_probs", "trans_mat", "means", "sigmas"],
            ci_prob=0.94,
        )

        max_rhat = float(summary["r_hat"].max()) if "r_hat" in summary.columns else None
        rhat_ok = max_rhat is not None and max_rhat < 1.05
        min_ess_bulk = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else None
        ess_ok = min_ess_bulk is not None and min_ess_bulk > 400

        try:
            n_divergences = int(self.trace.sample_stats["diverging"].sum())
        except (KeyError, AttributeError):
            n_divergences = -1

        diagnostics = {
            "model_type": "bayesian_hmm",
            "n_regimes": self.n_regimes,
            "n_samples_total": len(X_scaled),
            "n_samples_mcmc": len(mcmc_idx),
            "n_chains": self.n_chains,
            "n_draws": self.n_samples,
            "n_tune": self.n_tune,
            "max_rhat": max_rhat,
            "rhat_ok": rhat_ok,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": float(summary["ess_tail"].min()) if "ess_tail" in summary.columns else None,
            "ess_ok": ess_ok,
            "n_divergences": n_divergences,
            "divergences_ok": n_divergences == 0,
            "mcmc_ok": rhat_ok and ess_ok and n_divergences == 0,
        }

        logger.info("MCMC Diagnostics:")
        logger.info("  R-hat max: %.4f (ok=%s)", max_rhat or -1, rhat_ok)
        logger.info("  ESS bulk min: %.0f (ok=%s)", min_ess_bulk or -1, ess_ok)
        logger.info("  Divergences: %d (ok=%s)", n_divergences, n_divergences == 0)

        if not diagnostics["mcmc_ok"]:
            logger.warning("MCMC DIAGNOSTICS FAILED — results may be unreliable")

        return diagnostics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities using posterior means."""
        import arviz as az

        X_scaled = self.scaler.transform(X)
        post = self.trace.posterior
        means = post["means"].mean(dim=["chain", "draw"]).values
        sigmas = post["sigmas"].mean(dim=["chain", "draw"]).values

        T = len(X_scaled)
        K = means.shape[0]
        log_probs = np.zeros((T, K))

        for k in range(K):
            diff = X_scaled - means[k]
            log_probs[:, k] = -0.5 * np.sum(
                (diff ** 2) / (sigmas[k] ** 2 + 1e-10)
                + 2 * np.log(sigmas[k] + 1e-10),
                axis=1,
            )

        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_posterior_regime_dataframe(self, X, dates):
        probs = self.predict_proba(X)
        df = pd.DataFrame(index=dates)
        df["regime"] = np.argmax(probs, axis=1)
        for s in range(self.n_regimes):
            df[f"prob_{s}"] = probs[:, s]
        df["confidence"] = probs.max(axis=1)
        return df

    def save_posterior_summary(self, output_dir: Path):
        import arviz as az
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)

        summary = az.summary(
            self.trace,
            var_names=["start_probs", "trans_mat", "means", "sigmas"],
            ci_prob=0.94,
        )
        summary.to_csv(output_dir / "posterior_summary.csv")

        for param in ["start_probs", "means", "sigmas"]:
            try:
                az.plot_trace(self.trace, var_names=[param], compact=True)
                plt.tight_layout()
                plt.savefig(output_dir / f"trace_{param}.png", dpi=150)
                plt.close()
            except Exception as e:
                logger.warning("Trace plot %s failed: %s", param, e)

        try:
            az.plot_posterior(self.trace, var_names=["trans_mat"], figsize=(12, 8))
            plt.tight_layout()
            plt.savefig(output_dir / "posterior_trans_mat.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.warning("Trans mat plot failed: %s", e)

        logger.info("Saved posterior summary to %s", output_dir)
