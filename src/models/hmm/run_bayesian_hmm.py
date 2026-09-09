"""
Bayesian HMM — standalone run script.

Runs PyMC HMM with Dirichlet priors, NUTS sampling, full diagnostics.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.hmm.bayesian import BayesianHMM
from src.ingestion.config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("BAYESIAN HMM — PyMC NUTS Sampler")
    print("=" * 70)

    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("FATAL: Run feature engineering first.")
        sys.exit(1)

    features = pd.read_parquet(features_path)

    # Use reduced feature set for Bayesian model (MCMC is slower)
    key_features = [
        "nifty50_return_21d", "nifty50_return_63d",
        "nifty50_realized_vol_21d",
        "india_vix_level",
        "breadth_above_dma50",
        "mid_large_relative_21d",
        "fii_net_z_21d",
        "inr_momentum_21d",
    ]

    available = [f for f in key_features if f in features.columns]
    X_full = features[available].values
    valid_mask = ~np.isnan(X_full).any(axis=1)
    X = X_full[valid_mask]
    valid_dates = features.index[valid_mask]

    print(f"\nFeatures: {len(available)}, Valid samples: {X.shape[0]}")

    # Fit Bayesian HMM
    print("\n[1/1] Fitting Bayesian HMM (NUTS)...")
    print("  This may take several minutes depending on hardware.")

    bayes_hmm = BayesianHMM(
        n_regimes=5,
        n_samples=500,
        n_tune=500,
        n_chains=2,
        target_accept=0.98,
        random_state=42,
    )

    try:
        diagnostics = bayes_hmm.fit(X, feature_names=available)
    except Exception as e:
        logger.error("Bayesian HMM failed: %s", e)
        print(f"\nFATAL: Bayesian HMM failed — {e}")
        sys.exit(1)

    # Report diagnostics
    print("\n--- MCMC Diagnostics ---")
    print(f"  R-hat (max):      {diagnostics['max_rhat']:.4f} {'OK' if diagnostics['rhat_ok'] else 'FAILED'}")
    print(f"  ESS bulk (min):   {diagnostics['min_ess_bulk']:.0f} {'OK' if diagnostics['ess_ok'] else 'FAILED'}")
    print(f"  ESS tail (min):   {diagnostics['min_ess_tail']:.0f}")
    print(f"  Divergences:      {diagnostics['n_divergences']} {'OK' if diagnostics['divergences_ok'] else 'FAILED'}")
    print(f"  Overall MCMC OK:  {diagnostics['mcmc_ok']}")

    if not diagnostics["mcmc_ok"]:
        print("\nWARNING: MCMC diagnostics failed. Results may be unreliable.")
        print("Consider increasing n_samples, n_tune, or target_accept.")

    # Posterior regime assignments
    probs = bayes_hmm.predict_proba(X)
    regimes = np.argmax(probs, axis=1)

    print("\n--- Regime Distribution ---")
    for s in range(5):
        n = (regimes == s).sum()
        print(f"  State {s}: {n} days ({n/len(regimes)*100:.1f}%)")

    # Save outputs
    out_dir = PROCESSED_DIR / "bayesian_hmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_df = bayes_hmm.get_posterior_regime_dataframe(X, valid_dates)
    regime_df.to_parquet(out_dir / "regime_assignments.parquet")

    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    # Save posterior summary plots
    bayes_hmm.save_posterior_summary(out_dir)

    print(f"\n  Saved: {out_dir}")
    print("\n" + "=" * 70)
    print("BAYESIAN HMM: COMPLETE")
    print(f"  MCMC OK: {diagnostics['mcmc_ok']}")
    print(f"  R-hat: {diagnostics['max_rhat']:.4f}")
    print(f"  ESS: {diagnostics['min_ess_bulk']:.0f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
