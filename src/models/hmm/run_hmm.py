"""
Frequentist HMM — standalone run script.

Runs model selection (3/5/7 states), fits best model, produces
diagnostics and regime assignments. Logs to stdout + saves artifacts.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.models.hmm.frequentist import FrequentistHMM, select_best_n_regimes
from src.ingestion.config import PROCESSED_DIR, REGIME_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("FREQUENTIST HMM — Regime Detection")
    print("=" * 70)

    # Load features
    features_path = PROCESSED_DIR / "features.parquet"
    data_path = PROCESSED_DIR / "merged_market_data.parquet"

    if not features_path.exists() or not data_path.exists():
        print("FATAL: Run ingestion and feature engineering first.")
        sys.exit(1)

    features = pd.read_parquet(features_path)
    raw = pd.read_parquet(data_path)

    print(f"\nFeatures: {features.shape[1]} columns, {features.shape[0]} rows")

    # Select key features for HMM
    # Use returns + volatility + VIX + breadth + flow as core
    key_features = [
        "nifty50_return_21d", "nifty50_return_63d",
        "nifty50_realized_vol_21d", "nifty50_realized_vol_63d",
        "india_vix_level", "india_vix_change_5d",
        "breadth_above_dma50", "cross_index_corr_63d",
        "mid_large_relative_21d",
        "fii_net_z_21d", "dii_net_z_21d",
        "inr_momentum_21d", "gilt_yield_change_21d",
        "nifty50_golden_cross",
    ]

    # Filter to available features
    available = [f for f in key_features if f in features.columns]
    X_full = features[available].values

    # Drop rows with NaN (from rolling windows)
    valid_mask = ~np.isnan(X_full).any(axis=1)
    X = X_full[valid_mask]
    valid_dates = features.index[valid_mask]
    valid_features_names = available

    print(f"Valid samples: {X.shape[0]} (dropped {len(features) - X.shape[0]} from rolling windows)")
    print(f"Features used: {len(valid_features_names)}")

    # Step 1: Model selection
    print("\n[1/3] Model selection: comparing 3/5/7 states...")
    best_n, selection_results = select_best_n_regimes(
        X, candidate_regimes=[3, 5, 7]
    )

    print("\n  BIC comparison:")
    for n, res in sorted(selection_results.items()):
        marker = " <-- BEST" if n == best_n else ""
        print(f"    {n} states: BIC={res['bic']:.1f}, AIC={res['aic']:.1f}{marker}")

    # Step 2: Fit best model
    print(f"\n[2/3] Fitting {best_n}-state HMM...")
    hmm = FrequentistHMM(n_regimes=best_n, n_iter=300)
    diagnostics = hmm.fit(X, feature_names=valid_features_names)

    print(f"  Converged: {diagnostics['converged']}")
    print(f"  Log-likelihood: {diagnostics['log_likelihood']:.1f}")
    print(f"  Iterations used: {diagnostics['n_iter_used']}")

    # Step 3: Regime analysis
    print(f"\n[3/3] Regime analysis...")

    # Get regime assignments
    states = hmm.predict(X)
    probs = hmm.predict_proba(X)

    # Heuristic labelling
    labels = {}
    for s in range(best_n):
        mask = states == s
        if mask.sum() == 0:
            labels[s] = f"State {s}"
            continue
        regime_mean = X[mask].mean(axis=0)
        avg_return = np.mean(regime_mean[:5]) if len(regime_mean) > 5 else 0
        avg_vol = np.mean(regime_mean[10:15]) if len(regime_mean) > 15 else 0
        if avg_return > 0.05 and avg_vol < 0.15:
            labels[s] = "Risk-On"
        elif avg_return < -0.05 and avg_vol > 0.2:
            labels[s] = "Risk-Off"
        elif avg_vol > 0.25:
            labels[s] = "Post-Shock"
        elif avg_return < 0 and avg_vol > 0.15:
            labels[s] = "Late-Cycle"
        else:
            labels[s] = "Transitional"
    print("\n  Regime labels (heuristic):")
    for s, label in sorted(labels.items()):
        n_days = (states == s).sum()
        print(f"    State {s}: {label} ({n_days} days, {n_days/len(states)*100:.1f}%)")

    # Transition matrix
    print("\n  Transition matrix:")
    trans = np.array(diagnostics["transition_matrix"])
    for i in range(trans.shape[0]):
        row_str = "    " + " ".join(f"{trans[i,j]:.3f}" for j in range(trans.shape[1]))
        print(row_str)

    # Expected durations
    print("\n  Expected regime durations:")
    for s in range(best_n):
        stat_key = f"regime_{s}"
        if stat_key in diagnostics["regime_stats"]:
            dur = diagnostics["regime_stats"][stat_key]["avg_duration_days"]
            print(f"    {labels.get(s, f'State {s}')}: {dur:.1f} days")

    # Regime DataFrame
    regime_df = hmm.get_regime_dataframe(X, valid_dates)

    # Save outputs
    out_dir = PROCESSED_DIR / "hmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_df.to_parquet(out_dir / "regime_assignments.parquet")

    with open(out_dir / "diagnostics.json", "w") as f:
        # Convert numpy types for JSON serialization
        diag_serializable = json.loads(
            json.dumps(diagnostics, default=str)
        )
        json.dump(diag_serializable, f, indent=2)

    print(f"\n  Saved: {out_dir / 'regime_assignments.parquet'}")
    print(f"  Saved: {out_dir / 'diagnostics.json'}")

    # Visual summary
    print("\n" + "=" * 70)
    print("FREQUENTIST HMM: COMPLETE")
    print(f"  States: {best_n}")
    print(f"  BIC: {diagnostics['bic']:.1f}")
    print(f"  Regimes in data: {regime_df['regime'].nunique()}")
    print(f"  Confidence range: {regime_df['confidence'].min():.3f} to {regime_df['confidence'].max():.3f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
