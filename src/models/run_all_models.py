"""
Run all models and produce diagnostics.

Tests: Frequentist HMM, Bayesian HMM, RS-VAR, BDL, BOCPD.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def make_regime_labels(y_continuous):
    """Convert continuous return series to regime labels for supervised models."""
    # 5 regimes based on forward 21-day returns
    fwd_ret = pd.Series(y_continuous).rolling(21).mean().fillna(0)
    labels = pd.qcut(fwd_ret, q=5, labels=[0, 1, 2, 3, 4], duplicates="drop")
    return labels.values.astype(int)


def run_hmm_models(features):
    """Run frequentist and Bayesian HMM."""
    from src.models.hmm.frequentist import FrequentistHMM, select_best_n_regimes

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
    available = [f for f in key_features if f in features.columns]
    X = features[available].values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    dates = features.index[valid_mask]

    # Model selection
    best_n, selection = select_best_n_regimes(X, [3, 5, 7])

    # Fit best model
    hmm = FrequentistHMM(n_regimes=best_n, n_iter=200)
    diag = hmm.fit(X, feature_names=available)
    regime_df = hmm.get_regime_dataframe(X, dates)

    return diag, regime_df


def run_rsvar(features):
    """Run RS-VAR."""
    from src.models.rsvar.model import RegimeSwitchingVAR

    endog_features = [
        "nifty50_return_1d", "nifty50_realized_vol_21d",
        "india_vix_level", "breadth_above_dma50",
    ]
    available = [f for f in endog_features if f in features.columns]
    X = features[available].values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    dates = features.index[valid_mask]

    rsvar = RegimeSwitchingVAR(n_regimes=2)
    diag = rsvar.fit(X, endog_names=available)
    regime_df = rsvar.get_regime_dataframe(X, dates)

    return diag, regime_df


def run_bdl(features):
    """Run Bayesian Deep Learning models (PyTorch)."""
    from src.models.bdl.model import MCDropoutClassifier, VariationalBNN, DeepEnsemble

    key_features = [
        "nifty50_return_21d", "nifty50_realized_vol_21d",
        "india_vix_level", "breadth_above_dma50",
        "mid_large_relative_21d", "fii_net_z_21d",
        "inr_momentum_21d", "gilt_yield_change_21d",
    ]
    available = [f for f in key_features if f in features.columns]
    X = features[available].values

    # Create labels
    y = make_regime_labels(features["nifty50_return_1d"].values
                           if "nifty50_return_1d" in features.columns
                           else np.zeros(len(features)))

    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    # MC Dropout
    mc = MCDropoutClassifier(
        input_dim=X.shape[1], n_classes=5,
        hidden_dims=(32, 16), n_mc_samples=50,
        n_epochs=50, batch_size=128,
    )
    mc_diag = mc.fit(X, y, feature_names=available)

    # Variational BNN
    vbnn = VariationalBNN(
        input_dim=X.shape[1], n_classes=5,
        hidden_dims=(32, 16), n_epochs=50, batch_size=128,
    )
    vbnn_diag = vbnn.fit(X, y, feature_names=available)

    # Deep ensemble (M=3 for speed)
    de = DeepEnsemble(
        M=3, input_dim=X.shape[1], n_classes=5,
        hidden_dims=(32, 16), n_epochs=50, batch_size=128,
    )
    de_diag = de.fit(X, y, feature_names=available)

    combined_diag = {
        "mc_dropout": mc_diag,
        "variational_bnn": vbnn_diag,
        "deep_ensemble": de_diag,
    }

    return combined_diag, None


def run_bocpd(features):
    """Run BOCPD changepoint detection."""
    from src.models.sequential.model import run_bocpd_validation

    out_dir = PROCESSED_DIR / "bocpd"
    results = run_bocpd_validation(features, out_dir)
    return results


def main():
    print("=" * 70)
    print("ALL MODELS — Run and Validate")
    print("=" * 70)

    # Load features
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("FATAL: Run feature engineering first.")
        sys.exit(1)

    features = pd.read_parquet(features_path)
    print(f"\nFeatures: {features.shape[1]} cols, {features.shape[0]} rows")

    all_results = {}

    # 1. Frequentist HMM
    print("\n[1/4] Frequentist HMM...")
    try:
        hmm_diag, hmm_regimes = run_hmm_models(features)
        all_results["frequentist_hmm"] = hmm_diag
        hmm_out = PROCESSED_DIR / "hmm"
        hmm_regimes.to_parquet(hmm_out / "regime_assignments.parquet")
        with open(hmm_out / "diagnostics.json", "w") as f:
            json.dump(hmm_diag, f, indent=2, default=str)
        print(f"  OK: BIC={hmm_diag['bic']:.1f}, converged={hmm_diag['converged']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["frequentist_hmm"] = {"error": str(e)}

    # 2. RS-VAR
    print("\n[2/4] Regime-Switching VAR...")
    try:
        rsvar_diag, rsvar_regimes = run_rsvar(features)
        all_results["rs_var"] = rsvar_diag
        rsvar_out = PROCESSED_DIR / "rsvar"
        rsvar_out.mkdir(parents=True, exist_ok=True)
        rsvar_regimes.to_parquet(rsvar_out / "regime_assignments.parquet")
        with open(rsvar_out / "diagnostics.json", "w") as f:
            json.dump(rsvar_diag, f, indent=2, default=str)
        print(f"  OK: model_type={rsvar_diag['model_type']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["rs_var"] = {"error": str(e)}

    # 3. BDL
    print("\n[3/4] Bayesian Deep Learning...")
    try:
        bdl_diag, _ = run_bdl(features)
        if bdl_diag:
            all_results["bdl"] = bdl_diag
            bdl_out = PROCESSED_DIR / "bdl"
            bdl_out.mkdir(parents=True, exist_ok=True)
            with open(bdl_out / "diagnostics.json", "w") as f:
                json.dump(bdl_diag, f, indent=2, default=str)
            print(f"  OK: MC Dropout val_acc={bdl_diag['mc_dropout']['val_accuracy']:.3f}")
        else:
            print("  SKIPPED: TensorFlow not available")
            all_results["bdl"] = {"status": "skipped", "reason": "no tensorflow"}
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["bdl"] = {"error": str(e)}

    # 4. BOCPD
    print("\n[4/4] BOCPD Changepoint Detection...")
    try:
        bocpd_results = run_bocpd(features)
        all_results["bocpd"] = bocpd_results
        n_detected = sum(1 for v in bocpd_results.values() if v.get("detected", False))
        print(f"  OK: {n_detected}/{len(bocpd_results)} breakpoints detected")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["bocpd"] = {"error": str(e)}

    # Save combined results
    results_path = PROCESSED_DIR / "all_model_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ALL MODELS: COMPLETE")
    print(f"  Results saved to: {results_path}")
    for model, res in all_results.items():
        if isinstance(res, dict) and "error" not in res:
            print(f"  {model}: OK")
        elif isinstance(res, dict) and "error" in res:
            print(f"  {model}: FAILED ({res['error'][:50]})")
        else:
            print(f"  {model}: {res}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
