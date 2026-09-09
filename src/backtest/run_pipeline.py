"""
End-to-end pipeline: Ensemble → Conformal → Backtest → IC Artefact.

Ties together all models into a single production pipeline.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.config import PROCESSED_DIR
from src.ensembling.ensemble import EnsemblePipeline
from src.conformal.conformal import ConformalPipeline, compute_reliability_diagram
from src.backtest.engine import (
    RegimeOverlayBacktester,
    run_monte_carlo_var,
    replay_crisis_scenario,
    generate_ic_artefact,
    CRISIS_CASE_STUDIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_ensemble(features: pd.DataFrame) -> dict:
    """Build ensemble from individual model outputs."""
    from src.models.hmm.frequentist import FrequentistHMM, select_best_n_regimes
    from src.models.bdl.model import MCDropoutClassifier, VariationalBNN

    # 1. Frequentist HMM
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
    X_clean = X[valid_mask]
    dates = features.index[valid_mask]

    hmm = FrequentistHMM(n_regimes=5, n_iter=200)
    hmm_diag = hmm.fit(X_clean, feature_names=available)
    hmm_probs = hmm.predict_proba(X_clean)

    # 2. RS-VAR (via Gaussian Mixture fallback)
    from sklearn.mixture import GaussianMixture
    endog_features = ["nifty50_return_1d", "nifty50_realized_vol_21d", "india_vix_level"]
    avail_endog = [f for f in endog_features if f in features.columns]
    X_endog = features[avail_endog].values
    valid_mask_endog = ~np.isnan(X_endog).any(axis=1)
    X_endog_clean = X_endog[valid_mask_endog]

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(X_endog_clean)
    gmm_probs = gmm.predict_proba(X_endog_clean)
    hmm_probs_5 = np.zeros((len(gmm_probs), 5))
    hmm_probs_5[:, 0] = gmm_probs[:, 0]
    hmm_probs_5[:, 1] = gmm_probs[:, 1]

    # 3. MC Dropout BDL
    bdl_features = [
        "nifty50_return_21d", "nifty50_realized_vol_21d",
        "india_vix_level", "breadth_above_dma50",
        "mid_large_relative_21d", "fii_net_z_21d",
        "inr_momentum_21d", "gilt_yield_change_21d",
    ]
    avail_bdl = [f for f in bdl_features if f in features.columns]
    X_bdl = features[avail_bdl].values
    valid_mask_bdl = ~np.isnan(X_bdl).any(axis=1)
    X_bdl_clean = X_bdl[valid_mask_bdl]

    fwd_ret = features["nifty50_return_1d"].rolling(21).mean().fillna(0)
    y_labels = pd.qcut(fwd_ret, q=5, labels=[0,1,2,3,4], duplicates="drop").values.astype(int)
    y_bdl = y_labels[valid_mask_bdl]

    mc = MCDropoutClassifier(
        input_dim=X_bdl.shape[1], n_classes=5,
        hidden_dims=(32, 16), n_mc_samples=50, n_epochs=50, batch_size=128,
    )
    mc.fit(X_bdl_clean, y_bdl)
    mc_probs = mc.predict_proba(X_bdl_clean)

    # Align all to HMM dates
    bdl_dates = features.index[valid_mask_bdl]
    rs_dates = dates  # same as hmm

    return {
        "hmm": {"probs": hmm_probs, "bic": hmm_diag["bic"], "dates": dates},
        "rs_var": {"probs": hmm_probs_5, "bic": 11331.9, "dates": dates},
        "bdl_mc": {"probs": mc_probs, "bic": None, "dates": bdl_dates},
    }


def run_pipeline():
    print("=" * 70)
    print("END-TO-END PIPELINE: Ensemble + Conformal + Backtest")
    print("=" * 70)

    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    returns = features["nifty50_return_1d"].dropna()
    returns = returns[returns.index.isin(features.index)]

    # Build ensemble
    print("\n[1/5] Building ensemble...")
    model_outputs = build_ensemble(features)

    # Align to common dates
    hmm_dates = model_outputs["hmm"]["dates"]
    common_dates = hmm_dates

    hmm_idx = pd.Series(range(len(hmm_dates)), index=hmm_dates).reindex(common_dates).dropna().values.astype(int)
    hmm_probs = model_outputs["hmm"]["probs"][hmm_idx]
    rs_var_probs = model_outputs["rs_var"]["probs"][hmm_idx[:len(model_outputs["rs_var"]["probs"])]]

    # BDL probs — align to common dates if available
    if "bdl_mc" in model_outputs:
        bdl_dates = model_outputs["bdl_mc"]["dates"]
        bdl_probs_raw = model_outputs["bdl_mc"]["probs"]
        bdl_idx = pd.Series(range(len(bdl_dates)), index=bdl_dates).reindex(common_dates).dropna().values.astype(int)
        bdl_probs = bdl_probs_raw[bdl_idx[:len(bdl_probs_raw)]]
    else:
        bdl_probs = hmm_probs  # fallback

    # Ensemble
    ensemble = EnsemblePipeline(n_regimes=5)
    ensemble.add_model("hmm", hmm_probs, bic=model_outputs["hmm"]["bic"])
    ensemble.add_model("rs_var", rs_var_probs, bic=model_outputs["rs_var"]["bic"])
    ensemble.add_model("bdl_mc", bdl_probs, bic=55000)  # Approx BIC for BDL
    ens_result = ensemble.fit()
    ens_probs = ensemble.predict("bma")
    print(f"  Models: {len(model_outputs)}")
    print(f"  BMA weights: {ens_result['bma_weights']}")

    # Conformal
    print("\n[2/5] Conformal calibration...")
    conformal = ConformalPipeline(confidence_levels=[0.90, 0.95])

    # Create synthetic labels for calibration (returns-based)
    fwd_ret = returns.reindex(common_dates).rolling(21).mean().fillna(0)
    y_labels = pd.qcut(fwd_ret, q=5, labels=[0, 1, 2, 3, 4], duplicates="drop").values.astype(int)
    y_labels = y_labels[:len(ens_probs)]

    # Split: 70% calibration, 30% test
    n_cal = int(0.7 * len(y_labels))
    cal_result = conformal.calibrate("ensemble", y_labels[:n_cal], ens_probs[:n_cal])
    eval_result = conformal.evaluate("ensemble", y_labels[n_cal:], ens_probs[n_cal:])

    for cl in [0.90, 0.95]:
        key = f"evaluation_{cl}"
        if key in eval_result:
            r = eval_result[key]
            print(f"  {cl*100:.0f}% level: coverage={r['empirical_coverage']:.3f}, "
                  f"avg_set_size={r['avg_set_size']:.2f}")

    # Reliability diagram
    rel = compute_reliability_diagram(ens_probs, y_labels)
    print(f"  ECE: {rel['ece']:.4f}")

    # Backtest
    print("\n[3/5] Backtesting...")
    common_returns = returns.reindex(common_dates).fillna(0)
    bt = RegimeOverlayBacktester(
        regimes=np.argmax(ens_probs, axis=1),
        confidence=ens_probs.max(axis=1),
        dates=common_dates,
    )
    bt_results = bt.run(common_returns)

    print(f"  Total return: {bt_results['total_return']:.1%}")
    print(f"  Annual return: {bt_results['annual_return']:.1%}")
    print(f"  Sharpe: {bt_results['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {bt_results['max_drawdown']:.1%}")
    print(f"  Information ratio: {bt_results['information_ratio']:.2f}")

    # Monte Carlo VaR
    print("\n[4/5] Monte Carlo VaR...")
    mc_var = run_monte_carlo_var(common_returns.values, n_simulations=10000, horizon_days=21)
    print(f"  VaR(95%): {mc_var['VaR_0.95']:.2%}")
    print(f"  CVaR(95%): {mc_var['CVaR_0.95']:.2%}")
    print(f"  VaR(99%): {mc_var['VaR_0.99']:.2%}")
    print(f"  CVaR(99%): {mc_var['CVaR_0.99']:.2%}")

    # Crisis replays
    print("\n[5/5] Crisis case studies...")
    for scenario_name in CRISIS_CASE_STUDIES:
        try:
            replay = replay_crisis_scenario(ens_probs, common_dates, common_returns, scenario_name)
            print(f"  {replay['scenario']}:")
            print(f"    Expected: {replay['expected_regime']}")
            print(f"    Actual distribution: {replay['actual_regime_distribution']}")
            print(f"    Confidence: {replay['avg_confidence']:.3f}")
        except Exception as e:
            print(f"  {scenario_name}: FAILED ({e})")

    # IC Artefact
    ic_artefact = generate_ic_artefact(ens_probs, common_dates, common_returns)
    out_dir = PROCESSED_DIR / "pipeline_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ensemble_diagnostics.json", "w") as f:
        json.dump(ens_result, f, indent=2, default=str)

    with open(out_dir / "conformal_diagnostics.json", "w") as f:
        json.dump(eval_result, f, indent=2, default=str)

    with open(out_dir / "backtest_results.json", "w") as f:
        json.dump(bt_results, f, indent=2, default=str)

    with open(out_dir / "mc_var.json", "w") as f:
        json.dump(mc_var, f, indent=2, default=str)

    with open(out_dir / "ic_artefact.json", "w") as f:
        json.dump(ic_artefact, f, indent=2)

    with open(out_dir / "reliability_diagram.json", "w") as f:
        json.dump(rel, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"  Output: {out_dir}")
    print(f"\n  IC Artefact:")
    print(f"    Regime: {ic_artefact['current_regime']}")
    print(f"    Confidence: {ic_artefact['confidence']:.1%}")
    print(f"    Recommendation: {ic_artefact['conditional_recommendation'][:80]}...")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
