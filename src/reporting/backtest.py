import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from api.main import _load_features, _get_models, REGIME_LABELS

def compute_drawdown(returns: pd.Series) -> pd.Series:
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    return (cum_ret - peak) / peak

def run_backtest():
    print("Loading data for backtesting...")
    features = _load_features()
    models = _get_models(features)
    
    dates = pd.to_datetime(models["dates"])
    probs = np.array(models["bdl_probs"]) # Using Deep Ensemble probabilities
    
    merged = pd.read_parquet("data/processed/merged_market_data.parquet")
    nifty_rets = merged["nifty50_close"].pct_change()
    
    # Target Nifty returns 
    # Shift(-1) because today's regime call informs tomorrow's return
    nifty_rets = nifty_rets.reindex(dates).shift(-1).fillna(0)
    
    # Simple Regime Allocation Logic:
    # Risk-On (0), Late-Cycle (1), Transitional (2), Post-Shock (3), Risk-Off (4)
    # We create a continuous equity exposure scalar based on regime probabilities
    
    # 1.0 exposure for Risk-On/Post-Shock
    # 0.8 for Late-Cycle
    # 0.5 for Transitional
    # 0.2 for Risk-Off (move to cash)
    exposure_map = np.array([1.0, 0.8, 0.5, 1.0, 0.2])
    
    # Expected exposure is the dot product of probabilities and the exposure map
    target_exposure = np.dot(probs, exposure_map)
    
    # Apply turnover limits (hysteresis) to prevent whipsaw
    # Smooth the target exposure over 5 days
    target_exposure = pd.Series(target_exposure).rolling(5, min_periods=1).mean().values
    
    # Calculate strategy returns
    strategy_rets = nifty_rets * target_exposure
    
    # Backtest Metrics
    def calc_metrics(rets):
        ann_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        max_dd = compute_drawdown(rets).min()
        return ann_ret, ann_vol, sharpe, max_dd

    bmk_ret, bmk_vol, bmk_sharpe, bmk_dd = calc_metrics(nifty_rets)
    str_ret, str_vol, str_sharpe, str_dd = calc_metrics(strategy_rets)
    
    # Tracking Error & Info Ratio
    active_return = strategy_rets - nifty_rets
    tracking_error = active_return.std() * np.sqrt(252)
    info_ratio = (str_ret - bmk_ret) / tracking_error if tracking_error > 0 else 0
    
    print("\n=========================================")
    print("REGIME OVERLAY BACKTEST (2010 - PRESENT)")
    print("=========================================")
    print(f"Benchmark (100% Nifty):")
    print(f"  Return:     {bmk_ret*100:.2f}%")
    print(f"  Volatility: {bmk_vol*100:.2f}%")
    print(f"  Sharpe:     {bmk_sharpe:.2f}")
    print(f"  Max DD:     {bmk_dd*100:.2f}%")
    print("\nRegime-Conditioned Strategy:")
    print(f"  Return:     {str_ret*100:.2f}%")
    print(f"  Volatility: {str_vol*100:.2f}%")
    print(f"  Sharpe:     {str_sharpe:.2f}")
    print(f"  Max DD:     {str_dd*100:.2f}%")
    print(f"\n  Information Ratio: {info_ratio:.2f}")
    print(f"  Tracking Error:    {tracking_error*100:.2f}%")
    print("=========================================")
    
    # Save artefact
    results = {
        "benchmark": {"return": bmk_ret, "volatility": bmk_vol, "sharpe": bmk_sharpe, "max_drawdown": bmk_dd},
        "strategy": {"return": str_ret, "volatility": str_vol, "sharpe": str_sharpe, "max_drawdown": str_dd},
        "information_ratio": info_ratio,
        "tracking_error": tracking_error
    }
    
    out_path = Path("data/processed/reports/backtest_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Backtest results saved to {out_path}")

if __name__ == "__main__":
    run_backtest()
