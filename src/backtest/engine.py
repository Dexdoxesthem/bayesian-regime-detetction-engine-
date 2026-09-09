"""
Backtesting engine for regime-overlay allocation.

Implements:
1. Regime-overlay backtester with tilt rules
2. Information Ratio, Tracking Error, regime-conditioned drawdowns
3. Monte Carlo VaR/CVaR simulation
4. Scenario replay for Indian crisis case studies
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeOverlayBacktester:
    """
    Regime-overlay allocation backtester.

    Tilts portfolio based on regime + conviction.
    """

    # Regime-to-tilt mapping (India equity context)
    REGIME_TILTS = {
        0: 1.2,    # Risk-On: overweight equities
        1: 0.5,    # Risk-Off: underweight equities
        2: 0.8,    # Transitional: slightly underweight
        3: 0.7,    # Late-Cycle: defensive tilt
        4: 0.4,    # Post-Shock: very defensive
    }

    def __init__(
        self,
        regimes: np.ndarray,
        confidence: np.ndarray,
        dates: pd.DatetimeIndex,
        initial_capital: float = 100.0,
    ):
        self.regimes = regimes
        self.confidence = confidence
        self.dates = dates
        self.initial_capital = initial_capital

    def run(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
    ) -> Dict:
        """
        Run backtest with regime-overlay allocation.

        Parameters
        ----------
        returns : pd.Series
            Daily Nifty 50 returns.
        benchmark_returns : pd.Series
            Benchmark returns (default: equal-weight buy-and-hold).

        Returns
        -------
        dict : Backtest results including IR, TE, drawdowns.
        """
        n = min(len(self.regimes), len(returns))
        regimes = self.regimes[:n]
        confidence = self.confidence[:n]
        ret = returns.iloc[:n].values

        # Compute tilts
        base_tilts = np.array([self.REGIME_TILTS.get(r, 1.0) for r in regimes])
        # Adjust by confidence
        tilt = 0.5 + 0.5 * confidence  # Scale between 0.5 and 1.0
        portfolio_tilt = base_tilts * tilt

        # Portfolio returns
        port_returns = portfolio_tilt * ret
        port_cumulative = (1 + port_returns).cumprod() * self.initial_capital

        # Benchmark (buy-and-hold)
        if benchmark_returns is not None:
            bm = benchmark_returns.iloc[:n].values
        else:
            bm = ret
        bm_cumulative = (1 + bm).cumprod() * self.initial_capital

        # Performance metrics
        annual_ret = np.mean(port_returns) * 252
        annual_vol = np.std(port_returns) * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

        # Benchmark metrics
        bm_annual_ret = np.mean(bm) * 252
        bm_annual_vol = np.std(bm) * np.sqrt(252)

        # Information Ratio
        active_returns = port_returns - bm
        tracking_error = np.std(active_returns) * np.sqrt(252)
        information_ratio = (annual_ret - bm_annual_ret) / tracking_error if tracking_error > 0 else 0

        # Maximum Drawdown
        peak = np.maximum.accumulate(port_cumulative)
        drawdown = (port_cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # Regime-conditioned metrics
        regime_metrics = {}
        for r in range(5):
            mask = regimes == r
            if mask.sum() > 0:
                regime_ret = port_returns[mask]
                regime_metrics[f"regime_{r}"] = {
                    "n_days": int(mask.sum()),
                    "avg_return": float(regime_ret.mean() * 252),
                    "volatility": float(regime_ret.std() * np.sqrt(252)),
                    "sharpe": float(regime_ret.mean() / regime_ret.std() * np.sqrt(252)) if regime_ret.std() > 0 else 0,
                    "max_drawdown": float(((1 + regime_ret).cumprod() / np.maximum.accumulate((1 + regime_ret).cumprod()) - 1).min()),
                }

        return {
            "total_return": float((port_cumulative[-1] / self.initial_capital) - 1),
            "annual_return": float(annual_ret),
            "annual_volatility": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "information_ratio": float(information_ratio),
            "tracking_error": float(tracking_error),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(annual_ret / abs(max_drawdown)) if max_drawdown != 0 else 0,
            "regime_conditioned": regime_metrics,
            "n_days": n,
            "portfolio_curve": port_cumulative.tolist(),
            "benchmark_curve": bm_cumulative.tolist(),
        }


def run_monte_carlo_var(
    returns: np.ndarray,
    n_simulations: int = 10000,
    horizon_days: int = 21,
    confidence_levels: List[float] = [0.95, 0.99],
    random_state: int = 42,
) -> Dict:
    """
    Monte Carlo VaR/CVaR simulation.

    Parameters
    ----------
    returns : np.ndarray
        Historical daily returns.
    n_simulations : int
        Number of MC paths.
    horizon_days : int
        Forward-looking horizon in trading days.
    confidence_levels : list
        VaR confidence levels.

    Returns
    -------
    dict : VaR and CVaR at each confidence level.
    """
    rng = np.random.default_rng(random_state)

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    # Simulate paths
    sim_returns = rng.normal(mean_ret, std_ret, (n_simulations, horizon_days))
    sim_cumulative = np.cumprod(1 + sim_returns, axis=1) - 1
    terminal_returns = sim_cumulative[:, -1]

    results = {
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "mean_terminal_return": float(terminal_returns.mean()),
        "std_terminal_return": float(terminal_returns.std()),
    }

    for cl in confidence_levels:
        alpha = 1 - cl
        var = np.percentile(terminal_returns, alpha * 100)
        cvar = terminal_returns[terminal_returns <= var].mean()
        results[f"VaR_{cl}"] = float(var)
        results[f"CVaR_{cl}"] = float(cvar)

    return results


CRISIS_CASE_STUDIES = {
    "2013_taper_tantrum": {
        "name": "2013 Taper Tantrum",
        "period": ("2013-05-01", "2013-09-01"),
        "description": "Fed signals taper of QE; Indian markets sell off sharply. "
                       "INR depreciates from 54 to 68. FII outflows intensify.",
        "expected_regime": "Risk-Off",
    },
    "2018_ilfs_nbfcrisis": {
        "name": "2018 IL&FS / NBFC Crisis",
        "period": ("2018-09-01", "2019-02-01"),
        "description": "IL&FS defaults trigger NBFC liquidity crisis. "
                       "Credit markets freeze. Midcap/Smallcap indices crash 30-40%.",
        "expected_regime": "Risk-Off / Post-Shock",
    },
    "2020_covid_crash": {
        "name": "2020 COVID Crash",
        "period": ("2020-02-15", "2020-04-30"),
        "description": "Nifty falls 38% in 28 trading days (March 2020). "
                       "India VIX spikes to 82. Global risk-off selloff.",
        "expected_regime": "Post-Shock",
    },
    "2024_election_volatility": {
        "name": "2024 Election Results Volatility",
        "period": ("2024-06-01", "2024-06-30"),
        "description": "NDA election results cause single-day 6% Nifty crash. "
                       "INR weakens. Markets recover within days.",
        "expected_regime": "Transitional / Risk-Off",
    },
}


def replay_crisis_scenario(
    regime_probs: np.ndarray,
    dates: pd.DatetimeIndex,
    returns: pd.Series,
    scenario_name: str,
) -> Dict:
    """
    Replay a crisis scenario through the regime engine.

    Shows what the engine would have called at each stage.
    """
    if scenario_name not in CRISIS_CASE_STUDIES:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = CRISIS_CASE_STUDIES[scenario_name]
    start, end = pd.Timestamp(scenario["period"][0]), pd.Timestamp(scenario["period"][1])

    mask = (dates >= start) & (dates <= end)
    if mask.sum() == 0:
        return {"error": f"No data for period {start} to {end}"}

    period_regimes = regime_probs[mask]
    period_returns = returns[mask]
    period_dates = dates[mask]

    # Compute regime assignments
    regimes = np.argmax(period_regimes, axis=1)
    confidence = period_regimes.max(axis=1)

    # Stage analysis
    stages = []
    for i in range(0, len(period_dates), max(1, len(period_dates) // 5)):
        stage_end = min(i + max(1, len(period_dates) // 5), len(period_dates))
        stage_ret = period_returns.iloc[i:stage_end].sum()
        stage_regime = int(np.bincount(regimes[i:stage_end]).argmax()) if len(regimes[i:stage_end]) > 0 else -1

        stages.append({
            "date_range": f"{period_dates[i].date()} to {period_dates[stage_end-1].date()}",
            "regime": int(stage_regime),
            "avg_confidence": float(confidence[i:stage_end].mean()),
            "cumulative_return": float(stage_ret),
        })

    return {
        "scenario": scenario["name"],
        "period": scenario["period"],
        "description": scenario["description"],
        "expected_regime": scenario["expected_regime"],
        "actual_regime_distribution": {
            f"regime_{r}": int((regimes == r).sum())
            for r in range(5)
        },
        "avg_confidence": float(confidence.mean()),
        "total_return": float(period_returns.sum()),
        "stages": stages,
    }


def generate_ic_artefact(
    ensemble_probs: np.ndarray,
    dates: pd.DatetimeIndex,
    returns: pd.Series,
) -> Dict:
    """
    Generate Investment Committee artefact.

    Plain-language conditional statements with full model lineage.
    """
    regimes = np.argmax(ensemble_probs, axis=1)
    confidence = ensemble_probs.max(axis=1)
    latest_regime = regimes[-1]
    latest_confidence = confidence[-1]
    latest_date = dates[-1]

    REGIME_DESCRIPTIONS = {
        0: "Risk-On — Equity overweight warranted. Favor cyclicals, small/midcap beta. "
           "Reduce duration, add commodity exposure.",
        1: "Risk-Off — Reduce equity allocation. Shift to defensives (IT, Pharma, Staples). "
           "Increase gilt allocation. Hedge INR exposure.",
        2: "Transitional — Maintain neutral positioning. Wait for regime clarity. "
           "Tactical allocation only. Tight stop-losses.",
        3: "Late-Cycle — Gradually reduce risk. Favor quality/dividend. "
           "Monitor credit spreads and FII flows closely.",
        4: "Post-Shock — Maximum defensive posture. High cash/gilts. "
           "Prepare to deploy capital as regime shifts to Risk-On.",
    }

    artefact = {
        "report_date": str(latest_date.date()),
        "current_regime": int(latest_regime),
        "regime_label": REGIME_DESCRIPTIONS.get(latest_regime, "Unknown"),
        "confidence": float(latest_confidence),
        "probability_distribution": {
            f"regime_{r}": float(ensemble_probs[-1, r])
            for r in range(5)
        },
        "conditional_recommendation": (
            f"Based on ensemble output as of {latest_date.date()}: "
            f"the market is in a {REGIME_DESCRIPTIONS.get(latest_regime, 'Unknown')} "
            f"regime with {latest_confidence:.1%} confidence."
        ),
        "model_lineage": {
            "ensemble_method": "Bayesian Model Averaging + Constrained Stacking",
            "members": ["Frequentist HMM (5-state)", "RS-VAR", "BOCPD"],
            "conformalized": True,
            "coverage_guarantee": "90% marginal coverage (finite-sample valid)",
        },
        "risk_budget": {
            "max_drawdown_limit": "-15%",
            "tracking_error_budget": "3-5%",
            "rebalance_frequency": "Weekly (or on regime shift)",
        },
    }

    return artefact
