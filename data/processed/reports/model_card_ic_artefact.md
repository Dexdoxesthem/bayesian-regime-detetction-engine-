# Model Card: ic_artefact

**Generated:** 2026-08-30 16:26:55
**Version:** 1.0.0
**Status:** Operational

## 1. Model Description

| Field | Value |
|-------|-------|
| Model Type | N/A |
| Task | Regime Classification (5 states) |
| Training Data | Nifty 50, Midcap, Bank Nifty, India VIX, USDINR, FII/DII, SIP, Gilt |
| Date Range | 2010-01-01 to 2026-08-27 |

## 2. Inputs

| Feature | Description |
|---------|-------------|
| Nifty 50 returns (1d, 21d, 63d) | Price momentum at multiple horizons |
| Realized volatility (21d, 63d) | Risk regime proxy |
| India VIX | Implied volatility |
| Breadth (DMA crossover) | Market participation |
| FII/DII z-scores | Institutional flow dynamics |
| INR momentum | Currency regime |
| Gilt yield changes | Interest rate regime |

## 3. Priors (Bayesian models)

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| Transition matrix | Dirichlet(alpha=0.5 diagonal, 0.5 off-diagonal) | Encourages moderate persistence |
| Start probs | Dirichlet(alpha=1) | Uninformative |
| Emission means | Normal(0, 1) | Scaled features centered at 0 |
| Emission std | HalfNormal(1) | Weakly informative |

## 4. Diagnostics

| report_date | 2026-08-28 |
| current_regime | 0 |
| regime_label | Risk-On — Equity overweight warranted. Favor cyclicals, small/midcap beta. Reduce duration, add commodity exposure. |
| confidence | 0.9965593262664305 |
| probability_distribution | {'regime_0': 0.9965593262664305, 'regime_1': 0.0034406737335694925, 'regime_2': 0.0, 'regime_3': 0.0, 'regime_4': 0.0} |
| conditional_recommendation | Based on ensemble output as of 2026-08-28: the market is in a Risk-On — Equity overweight warranted. Favor cyclicals, small/midcap beta. Reduce duration, add commodity exposure. regime with 99.7% confidence. |
| model_lineage | {'ensemble_method': 'Bayesian Model Averaging + Constrained Stacking', 'members': ['Frequentist HMM (5-state)', 'RS-VAR', 'BOCPD'], 'conformalized': True, 'coverage_guarantee': '90% marginal coverage (finite-sample valid)'} |
| risk_budget | {'max_drawdown_limit': '-15%', 'tracking_error_budget': '3-5%', 'rebalance_frequency': 'Weekly (or on regime shift)'} |

## 5. Regime Definitions

| State | Label | Interpretation |
|-------|-------|----------------|
| 0 | Risk-On | Equity overweight, favor cyclicals, reduce duration |
| 1 | Risk-Off | Reduce equity, shift to defensives, increase gilt allocation |
| 2 | Transitional | Neutral positioning, wait for regime clarity |
| 3 | Late-Cycle | Gradually reduce risk, favor quality/dividend |
| 4 | Post-Shock | Maximum defensive, high cash/gilts, prepare to deploy |

## 6. Known Limitations

- Bayesian HMM MCMC diagnostics may not meet convergence thresholds on all runs
- BOCPD hazard rate requires tuning for Indian market dynamics
- FII/DII historical data uses calibrated synthetic flows (NSE API provides only current-day)
- AMFI SIP data uses synthetic fallback (AMFI website returned 404)

## 7. Intended Use

- **Primary:** Daily regime classification for Indian equity allocation overlay
- **Secondary:** Risk management, stress testing, scenario analysis
- **NOT suitable for:** Individual stock selection, high-frequency trading

## 8. Ethical Considerations

- Model outputs are probabilistic; decisions should incorporate human judgment
- Regime labels are heuristic; semantic meaning may not perfectly align with market reality
- Past performance does not guarantee future results
