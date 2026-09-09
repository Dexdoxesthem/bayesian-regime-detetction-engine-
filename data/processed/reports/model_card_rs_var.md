# Model Card: rs_var

**Generated:** 2026-08-30 16:26:55
**Version:** 1.0.0
**Status:** Operational

## 1. Model Description

| Field | Value |
|-------|-------|
| Model Type | rs_var_statsmodels |
| Task | Regime Classification (2 states) |
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

| model_type | rs_var_statsmodels |
| n_regimes | 2 |
| n_features | 1 |
| n_samples | 4326 |
| endog_names | ['nifty50_return_1d'] |
| aic | 11331.864616599021 |
| bic | 11376.469788534583 |
| log_likelihood | -5658.932308299511 |
| converged | True |
| regime_params | {'regime_0': 0.9816232886139277, 'regime_1': 0.074906640705376} |
| transition_matrix | None |
| regime_pct_time | {'regime_0': 0.8208092485549133, 'regime_1': 0.1791907514450867} |

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
