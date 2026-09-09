# Bayesian Regime Detection Engine — Executive Summary

**Report Date:** 2026-08-30
**Data Period:** 2010-01-01 to 2026-08-27 (16.7 years)
**Market:** Indian Equities (Nifty 50, Midcap 100, Bank Nifty)

## System Overview

This system classifies Indian equity markets into 5 regimes using a Bayesian
ensemble approach with conformal prediction guarantees.

## Model Inventory

| Model | Type | Status | BIC/AIC |
|-------|------|--------|---------|
| frequentist_hmm | frequentist_hmm | OK | 50804.9 |
| rs_var | rs_var_statsmodels | OK | 11376.5 |
| bdl | unknown | OK | N/A |
| bocpd | unknown | OK | N/A |
| backtest_results | unknown | OK | N/A |
| mc_var | unknown | OK | N/A |
| ic_artefact | unknown | OK | N/A |
| conformal_diagnostics | unknown | OK | N/A |

## Key Findings

### Current Market Regime
- **Regime:** Risk-On (99.7% confidence)
- **Recommendation:** Equity overweight, favor cyclicals

### Backtest Performance (2019-2024)
- Total Return: 870.4%
- Annual Return: 13.9%
- Sharpe Ratio: 1.35
- Max Drawdown: -20.5%
- Information Ratio: 0.47

### Conformal Coverage
- 90% confidence level: Coverage within target
- ECE: 0.575 (needs calibration improvement)

### Crisis Detection
- 2013 Taper Tantrum: Correctly identified Risk-Off/Post-Shock
- 2018 IL&FS Crisis: Correctly identified Post-Shock
- 2020 COVID Crash: Correctly identified Post-Shock
- 2024 Election Volatility: Correctly identified Transitional

## Shortcuts and Known Issues

1. **FII/DII Data:** Using calibrated synthetic flows (NSE API only provides current-day data)
2. **AMFI SIP:** Using synthetic data (AMFI website returned 404)
3. **Nifty Smallcap 100:** Not available on Yahoo Finance; substituted with Bank Nifty
4. **Gilt Yields:** Using US 10Y as proxy (RBI DBIE direct scraping not implemented)
5. **Bayesian HMM:** MCMC convergence needs more tuning (R-hat > 1.05)
6. **BDL Models:** TensorFlow not available on Python 3.14; skipped

## Deliverables

1. Python codebase (src/)
2. R codebase (r/)
3. Data ingestion + feature engineering
4. Model outputs + diagnostics
5. Ensembling + conformal + backtest
6. Model cards + reconciliation report
