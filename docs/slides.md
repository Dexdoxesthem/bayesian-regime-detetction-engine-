---
marp: true
theme: default
class: lead
---

# Bayesian Regime Detection Engine
*Institutional-Grade Equity Direction Forecasting*

---

# Why Regime Detection Matters
- **The Problem:** Price-prediction models fail at long horizons due to noise and non-stationarity.
- **The Solution:** Focus on *direction* over *price*. 
- **The Output:** A calibrated probability distribution over discrete market regimes (Risk-On, Risk-Off, Transitional, Late-Cycle).
- **The Goal:** Defensible, audit-ready tactical allocation tilts for Tier 1 Indian Asset Managers.

---

# System Architecture Overview
*The "Two-Speed" Precompute Architecture*

- **The Challenge:** Serverless environments (Render) lack the memory/compute to run PyMC MCMC sampling live.
- **The Batch Speed:** During the Docker `RUN` build, the system scrapes live Yahoo Finance data, engineers 30+ features, fits all Bayesian models, and caches output to `api_cache.json`.
- **The Online Speed:** At runtime, FastAPI boots instantly and serves the JSON payload in <20ms to the React dashboard.

---

# Model 1: Frequentist HMM
- **Role:** Fast, baseline maximum-likelihood state estimation.
- **Implementation:** `hmmlearn.GaussianHMM`
- **Specs:** 5 components, trained on 14 engineered features.
- **Drawback:** Lacks uncertainty quantification for transition matrices.

---

# Model 2: Bayesian HMM (PyMC)
- **Role:** Rigorous uncertainty tracking for regime transitions.
- **Implementation:** PyMC with NUTS (No-U-Turn Sampler).
- **Priors:** Dirichlet priors on transition matrices (favoring stickiness), Normal priors on emissions.
- **Key Output:** Full posterior credible intervals, allowing the Investment Committee to see exactly how confident the model is.

---

# Model 3: Deep Ensemble (Bayesian Deep Learning)
- **Role:** Capturing complex, non-linear cross-asset relationships.
- **Implementation:** 10 independent PyTorch feed-forward networks using Monte Carlo Dropout.
- **Uncertainty Decomposition:** 
  - *Epistemic* (model disagreement) measured by variance across the 10 networks.
  - *Aleatoric* (data noise) measured by mean softmax entropy.

---

# Model 4: Regime-Switching VAR
- **Role:** Modeling joint cross-asset dynamics.
- **Implementation:** `statsmodels` MarkovRegression.
- **Specs:** Captures the structural co-movement between Nifty, India VIX, FII flows, and Gilt yields differently depending on the active regime.

---

# Feature Engineering
*Beyond Simple Price Momentum (30+ Features)*

- **Macro & Flow:** FII/DII Net Z-scores, SIP momentum, INR volatility. 
- **Topological Data Analysis (TDA):** Spectral entropy and eigenvalues of rolling correlation matrices to detect hidden structural shifts.
- **Price & Trend:** DMA crosses, Garman-Klass volatility, market breadth dispersion.

---

# Deployment Architecture
- **Backend:** FastAPI (Python)
- **Frontend:** React + Vite + Tailwind CSS
- **Hosting:** Dockerized Web Service on Render
- **Design Aesthetic:** "Wall Street Journal" classic typography (Lato / Lora) and sophisticated navy/ivory palettes for institutional credibility.

---

# Live Dashboard
[INSERT SCREENSHOT: dashboard homepage]

*Features real-time Nifty 50, VIX, FII/DII flow metrics, and a dynamic Regime Probability distribution.*

---

# Engineering Case Study: Eliminating Look-Ahead Bias
- **The Problem:** The Deep Ensemble showed impossibly high out-of-sample accuracy during audit.
- **The Root Cause:** The target return labels were calculated using a backward-looking rolling mean (`rolling(21).mean()`), leaking future data into the training set.
- **The Fix:** Rewrote the pipeline to use a strict `shift(-21)` and masked the missing tail data.

---

# Engineering Case Study: The Topological Bottleneck
- **The Problem:** The pipeline was crashing during deployment due to extreme feature engineering times.
- **The Root Cause:** The TDA cross-sectional correlation feature used a brute-force `try/except` iteration over pandas rows.
- **The Fix:** Fully vectorized the logic using `unstack()` and optimized matrix multiplication. Time reduced from 4 minutes to <150ms.

---

# Engineering Case Study: Serverless Stale Data
- **The Problem:** The live API was serving frozen market data from August 2026.
- **The Root Cause:** The Docker container was building the `api_cache.json` using static `.parquet` files committed to GitHub.
- **The Fix:** Hooked the live `ingest_all()` scraping pipeline directly into the Docker `RUN` command to guarantee up-to-the-minute data on every deploy.

---

# Engineering Case Study: The Routing Collision
- **The Problem:** The frontend displayed an infinite loading screen in production, despite working locally.
- **The Root Cause:** Vercel deployment blocked the FastAPI backend, and the Vite proxy failed to route `/api` correctly on Render.
- **The Fix:** Consolidated both the React SPA and the FastAPI endpoints onto a single Render Web Service, perfectly resolving CORS and routing errors.

---

# Current Regime Analysis
[INSERT SCREENSHOT: Natural Language Insight Card]

- **Status:** The market is currently flagged as Late-Cycle / Transitional.
- **Drivers:** Driven by deteriorating market breadth and FII/DII flow divergence, despite Nifty holding above its 200-DMA.

---

# Limitations & Roadmap
*What's Next for V2?*
- **Foundation Models:** Integrating `Chronos` or `TimesFM` as embedding-extractors (deferred due to extreme GPU constraints).
- **Online Sequential Inference:** Implementing particle filters for tick-level intraday tracking.
- **Backtesting & Overlay:** Upgrading the naive backtester to a full `vectorbt` implementation with Regime-Conditioned Monte Carlo VaR simulations.

---

# Thank You / Q&A
*Repository, Codebase, and 40-Page Main Report available upon request.*
