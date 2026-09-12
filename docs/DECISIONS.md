# Architecture & Modeling Decision Log

This document records the major design, engineering, and modeling decisions made during the development of the Bayesian Regime Detection Engine.

## 1. System Architecture Decisions

### 1.1 The "Two-Speed" Precompute Architecture
* **Decision:** Decoupled model training (batch) from the API serving (online) by running all heavy PyMC/PyTorch training during the Docker image build process (`api/precompute.py`).
* **Rationale:** Serverless hosting environments (like Render and Vercel) have strict memory limits and CPU timeouts. Running MCMC sampling on a live API endpoint would guarantee timeouts (HTTP 504). By precomputing probabilities and saving them to `api_cache.json`, the live FastAPI server achieves sub-20ms latency.
* **Trade-off:** The system cannot react to intraday tick data. It only updates when a new build/deployment is triggered. 

### 1.2 Unified Render Deployment
* **Decision:** Migrated both the React frontend and the FastAPI backend to a single Render Web Service rather than splitting them across Vercel and Render.
* **Rationale:** The Vercel deployment suffered from severe CORS issues and infinite loading loops because it could not natively run the Python backend. Serving the React static files directly through FastAPI via `app.mount()` ensures perfect routing and zero cross-origin friction.

## 2. Modeling Decisions

### 2.1 Multi-Model Ensemble (No Single Black Box)
* **Decision:** Implemented a combination of Frequentist HMM, Bayesian HMM, Deep Ensembles (BDL), and RS-VAR, rather than relying on a single mega-model.
* **Rationale:** Financial time series are highly non-stationary. Neural networks (Deep Ensembles) capture complex non-linearities but are prone to overfitting and opaque failures. Classical Bayesian HMMs are rigid but provide mathematically defensible uncertainty bounds. Fusing them provides a "fail-safe" where classical statistics gate AI outputs.

### 2.2 Deep Ensemble for Bayesian Deep Learning (BDL)
* **Decision:** Used a Deep Ensemble (training 10 separate neural networks) instead of heavier Bayesian Neural Networks (e.g., using Stochastic Weight Averaging or Bayes-by-Backprop).
* **Rationale:** Deep Ensembles are easier to train, trivially parallelizable, and have been empirically proven to provide better calibrated epistemic uncertainty (model disagreement) in out-of-distribution financial data than traditional variational inference.

### 2.3 Dropping Foundation Models (Chronos / TimesFM)
* **Decision:** Deferred the integration of time-series foundation models to future work.
* **Rationale:** HuggingFace transformers require significant GPU VRAM for inference. Our mandate was to build a deployable application on free-tier/CPU infrastructure. Attempting to run Chronos on a Render CPU would have crippled the pipeline.

## 3. Data & Feature Engineering Decisions

### 3.1 Synthetic Flow Data Stubbing
* **Decision:** Used synthetic, calibrated generators for FII/DII Net Flows and Mutual Fund SIP data in `src/data/loader.py`.
* **Rationale:** While Nifty price data is freely available via `yfinance`, true institutional flow data requires expensive APIs (Bloomberg, NSE tick data). To prove the system architecture worked, we stubbed these critical features. This is a known technical debt that must be replaced before production use.

### 3.2 Topological Data Analysis (TDA)
* **Decision:** Engineered spectral entropy and correlation eigenvalues.
* **Rationale:** Traditional momentum features lag the market. By analyzing the structural shape of the market (how closely all stocks move together), TDA provides a leading indicator of regime shifts (e.g., correlations converging to 1.0 right before a crash).

### 3.3 Fixing Look-Ahead Bias
* **Decision:** Switched target labels from a rolling mean (`rolling(21).mean()`) to a strict forward shift (`shift(-21)`).
* **Rationale:** The rolling mean leaked future data into the current day's training row, artificially inflating out-of-sample accuracy. The shift enforces true forecasting discipline.

## 4. Frontend & UX Decisions

### 4.1 "Wall Street Journal" Aesthetic
* **Decision:** Transitioned from a dark-mode developer UI to a light-themed, ivory/navy color palette using classic serif fonts (Lato/Lora).
* **Rationale:** The target audience for this engine is Investment Committees and Chief Investment Officers. The UI needs to convey institutional trust, rigor, and clarity, moving away from "crypto-trader" dark themes to professional financial reporting aesthetics.

### 4.2 Natural Language "Insight" Cards
* **Decision:** Added a gradient card that translates raw probabilities into plain English (e.g., "Model confidence is 82%, driven primarily by institutional flow dynamics").
* **Rationale:** Portfolio Managers don't want to decipher raw log-likelihoods. They need immediate, actionable narratives summarizing *why* the model made its decision.
