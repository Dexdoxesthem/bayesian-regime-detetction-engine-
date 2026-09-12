# Deliverable 5: Model Card & Calibration Pack

## 1. Frequentist HMM (Baseline)
- **Model Name:** 5-State Gaussian HMM
- **File Location:** `src/models/hmm/frequentist.py`
- **Library / Version:** `hmmlearn >= 0.3`
- **Inputs:** 14 features (Nifty returns, Midcap returns, Bank Nifty returns, realized volatility, VIX levels, FII/DII flows, Gilt yields, etc.)
- **Training Data Range:** 2010 - Present
- **Known Limitations:** The frequentist HMM is prone to getting stuck in local optima during Expectation-Maximization. It does not natively provide credible intervals on transition probabilities, making uncertainty quantification brittle.
- **Reliability Diagram:** [TO INSERT: run reliability diagram and paste result]

## 2. Bayesian HMM (MCMC)
- **Model Name:** PyMC Bayesian HMM with NUTS Sampler
- **File Location:** `src/models/hmm/bayesian.py`
- **Library / Version:** `pymc >= 5.10`, `arviz >= 0.17`
- **Inputs:** 8 principal components / scaled features (dimensionality reduced for MCMC tractability).
- **Priors Used:**
  - `start_probs`: Dirichlet prior `pm.Dirichlet('start_probs', a=np.ones(n_states))`
  - `trans_mat`: Dirichlet priors favoring self-transition (stickiness) `pm.Dirichlet('trans_mat', a=a_trans)`
  - `means`: Normal prior `pm.Normal('means', mu=0, sigma=2)`
  - `sigmas`: HalfNormal prior `pm.HalfNormal('sigmas', sigma=2)`
- **MCMC Diagnostics (from latest fast execution):**
  - **R-hat (max):** 2.9825
  - **ESS bulk (min):** 2
  - **Divergences:** 995
  *Note: These diagnostics are from an accelerated test run (50 tune steps) which caused the chains to fail convergence. Production runs require a minimum of 1000 tune steps and 2000 draws to resolve these divergences and bring R-hat < 1.01.*
- **Training Data Range:** 2010 - Present

## 3. Bayesian Deep Learning (Deep Ensemble)
- **Model Name:** Deep Ensemble (Variational / MC Dropout)
- **File Location:** `src/models/bdl/model.py`
- **Library / Version:** `torch`, `torch.nn`
- **Inputs:** Full 30+ engineered feature set, including TDA spectral entropy and cross-asset correlations.
- **Ensemble Size:** 10 Independent PyTorch Networks
- **Uncertainty Decomposition Method:** 
  - The model isolates *epistemic uncertainty* by measuring the variance (disagreement) *across* the 10 ensemble members.
  - *Aleatoric uncertainty* is captured via the mean entropy of the individual softmax outputs.
- **Known Limitations:** Deep learning models are inherently data-hungry. With only ~4,000 daily observations in Indian equities since 2010, the BDL is prone to overfitting. We mitigated this during the "Development Journey" by fixing a severe look-ahead bias in the target labels (using a strict `shift(-21)` rather than a backward-looking rolling mean).

## 4. Regime-Switching VAR (RS-VAR)
- **Model Name:** Multivariate Regime-Switching Vector Autoregression
- **File Location:** `src/models/rsvar/model.py`
- **Library / Version:** `statsmodels >= 0.14` (MarkovRegression baseline)
- **Inputs:** Nifty Returns, Volatility (VIX), Breadth, FII Flows, INR/USD.
- **Training Data Range:** 2010 - Present
- **Known Limitations:** The RS-VAR model captures cross-asset dynamics beautifully but struggles computationally when the feature space expands beyond 5-6 dimensions due to the explosion of the covariance matrix parameters. 

---

## Cross-Model Validation & Ensembling
The final serving layer (`api/main.py` and `api/precompute.py`) does not rely on any single model. Instead, it aggregates the outputs from the Frequentist HMM, the BDL Ensemble, and the RS-VAR. If the Deep Ensemble exhibits extremely high epistemic uncertainty (i.e., the networks violently disagree), the system defaults weight toward the more rigid, structurally sound Bayesian HMM. This creates a "fail-safe" where opaque neural networks are gated by classical statistical models.
