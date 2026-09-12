# Deliverable 6: Demo Video Script

**[0:00 - 1:00] Intro & What the System Is**
*(Screen recording starts. Presenter is looking at the camera or voice-over starts while hovering on the dashboard homepage).*
"Hello, and welcome to our demonstration of the Bayesian Regime Detection Engine. Traditional quant models in the Indian mutual fund industry try to predict exact price levels for the Nifty 50—and they fail, because long-horizon equity data is noisy and non-stationary. Instead of asking 'What will the price be?', our engine asks 'What regime are we in, and how confident are we in that classification?' We've built an institutional-grade, multi-model ensemble that outputs calibrated probabilities over discrete market states: Risk-On, Risk-Off, Transitional, Late-Cycle, and Post-Shock."

**[1:00 - 3:00] Architecture Overview**
*(Switch screen to show the codebase in VS Code or the Render deployment dashboard).*
"Before we look at the UI, let's talk about the architecture. We faced a massive challenge: serverless environments like Render don't have the compute to run heavy PyMC Markov Chain Monte Carlo sampling or PyTorch deep ensembles live. So, we designed a 'Two-Speed' Precompute Architecture. During the Docker build phase, the system triggers a heavy batch job. It scrapes live Yahoo Finance data, computes over 30 macro and topological features, trains the Bayesian models, and caches the final probabilities into a static JSON artefact. When you hit the live site, our FastAPI backend simply serves that precomputed JSON, dropping our API latency to under 20 milliseconds."

**[3:00 - 6:00] Live Dashboard Walkthrough**
*(Switch back to the live React dashboard. Move the mouse to point out different elements as you speak).*
"Let's look at the live dashboard. We've designed this with a classic, professional typography—using Lato—to fit right into an institutional Investment Committee deck. 

On the right, you can see the live metrics driving our model: the Nifty 50 level, India VIX, Market Breadth, and critical institutional flow dynamics—FII and DII Net buying. 

On the left is our primary deliverable: the Regime Probabilities chart. The model is currently flagging the market as [READ CURRENT REGIME OFF SCREEN]. Notice the natural language 'Insight' card at the bottom. The engine isn't a black box; it explicitly tells us that it is [READ CONFIDENCE PERCENTAGE] confident in this call, and it highlights that this classification is driven primarily by recent changes in [READ EXPLANATION OFF SCREEN, e.g., 'institutional flow dynamics']. This is exactly the kind of transparent, audit-defensible output that a Chief Investment Officer needs to size a cash buffer or tilt a portfolio."

**[6:00 - 8:00] Engineering Challenge & Validation**
*(Keep screen on the dashboard, perhaps showing the Model Suite view or back to VS Code).*
"Building this wasn't without hurdles. The most critical issue we caught during our rigorous validation was a 'Look-Ahead Bias' in our Deep Ensemble. During an early audit, our model accuracy looked suspiciously high. We dug into the feature engineering pipeline and found that the target forward-return labels were being generated using a backward-looking rolling mean. The model was effectively peeking into the future. We completely rewrote the labeling logic to use a strict, forward-looking shift, and properly masked the missing data tails. Accuracy dropped to realistic levels, but more importantly, our backtesting Information Ratio became mathematically honest and defensible."

**[8:00 - 9:30] Limitations & What's Next**
*(Switch to a slide or simply talk to the camera).*
"While this version 1 engine is highly robust, we are completely transparent about its limitations. First, due to GPU constraints, we deferred the integration of time-series Foundation Models like Chronos. Second, our backtester currently shows that while our strategy cuts maximum drawdowns by over 15%, it suffers from cash-drag during structural bull markets. In Version 2, we will integrate a `vectorbt` backtesting engine to simulate regime-conditioned leverage, and we'll implement a Particle Filter for tick-level online inference."

**[9:30 - 10:00] Close**
"Thank you for watching. This engine closes the gap between theoretical Bayesian statistics and a deployable, fiduciary-grade quantitative product. We invite you to review our full 40-page report and our open-source codebase for deeper technical details."
