#!/usr/bin/env Rscript
# =============================================================================
# Bayesian Regime Detection Engine — R Codebase
# Parallel implementations: HMM, Markov-switching, changepoint detection
# =============================================================================

library(depmixS4)
library(changepoint)
library(MSwM)

cat("=" , rep("=", 69), "\n", sep="")
cat("R CODEBASE — Regime Detection Models\n")
cat("=" , rep("=", 69), "\n\n")

# ---------------------------------------------------------------------------
# 1. Load data from Python output
# ---------------------------------------------------------------------------

data_dir <- file.path("data", "processed")
features_file <- file.path(data_dir, "features.parquet")

if (!file.exists(features_file)) {
    cat("FATAL: Run Python ingestion first.\n")
    quit(status = 1)
}

# Read parquet via arrow or reticulate
tryCatch({
    library(arrow)
    features <- read_parquet(features_file)
}, error = function(e) {
    cat("arrow not available, using reticulate fallback\n")
    library(reticulate)
    pd <- import("pandas")
    df <- pd$read_parquet(features_file)
    features <<- as.data.frame(df)
})

cat("Loaded features:", nrow(features), "rows,", ncol(features), "columns\n\n")

# ---------------------------------------------------------------------------
# 2. Frequentist HMM (depmixS4)
# ---------------------------------------------------------------------------

cat("[1/3] Frequentist HMM via depmixS4...\n")

# Select core features
core_features <- c("nifty50_return_1d", "nifty50_realized_vol_21d",
                   "india_vix_level", "breadth_above_dma50")
core_features <- core_features[core_features %in% names(features)]

if (length(core_features) < 2) {
    cat("  Insufficient features for HMM, using first 2 columns\n")
    core_features <- names(features)[1:2]
}

# Remove NAs for HMM
hmm_data <- features[, core_features]
complete_rows <- complete.cases(hmm_data)
hmm_data <- hmm_data[complete_rows, ]

if (nrow(hmm_data) > 1000) {
    # Subsample for speed
    idx <- seq(1, nrow(hmm_data), length.out = 1000)
    hmm_data <- hmm_data[idx, ]
}

cat("  Data for HMM:", nrow(hmm_data), "rows,", ncol(hmm_data), "features\n")

# Fit HMM with 5 states
set.seed(42)
tryCatch({
    hmm_model <- depmix(
        as.formula(paste(core_features, collapse = " ~ ")),
        data = hmm_data,
        nstates = 5,
        family = gaussian()
    )
    hmm_fit <- fit(hmm_model)

    # Extract posterior state probabilities
    posterior <- posterior(hmm_fit)
    states <- posterior$state

    cat("  HMM fitted: 5 states\n")
    cat("  State distribution:\n")
    for (s in 1:5) {
        pct <- mean(states == s) * 100
        cat(sprintf("    State %d: %.1f%%\n", s, pct))
    }
}, error = function(e) {
    cat("  HMM failed:", conditionMessage(e), "\n")
})

# ---------------------------------------------------------------------------
# 3. Markov-Switching Model (MSwM)
# ---------------------------------------------------------------------------

cat("\n[2/3] Markov-Switching Model via MSwM...\n")

tryCatch({
    # Use Nifty returns for MSwM
    ms_data <- data.frame(
        returns = hmm_data[, 1],
        vol = hmm_data[, min(2, ncol(hmm_data))]
    )

    # Fit Markov-switching linear model
    ms_model <- lm(returns ~ 1, data = ms_data)
    ms_fit <- MSwM::msmFit(ms_model, k = 2, sw = c(TRUE, FALSE))

    cat("  MSwM fitted: 2-regime switching model\n")
    cat("  Regime means:\n")
    cat(sprintf("    Regime 1: %.4f\n", ms_fit@Coef$`(Intercept)_1`))
    cat(sprintf("    Regime 2: %.4f\n", ms_fit@Coef$`(Intercept)_2`))

    # Smoothed regime probabilities
    ms_probs <- ms_fit@posterior[, 2]
    ms_regimes <- ifelse(ms_probs > 0.5, 2, 1)
    cat(sprintf("  Regime 1: %.1f%%, Regime 2: %.1f%%\n",
                mean(ms_regimes == 1) * 100, mean(ms_regimes == 2) * 100))
}, error = function(e) {
    cat("  MSwM failed:", conditionMessage(e), "\n")
})

# ---------------------------------------------------------------------------
# 4. Changepoint Detection (changepoint package)
# ---------------------------------------------------------------------------

cat("\n[3/3] Changepoint Detection via changepoint package...\n")

tryCatch({
    returns <- hmm_data[, 1]

    # Binary segmentation with PELT
    cp_result <- cpt.meanvar(returns, method = "PELT", penalty = "BIC")
    changepoints <- cpts(cp_result)

    cat("  Detected changepoints:", length(changepoints), "\n")
    if (length(changepoints) > 0) {
        cat("  First 10 changepoints:", head(changepoints, 10), "\n")
    }
}, error = function(e) {
    cat("  Changepoint detection failed:", conditionMessage(e), "\n")
})

# ---------------------------------------------------------------------------
# 5. Save R outputs for reconciliation with Python
# ---------------------------------------------------------------------------

cat("\nSaving R outputs for cross-language reconciliation...\n")

output_dir <- file.path(data_dir, "r_output")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Save HMM states
if (exists("states")) {
    write.csv(data.frame(state = states),
              file.path(output_dir, "hmm_states.csv"),
              row.names = FALSE)
}

# Save MSwM regimes
if (exists("ms_regimes")) {
    write.csv(data.frame(regime = ms_regimes, prob = ms_probs),
              file.path(output_dir, "mswm_regimes.csv"),
              row.names = FALSE)
}

# Save changepoints
if (exists("changepoints")) {
    write.csv(data.frame(changepoint = changepoints),
              file.path(output_dir, "changepoints.csv"),
              row.names = FALSE)
}

cat("\n" , rep("=", 70), "\n", sep="")
cat("R CODEBASE: COMPLETE\n")
cat("Outputs saved to:", output_dir, "\n")
cat(rep("=", 70), "\n", sep="")
