"""
Feature engineering module for Indian equity regime detection.

Produces 30+ features across six categories:
1. Returns / Trend / Volatility / Breadth
2. Cap-segment relative features
3. Flow features (FII/DII, SIP)
4. Macro features (RBI stance proxy, real rates, INR)
5. Topological features (persistence landscapes)
6. Cross-sectional features

All features are computed on a rolling basis with strict time-respecting
windows to prevent look-ahead bias.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Returns / Trend / Volatility / Breadth
# ---------------------------------------------------------------------------

def compute_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute return-based features for all price series.

    Features:
    - nifty50_return_1d, _5d, _21d, _63d, _126d, _252d
    - nifty50_log_return_1d
    - nifty_midcap_return_1d, _21d
    - nifty_bank_return_1d, _21d
    """
    features = pd.DataFrame(index=df.index)

    for prefix in ["nifty50", "nifty_midcap", "nifty_bank"]:
        close = df[f"{prefix}_close"]
        # Multiple horizons
        for days in [1, 5, 21, 63, 126, 252]:
            if prefix == "nifty50":  # All horizons for primary index
                features[f"{prefix}_return_{days}d"] = close.pct_change(days)
            elif days in [1, 21]:  # Fewer horizons for secondary indices
                features[f"{prefix}_return_{days}d"] = close.pct_change(days)

    # Log returns for Nifty50
    features["nifty50_log_return_1d"] = np.log(
        df["nifty50_close"] / df["nifty50_close"].shift(1)
    )

    return features


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trend-following features.

    Features:
    - nifty50_dma_{50,200} and crossover signals
    - nifty50_trend_strength (ADX-like)
    - nifty50_price_vs_dma_{50,200}
    - nifty50_consecutive_up/down days
    """
    features = pd.DataFrame(index=df.index)
    close = df["nifty50_close"]

    # Moving averages
    for w in [50, 200]:
        dma = close.rolling(w, min_periods=w).mean()
        features[f"nifty50_dma_{w}"] = dma
        features[f"nifty50_price_vs_dma_{w}"] = (close - dma) / dma

    # Golden/death cross signal
    dma50 = close.rolling(50, min_periods=50).mean()
    dma200 = close.rolling(200, min_periods=200).mean()
    features["nifty50_golden_cross"] = (dma50 > dma200).astype(float)
    features["nifty50_cross_signal"] = (dma50 - dma200) / dma200

    # Exponential moving averages
    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    features["nifty50_macd"] = (ema12 - ema26) / close

    # Trend strength: directional movement index proxy
    high = df["nifty50_high"]
    low = df["nifty50_low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    features["nifty50_atr_ratio"] = atr14 / close

    # Consecutive up/down days
    daily_return = close.pct_change()
    up = (daily_return > 0).astype(int)
    down = (daily_return < 0).astype(int)

    # Count consecutive ups
    up_streak = up.copy()
    for i in range(1, len(up_streak)):
        if up.iloc[i] == 1:
            up_streak.iloc[i] = up_streak.iloc[i - 1] + 1
        else:
            up_streak.iloc[i] = 0
    features["nifty50_consecutive_up"] = up_streak

    # Count consecutive downs
    down_streak = down.copy()
    for i in range(1, len(down_streak)):
        if down.iloc[i] == 1:
            down_streak.iloc[i] = down_streak.iloc[i - 1] + 1
        else:
            down_streak.iloc[i] = 0
    features["nifty50_consecutive_down"] = down_streak

    return features


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility-based features.

    Features:
    - nifty50_realized_vol_{5,21,63}d
    - nifty50_vol_of_vol
    - nifty50_garman_klass_vol
    - india_vix_level, india_vix_change_5d
    - vix_term_structure (if available)
    - nifty50_skewness_21d, kurtosis_21d
    """
    features = pd.DataFrame(index=df.index)

    # Log returns for vol computation
    log_ret = np.log(df["nifty50_close"] / df["nifty50_close"].shift(1))

    # Realized volatility (annualized)
    for w in [5, 21, 63]:
        features[f"nifty50_realized_vol_{w}d"] = (
            log_ret.rolling(w, min_periods=w).std() * np.sqrt(252)
        )

    # Vol of vol
    rv21 = features["nifty50_realized_vol_21d"]
    features["nifty50_vol_of_vol"] = rv21.rolling(63, min_periods=21).std()

    # Garman-Klass volatility estimator
    log_hl = np.log(df["nifty50_high"] / df["nifty50_low"])
    log_co = np.log(df["nifty50_close"] / df["nifty50_open"])
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    features["nifty50_garman_klass_vol"] = (
        gk.rolling(21, min_periods=14).mean().apply(lambda x: np.sqrt(max(x, 0))) * np.sqrt(252)
    )

    # VIX features
    if "india_vix_close" in df.columns:
        vix = df["india_vix_close"]
        features["india_vix_level"] = vix
        features["india_vix_change_5d"] = vix.pct_change(5)
        features["india_vix_z_21d"] = (
            (vix - vix.rolling(21, min_periods=14).mean())
            / vix.rolling(21, min_periods=14).std().replace(0, np.nan)
        )

    # Higher moments
    features["nifty50_skewness_21d"] = log_ret.rolling(21, min_periods=21).skew()
    features["nifty50_kurtosis_21d"] = log_ret.rolling(21, min_periods=21).kurt()

    return features


def compute_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market breadth proxy features.

    Since we don't have advance/decline data, use cross-index breadth:
    - Proportion of indices trading above their 50-DMA
    - Cross-index correlation
    - Relative strength between cap segments
    """
    features = pd.DataFrame(index=df.index)

    # Compute 50-DMA for each index
    indices = {}
    for prefix in ["nifty50", "nifty_midcap", "nifty_bank"]:
        close = df[f"{prefix}_close"]
        dma50 = close.rolling(50, min_periods=50).mean()
        indices[prefix] = (close > dma50).astype(float)

    # Breadth: proportion above 50-DMA
    breadth_df = pd.DataFrame(indices)
    features["breadth_above_dma50"] = breadth_df.mean(axis=1)

    # All-above / all-below signals
    features["breadth_all_above"] = breadth_df.all(axis=1).astype(float)
    features["breadth_all_below"] = (~breadth_df.any(axis=1)).astype(float)

    # Cross-index correlation (rolling 63-day)
    returns = pd.DataFrame({
        prefix: df[f"{prefix}_close"].pct_change()
        for prefix in ["nifty50", "nifty_midcap", "nifty_bank"]
    })
    # Average pairwise correlation
    features["cross_index_corr_63d"] = returns.rolling(63, min_periods=42).corr().groupby(level=0).mean().mean(axis=1)
    # Fix the index alignment
    rolling_corr = returns.rolling(63, min_periods=42).corr()
    avg_corr = []
    for date in returns.index:
        try:
            sub = rolling_corr.loc[date]
            if isinstance(sub, pd.DataFrame) and sub.shape == (3, 3):
                # Average off-diagonal
                mask = np.ones(sub.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_corr.append(sub.values[mask].mean())
            else:
                avg_corr.append(np.nan)
        except:
            avg_corr.append(np.nan)
    features["cross_index_corr_63d"] = avg_corr

    return features


# ---------------------------------------------------------------------------
# 2. Cap-segment relative features
# ---------------------------------------------------------------------------

def compute_cap_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cap-segment relative performance features.

    Features:
    - mid_large_relative_21d (midcap vs largecap performance)
    - bank_nifty_relative_21d
    - midcap_vol_ratio (midcap vol / largecap vol)
    - breadth_dispersion
    """
    features = pd.DataFrame(index=df.index)

    # Relative performance
    nifty_ret = df["nifty50_close"].pct_change(21)
    midcap_ret = df["nifty_midcap_close"].pct_change(21)
    bank_ret = df["nifty_bank_close"].pct_change(21)

    features["mid_large_relative_21d"] = midcap_ret - nifty_ret
    features["bank_nifty_relative_21d"] = bank_ret - nifty_ret

    # Volatility ratio (21-day)
    nifty_vol = np.log(df["nifty50_close"] / df["nifty50_close"].shift(1)).rolling(21).std()
    midcap_vol = np.log(df["nifty_midcap_close"] / df["nifty_midcap_close"].shift(1)).rolling(21).std()
    features["midcap_vol_ratio"] = midcap_vol / nifty_vol.replace(0, np.nan)

    # Breadth dispersion: std of cross-index 21d returns
    cross_ret = pd.DataFrame({
        "nifty50": nifty_ret,
        "midcap": midcap_ret,
        "bank": bank_ret,
    })
    features["breadth_dispersion"] = cross_ret.std(axis=1)

    return features


# ---------------------------------------------------------------------------
# 3. Flow features
# ---------------------------------------------------------------------------

def compute_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fund flow features.

    Features:
    - fii_net_z_21d, fii_net_z_63d (z-scores of FII net flows)
    - dii_net_z_21d, dii_net_z_63d
    - flow_balance_ratio (DII net / (FII abs net + DII abs net))
    - sip_momentum_3m, sip_momentum_6m
    - fii_dii_divergence (FII and DII moving in opposite directions)
    """
    features = pd.DataFrame(index=df.index)

    # FII z-scores
    if "fii_net" in df.columns:
        fii = df["fii_net"]
        for w in [21, 63]:
            mean = fii.rolling(w, min_periods=w // 2).mean()
            std = fii.rolling(w, min_periods=w // 2).std().replace(0, np.nan)
            features[f"fii_net_z_{w}d"] = (fii - mean) / std

        # Cumulative FII flow
        features["fii_cumulative_21d"] = fii.rolling(21, min_periods=14).sum()

    # DII z-scores
    if "dii_net" in df.columns:
        dii = df["dii_net"]
        for w in [21, 63]:
            mean = dii.rolling(w, min_periods=w // 2).mean()
            std = dii.rolling(w, min_periods=w // 2).std().replace(0, np.nan)
            features[f"dii_net_z_{w}d"] = (dii - mean) / std

    # Flow balance ratio: DII absorption of FII flows
    if "fii_net" in df.columns and "dii_net" in df.columns:
        fii_abs = df["fii_net"].abs()
        dii_abs = df["dii_net"].abs()
        total = fii_abs + dii_abs
        features["flow_balance_ratio"] = dii_abs / total.replace(0, np.nan)

        # FII-DII divergence: opposite signs
        features["fii_dii_divergence"] = (
            (df["fii_net"] > 0) != (df["dii_net"] > 0)
        ).astype(float)

        # Rolling correlation of FII and DII
        features["fii_dii_corr_63d"] = (
            df["fii_net"].rolling(63, min_periods=21).corr(df["dii_net"])
        )

    # SIP momentum
    if "sip_inflow" in df.columns:
        sip = df["sip_inflow"]
        features["sip_momentum_3m"] = sip.pct_change(3)  # 3-month pct change
        features["sip_momentum_6m"] = sip.pct_change(6)  # 6-month pct change
        features["sip_z_12m"] = (
            (sip - sip.rolling(12, min_periods=6).mean())
            / sip.rolling(12, min_periods=6).std().replace(0, np.nan)
        )

    return features


# ---------------------------------------------------------------------------
# 4. Macro features
# ---------------------------------------------------------------------------

def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute macro-economic proxy features.

    Features:
    - real_rate_proxy (gilt yield - inflation proxy)
    - inr_trend_21d, inr_volatility_21d
    - gilt_yield_change_21d
    - yield_curve_slope (if multiple tenors available)
    - inr_momentum (INR strengthening = risk-on for India)
    """
    features = pd.DataFrame(index=df.index)

    # Gilt yield features
    if "gilt_10y" in df.columns:
        gilt = df["gilt_10y"]
        features["gilt_yield_level"] = gilt
        features["gilt_yield_change_21d"] = gilt.diff(21)
        features["gilt_yield_change_63d"] = gilt.diff(63)
        features["gilt_yield_z_63d"] = (
            (gilt - gilt.rolling(63, min_periods=21).mean())
            / gilt.rolling(63, min_periods=21).std().replace(0, np.nan)
        )

        # Real rate proxy: use gilt yield - 4% (rough CPI assumption)
        features["real_rate_proxy"] = gilt - 4.0

        # RBI stance proxy: rising yields = tightening
        features["rbi_stance_proxy"] = gilt.diff(5).rolling(21).mean()

    # AAA-Gilt spread
    if "aaa_gilt_spread" in df.columns:
        features["aaa_gilt_spread"] = df["aaa_gilt_spread"]
        features["spread_change_21d"] = df["aaa_gilt_spread"].diff(21)

    # INR features
    if "usdinr_close" in df.columns:
        inr = df["usdinr_close"]
        features["inr_level"] = inr
        features["inr_change_21d"] = inr.pct_change(21)
        features["inr_volatility_21d"] = (
            np.log(inr / inr.shift(1)).rolling(21, min_periods=14).std() * np.sqrt(252)
        )
        # INR strengthening = lower USDINR = risk-on
        features["inr_momentum_21d"] = -inr.pct_change(21)

    return features


# ---------------------------------------------------------------------------
# 5. Topological features (persistence landscapes)
# ---------------------------------------------------------------------------

def compute_topological_features(
    df: pd.DataFrame,
    window: int = 63,
    step: int = 21,
) -> pd.DataFrame:
    """
    Compute topological data analysis features.

    Uses rolling correlation matrices and computes persistence-based
    summary statistics as proxies for regime-dependent correlation
    structure changes.

    Features:
    - topo_corr_eigenvalue_1, _2 (top eigenvalues of rolling corr matrix)
    - topo_corr_entropy (spectral entropy of correlation matrix)
    - topo_corr_stability (determinant of correlation matrix)
    - topo_corr_condition_number

    NOTE: Full persistence landscapes via gtda require additional
    dependencies. These spectral summaries are computationally cheaper
    and capture similar information about correlation regime changes.
    """
    features = pd.DataFrame(index=df.index)

    # Build returns matrix
    returns = pd.DataFrame({
        prefix: np.log(df[f"{prefix}_close"] / df[f"{prefix}_close"].shift(1))
        for prefix in ["nifty50", "nifty_midcap", "nifty_bank"]
        if f"{prefix}_close" in df.columns
    })

    # Add VIX returns if available
    if "india_vix_close" in df.columns:
        returns["india_vix"] = np.log(
            df["india_vix_close"] / df["india_vix_close"].shift(1)
        )

    # Add USDINR returns if available
    if "usdinr_close" in df.columns:
        returns["usdinr"] = np.log(
            df["usdinr_close"] / df["usdinr_close"].shift(1)
        )

    n_assets = returns.shape[1]
    dates = returns.index

    eigenvalue_1 = np.full(len(dates), np.nan)
    eigenvalue_2 = np.full(len(dates), np.nan)
    spectral_entropy = np.full(len(dates), np.nan)
    corr_determinant = np.full(len(dates), np.nan)
    condition_number = np.full(len(dates), np.nan)

    for i in range(window, len(dates)):
        window_data = returns.iloc[i - window:i].dropna()
        if window_data.shape[0] < window // 2 or window_data.shape[1] < 2:
            continue

        corr = window_data.corr().values
        # Handle NaN in correlation matrix
        if np.any(np.isnan(corr)):
            continue

        try:
            eigenvalues = np.linalg.eigvalsh(corr)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

            eigenvalue_1[i] = eigenvalues[0]
            if len(eigenvalues) > 1:
                eigenvalue_2[i] = eigenvalues[1]

            # Spectral entropy
            eigs_pos = eigenvalues[eigenvalues > 0]
            if len(eigs_pos) > 0:
                p = eigs_pos / eigs_pos.sum()
                spectral_entropy[i] = -np.sum(p * np.log(p + 1e-10))

            # Determinant
            corr_determinant[i] = np.linalg.det(corr)

            # Condition number
            condition_number[i] = eigenvalues[0] / max(eigenvalues[-1], 1e-10)

        except np.linalg.LinAlgError:
            continue

    features["topo_corr_eigenvalue_1"] = eigenvalue_1
    features["topo_corr_eigenvalue_2"] = eigenvalue_2
    features["topo_spectral_entropy"] = spectral_entropy
    features["topo_corr_determinant"] = corr_determinant
    features["topo_corr_condition_number"] = condition_number

    return features


# ---------------------------------------------------------------------------
# 6. Cross-sectional and interaction features
# ---------------------------------------------------------------------------

def compute_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional interaction features.

    Features:
    - risk_on_composite (VIX down + INR up + breadth up)
    - risk_off_composite (VIX up + INR down + breadth down)
    - flow_momentum_interaction
    - vol_flow_interaction
    """
    features = pd.DataFrame(index=df.index)

    # Risk-on composite
    vix_down = 0
    inr_up = 0
    breadth_up = 0

    if "india_vix_close" in df.columns:
        vix_z = (
            df["india_vix_close"] - df["india_vix_close"].rolling(21).mean()
        ) / df["india_vix_close"].rolling(21).std().replace(0, np.nan)
        vix_down = (-vix_z).clip(lower=0) / 3  # Normalized

    if "usdinr_close" in df.columns:
        inr_chg = -df["usdinr_close"].pct_change(21)  # Negative = strengthening
        inr_up = inr_chg.clip(lower=0) / 0.03  # Normalize by ~3% move

    # Breadth from earlier computation
    nifty50_above = (
        df["nifty50_close"] > df["nifty50_close"].rolling(50).mean()
    ).astype(float)
    midcap_above = (
        df["nifty_midcap_close"] > df["nifty_midcap_close"].rolling(50).mean()
    ).astype(float)
    breadth_up = (nifty50_above + midcap_above) / 2

    features["risk_on_composite"] = (vix_down + inr_up + breadth_up) / 3
    features["risk_off_composite"] = 1 - features["risk_on_composite"]

    return features


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_all_features(
    df: pd.DataFrame,
    include_topo: bool = True,
    topo_window: int = 63,
) -> pd.DataFrame:
    """
    Build all feature sets and combine into a single DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged market data from ingestion.
    include_topo : bool
        Whether to compute topological features (slow).
    topo_window : int
        Window for topological features.

    Returns
    -------
    pd.DataFrame
        All features with DatetimeIndex.
    """
    logger.info("Building feature sets...")

    feature_builders = [
        ("returns", compute_returns_features),
        ("trend", compute_trend_features),
        ("volatility", compute_volatility_features),
        ("breadth", compute_breadth_features),
        ("cap_segment", compute_cap_segment_features),
        ("flow", compute_flow_features),
        ("macro", compute_macro_features),
        ("cross", compute_cross_features),
    ]

    all_features = {}

    for name, builder in feature_builders:
        try:
            feats = builder(df)
            all_features[name] = feats
            logger.info("  %s: %d features", name, len(feats.columns))
        except Exception as e:
            logger.error("  %s failed: %s", name, e)

    # Topological features (computationally expensive)
    if include_topo:
        try:
            topo_feats = compute_topological_features(df, window=topo_window)
            all_features["topological"] = topo_feats
            logger.info("  topological: %d features", len(topo_feats.columns))
        except Exception as e:
            logger.error("  topological failed: %s", e)

    # Combine all features
    combined = pd.concat(all_features.values(), axis=1)
    combined.index = df.index
    combined.index.name = "date"

    logger.info("Total features: %d", len(combined.columns))

    # Feature quality report
    missing_pct = combined.isna().mean()
    high_missing = missing_pct[missing_pct > 0.1]
    if len(high_missing) > 0:
        logger.warning("Features with >10%% missing:")
        for col, pct in high_missing.items():
            logger.warning("  %s: %.1f%%", col, pct * 100)

    return combined


def get_feature_descriptions() -> Dict[str, str]:
    """Return human-readable descriptions of all features."""
    return {
        # Returns
        "nifty50_return_1d": "Nifty 50 1-day return",
        "nifty50_return_5d": "Nifty 50 5-day return",
        "nifty50_return_21d": "Nifty 50 21-day return",
        "nifty50_return_63d": "Nifty 50 63-day return",
        "nifty50_return_126d": "Nifty 50 126-day return",
        "nifty50_return_252d": "Nifty 50 252-day return",
        "nifty50_log_return_1d": "Nifty 50 log 1-day return",
        # Trend
        "nifty50_dma_50": "Nifty 50 50-day moving average",
        "nifty50_dma_200": "Nifty 50 200-day moving average",
        "nifty50_price_vs_dma_50": "Nifty 50 price relative to 50-DMA",
        "nifty50_price_vs_dma_200": "Nifty 50 price relative to 200-DMA",
        "nifty50_golden_cross": "Golden cross signal (50-DMA > 200-DMA)",
        "nifty50_cross_signal": "50/200 DMA crossover magnitude",
        "nifty50_macd": "MACD signal (12/26 EMA)",
        "nifty50_atr_ratio": "ATR(14)/Price ratio",
        # Volatility
        "nifty50_realized_vol_5d": "Nifty 50 5-day realized volatility",
        "nifty50_realized_vol_21d": "Nifty 50 21-day realized volatility",
        "nifty50_realized_vol_63d": "Nifty 50 63-day realized volatility",
        "nifty50_vol_of_vol": "Volatility of volatility (63d rolling)",
        "nifty50_garman_klass_vol": "Garman-Klass volatility estimator",
        "india_vix_level": "India VIX level",
        "india_vix_change_5d": "India VIX 5-day change",
        "india_vix_z_21d": "India VIX z-score (21d)",
        "nifty50_skewness_21d": "Nifty 50 return skewness (21d)",
        "nifty50_kurtosis_21d": "Nifty 50 return kurtosis (21d)",
        # Breadth
        "breadth_above_dma50": "Cross-index breadth: % above 50-DMA",
        "breadth_all_above": "All indices above 50-DMA",
        "breadth_all_below": "All indices below 50-DMA",
        "cross_index_corr_63d": "Cross-index correlation (63d)",
        # Cap segment
        "mid_large_relative_21d": "Midcap vs Largecap relative 21d return",
        "bank_nifty_relative_21d": "Bank Nifty vs Nifty relative 21d return",
        "midcap_vol_ratio": "Midcap/Largecap volatility ratio",
        "breadth_dispersion": "Cross-index return dispersion",
        # Flow
        "fii_net_z_21d": "FII net flow z-score (21d)",
        "fii_net_z_63d": "FII net flow z-score (63d)",
        "dii_net_z_21d": "DII net flow z-score (21d)",
        "dii_net_z_63d": "DII net flow z-score (63d)",
        "flow_balance_ratio": "DII absorption ratio",
        "fii_dii_divergence": "FII-DII directional divergence",
        "fii_dii_corr_63d": "FII-DII correlation (63d)",
        "sip_momentum_3m": "SIP inflow momentum (3-month)",
        "sip_momentum_6m": "SIP inflow momentum (6-month)",
        # Macro
        "gilt_yield_level": "10Y Gilt yield level",
        "gilt_yield_change_21d": "Gilt yield 21-day change",
        "gilt_yield_change_63d": "Gilt yield 63-day change",
        "gilt_yield_z_63d": "Gilt yield z-score (63d)",
        "real_rate_proxy": "Real interest rate proxy",
        "rbi_stance_proxy": "RBI monetary stance proxy",
        "inr_change_21d": "USDINR 21-day change",
        "inr_volatility_21d": "INR realized volatility (21d)",
        "inr_momentum_21d": "INR momentum (negative = strengthening)",
        # Topological
        "topo_corr_eigenvalue_1": "Top eigenvalue of rolling correlation matrix",
        "topo_corr_eigenvalue_2": "Second eigenvalue of rolling correlation matrix",
        "topo_spectral_entropy": "Spectral entropy of correlation matrix",
        "topo_corr_determinant": "Determinant of rolling correlation matrix",
        "topo_corr_condition_number": "Condition number of correlation matrix",
        # Cross-sectional
        "risk_on_composite": "Composite risk-on indicator",
        "risk_off_composite": "Composite risk-off indicator",
    }
