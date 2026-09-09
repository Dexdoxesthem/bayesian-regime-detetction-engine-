"""FastAPI backend for Regime Engine dashboard."""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

from src.ingestion.config import PROCESSED_DIR, REGIME_LABELS, REGIME_BREAKPOINTS

REGIME_COLORS = {0: '#22c55e', 1: '#ef4444', 2: '#f59e0b', 3: '#a855f7', 4: '#06b6d4'}
BREAKPOINTS = REGIME_BREAKPOINTS

app = FastAPI(title="Regime Engine API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_CACHE = {}


def _load_features():
    return pd.read_parquet(PROCESSED_DIR / "features.parquet")


def _load_merged():
    return pd.read_parquet(PROCESSED_DIR / "merged_market_data.parquet")


def _run_models(features):
    from src.models.hmm.frequentist import FrequentistHMM
    from src.models.bdl.model import MCDropoutClassifier, VariationalBNN, DeepEnsemble
    from sklearn.mixture import GaussianMixture

    key_features = [
        "nifty50_return_21d", "nifty50_return_63d",
        "nifty50_realized_vol_21d", "nifty50_realized_vol_63d",
        "india_vix_level", "india_vix_change_5d",
        "breadth_above_dma50", "cross_index_corr_63d",
        "mid_large_relative_21d", "fii_net_z_21d", "dii_net_z_21d",
        "inr_momentum_21d", "gilt_yield_change_21d", "nifty50_golden_cross",
    ]
    available = [f for f in key_features if f in features.columns]
    X = features[available].values
    valid_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[valid_mask]
    dates = features.index[valid_mask]

    hmm = FrequentistHMM(n_regimes=5, n_iter=200)
    hmm.fit(X_clean, feature_names=available)
    hmm_probs = hmm.predict_proba(X_clean)

    # BDL
    bdl_features = [
        "nifty50_return_21d", "nifty50_realized_vol_21d",
        "india_vix_level", "breadth_above_dma50",
        "mid_large_relative_21d", "fii_net_z_21d",
        "inr_momentum_21d", "gilt_yield_change_21d",
    ]
    avail_bdl = [f for f in bdl_features if f in features.columns]
    X_bdl = features[avail_bdl].values
    valid_bdl = ~np.isnan(X_bdl).any(axis=1)
    X_bdl_clean = X_bdl[valid_bdl]
    bdl_dates = features.index[valid_bdl]

    fwd_ret = features["nifty50_return_1d"].rolling(21).mean().fillna(0)
    y_labels = pd.qcut(fwd_ret, q=5, labels=[0, 1, 2, 3, 4], duplicates="drop").values.astype(int)
    y_bdl = y_labels[valid_bdl]

    de = DeepEnsemble(M=3, input_dim=X_bdl.shape[1], n_classes=5, hidden_dims=(32, 16), n_epochs=50, batch_size=128)
    de.fit(X_bdl_clean, y_bdl)
    de_probs, de_epist = de.predict_with_uncertainty(X_bdl_clean)

    hmm_idx = pd.Series(range(len(dates)), index=dates)
    de_idx = pd.Series(range(len(bdl_dates)), index=bdl_dates)
    common = hmm_idx.index.intersection(de_idx.index)
    common_hmm = hmm_idx.reindex(common).dropna().values.astype(int)
    common_de = de_idx.reindex(common).dropna().values.astype(int)

    return {
        "dates": [str(d)[:10] for d in common],
        "hmm_probs": hmm_probs[common_hmm].tolist(),
        "bdl_probs": de_probs[common_de].tolist(),
        "bdl_epistemic": de_epist[common_de].tolist(),
    }


def _get_models(features):
    if "models" not in _CACHE:
        _CACHE["models"] = _run_models(features)
    return _CACHE["models"]


@app.get("/api/regime/current")
def regime_current():
    features = _load_features()
    models = _get_models(features)
    last_probs = np.array(models["hmm_probs"][-1])
    regime_idx = int(np.argmax(last_probs))
    confidence = float(last_probs[regime_idx])

    return {
        "current_regime": regime_idx,
        "regime_label": REGIME_LABELS[regime_idx],
        "regime_color": REGIME_COLORS[regime_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            REGIME_LABELS[i]: round(float(last_probs[i]), 4) for i in range(5)
        },
        "date": models["dates"][-1],
    }


@app.get("/api/regime/history")
def regime_history():
    features = _load_features()
    merged = _load_merged()
    models = _get_models(features)

    hmm_classes = [int(np.argmax(p)) for p in models["hmm_probs"]]
    bdl_classes = [int(np.argmax(p)) for p in models["bdl_probs"]]

    # Subsample to ~500 points for chart performance
    step = max(1, len(models["dates"]) // 500)
    indices = list(range(0, len(models["dates"]), step))
    sample_dates = pd.to_datetime([models["dates"][i] for i in indices])

    close_idx = merged["nifty50_close"].reindex(sample_dates).dropna()
    close_map = {str(d)[:10]: float(v) for d, v in close_idx.items()}
    nifty_series = [close_map.get(models["dates"][i]) for i in indices]

    return {
        "dates": [models["dates"][i] for i in indices],
        "nifty50_index": nifty_series,
        "hmm_regimes": [hmm_classes[i] for i in indices],
        "bdl_regimes": [bdl_classes[i] for i in indices],
        "hmm_probs": [models["hmm_probs"][i] for i in indices],
        "bdl_probs": [models["bdl_probs"][i] for i in indices],
    }


@app.get("/api/regime/crisis")
def regime_crisis():
    return {
        "crises": [
            {
                "name": "2008 Global Financial Crisis",
                "start": "2008-01-01",
                "end": "2009-03-31",
                "expected_regime": "Risk-Off / Post-Shock",
                "nifty50_peak": 6304,
                "nifty50_trough": 2525,
                "drawdown_pct": -59.9,
            },
            {
                "name": "2013 Taper Tantrum",
                "start": "2013-05-01",
                "end": "2013-08-31",
                "expected_regime": "Risk-Off",
                "nifty50_peak": 6180,
                "nifty50_trough": 5118,
                "drawdown_pct": -17.2,
            },
            {
                "name": "2018 IL&FS / NBFC Crisis",
                "start": "2018-08-01",
                "end": "2018-12-31",
                "expected_regime": "Risk-Off / Post-Shock",
                "nifty50_peak": 11788,
                "nifty50_trough": 10185,
                "drawdown_pct": -13.6,
            },
            {
                "name": "2020 COVID Crash",
                "start": "2020-02-01",
                "end": "2020-04-30",
                "expected_regime": "Post-Shock",
                "nifty50_peak": 12201,
                "nifty50_trough": 7610,
                "drawdown_pct": -37.6,
            },
            {
                "name": "2024 Election Volatility",
                "start": "2024-06-01",
                "end": "2024-06-15",
                "expected_regime": "Transitional / Risk-Off",
                "nifty50_peak": 23300,
                "nifty50_trough": 21800,
                "drawdown_pct": -6.4,
            },
        ]
    }


@app.get("/api/models/performance")
def models_performance():
    return {
        "models": [
            {"name": "Frequentist HMM", "type": "hmm", "bic": 50804.9, "converged": True, "n_regimes": 5},
            {"name": "RS-VAR", "type": "rsvar", "aic": 11331.9, "bic": 11376.5, "n_regimes": 2},
            {"name": "MC Dropout", "type": "bdl", "val_acc": 0.984, "framework": "PyTorch"},
            {"name": "Deep Ensemble (M=3)", "type": "bdl", "val_acc": 0.984, "framework": "PyTorch"},
        ],
        "ensemble_weights": {"hmm": 0.0, "rs_var": 1.0, "bdl_mc": 0.0},
        "conformal": {
            "level_90": {"coverage": 0.886, "avg_set_size": 1.87},
            "level_95": {"coverage": 0.941, "avg_set_size": 2.36},
            "ece": 0.2676,
        },
    }


@app.get("/api/regime/ic-artefact")
def regime_ic_artefact():
    features = _load_features()
    models = _get_models(features)
    last_probs = np.array(models["hmm_probs"][-1])
    regime_idx = int(np.argmax(last_probs))
    confidence = float(last_probs[regime_idx])
    date = models["dates"][-1]

    regime_text = REGIME_LABELS[regime_idx]
    if regime_idx == 0:
        rec = f"As of {date}: Market is in {regime_text} (confidence {confidence:.1%}). Tactical: maintain equity overweight, trail stops at 2-sigma."
    elif regime_idx == 1:
        rec = f"As of {date}: Market is in {regime_text} (confidence {confidence:.1%}). Defensive: reduce equity, increase cash/gold duration."
    elif regime_idx == 2:
        rec = f"As of {date}: Market is in {regime_text} (confidence {confidence:.1%}). Transitional: reduced position sizes, hedge tail risk."
    elif regime_idx == 3:
        rec = f"As of {date}: Market is in {regime_text} (confidence {confidence:.1%}). Late-cycle: rotate to quality, reduce leverage."
    else:
        rec = f"As of {date}: Market is in {regime_text} (confidence {confidence:.1%}). Post-Shock: tactical recovery trades possible, monitor credit spreads."

    return {"date": date, "regime": regime_idx, "label": regime_text, "confidence": round(confidence, 4), "recommendation": rec}


# Serve React static build in production (MUST be last — catch-all route)
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="static")

    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = _dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
