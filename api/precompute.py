"""Precompute all API response payloads and bake them into the image.

Run during the Docker build so the deployed API responds instantly on boot
instead of re-training HMM/BDL models on the first request.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import CACHE_FILE, _build_crisis, _build_current, _build_history, _build_ic_artefact, _build_performance
from src.ingestion.loader import ingest_all
from src.features.engineer import build_all_features
from src.ingestion.config import PROCESSED_DIR

if __name__ == "__main__":
    print("Precomputing API responses...", flush=True)
    t0 = time.time()

    print("Fetching live data and building features...", flush=True)
    df = ingest_all(start_date="2010-01-01", use_cache=False)
    features = build_all_features(df, include_topo=True, topo_window=63)
    features.to_parquet(PROCESSED_DIR / "features.parquet")
    print(f"Features updated. Latest date: {features.index.max()}", flush=True)

    payloads = {
        "current": _build_current(),
        "history": _build_history(),
        "crisis": _build_crisis(),
        "performance": _build_performance(),
        "ic": _build_ic_artefact(),
    }

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(payloads), encoding="utf-8")
    print(f"Wrote {CACHE_FILE} ({CACHE_FILE.stat().st_size/1024:.1f} KB) in {time.time()-t0:.1f}s", flush=True)