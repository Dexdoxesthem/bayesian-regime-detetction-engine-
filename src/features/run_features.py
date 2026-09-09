"""
Feature engineering validation script.

Builds all features from the merged dataset, validates completeness,
and saves diagnostics.
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.features.engineer import build_all_features, get_feature_descriptions
from src.ingestion.config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("FEATURE ENGINEERING — Validation")
    print("=" * 70)

    # Load merged data
    data_path = PROCESSED_DIR / "merged_market_data.parquet"
    if not data_path.exists():
        print(f"FATAL: {data_path} not found. Run ingestion first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    print(f"\nInput data: {df.shape[0]} rows x {df.shape[1]} columns")

    # Build features
    print("\nBuilding features...")
    features = build_all_features(df, include_topo=True)

    # Validate
    print(f"\nOutput features: {features.shape[1]} columns")
    print(f"Date range: {features.index.min()} to {features.index.max()}")

    # Missing data summary
    missing = features.isna().mean()
    complete_features = (missing == 0).sum()
    partial_features = ((missing > 0) & (missing <= 0.1)).sum()
    high_missing = (missing > 0.1).sum()
    print(f"\nComplete features (0% missing): {complete_features}")
    print(f"Partial features (<10% missing): {partial_features}")
    print(f"High missing (>10% missing): {high_missing}")

    if high_missing > 0:
        print("\nHigh-missing features:")
        for col in missing[missing > 0.1].sort_values(ascending=False).index:
            print(f"  {col}: {missing[col]*100:.1f}%")

    # Feature descriptions
    descriptions = get_feature_descriptions()
    matched = sum(1 for col in features.columns if col in descriptions)
    print(f"\nDocumented features: {matched}/{len(features.columns)}")

    # Save features
    out_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out_path)
    print(f"\nSaved: {out_path}")

    # Save feature catalog
    catalog = {
        col: descriptions.get(col, "No description")
        for col in features.columns
    }
    catalog_path = PROCESSED_DIR / "feature_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Feature catalog: {catalog_path}")

    # Final validation
    print("\n" + "=" * 70)
    if features.shape[1] >= 30:
        print(f"FEATURE ENGINEERING: SUCCESS ({features.shape[1]} features)")
        return 0
    else:
        print(f"FEATURE ENGINEERING: PARTIAL ({features.shape[1]} features, need 30+)")
        return 1


if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())
