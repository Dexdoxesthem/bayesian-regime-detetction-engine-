"""
End-to-end ingestion test script.

Runs full data ingestion pipeline on real data, saves diagnostics,
and verifies the output is usable for downstream feature engineering.
"""
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.loader import ingest_all, run_quality_diagnostics
from src.ingestion.config import PROCESSED_DIR, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BAYESIAN REGIME DETECTION ENGINE — Data Ingestion Pipeline")
    print("=" * 70)

    # 1. Ingest all data
    print("\n[1/3] Ingesting all data sources...")
    try:
        df = ingest_all(start_date="2010-01-01", use_cache=True, save=True)
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        print(f"\nFATAL: Ingestion failed — {e}")
        sys.exit(1)

    # 2. Run quality diagnostics
    print("\n[2/3] Running data quality diagnostics...")
    dq_report = run_quality_diagnostics(df)

    # Save diagnostics
    dq_path = output_dir / "data_quality_report.json"
    with open(dq_path, "w") as f:
        json.dump(dq_report, f, indent=2, default=str)
    print(f"  Diagnostics saved to: {dq_path}")

    # 3. Summary statistics
    print("\n[3/3] Summary statistics:")
    print(f"  Rows:            {dq_report['n_rows']}")
    print(f"  Columns:         {dq_report['n_cols']}")
    print(f"  Date range:      {dq_report['date_range'][0]} to {dq_report['date_range'][1]}")
    print(f"  History:         {dq_report['history_years']:.1f} years")
    print(f"  Columns:         {', '.join(dq_report['columns'][:10])}...")

    # Show missing data summary
    missing_cols = {k: v for k, v in dq_report["missing_pct"].items() if v > 0}
    if missing_cols:
        print(f"\n  Missing data:")
        for col, pct in sorted(missing_cols.items(), key=lambda x: -x[1]):
            print(f"    {col}: {pct}%")
    else:
        print(f"\n  No missing data detected")

    # Show warnings
    if dq_report["warnings"]:
        print(f"\n  Warnings ({len(dq_report['warnings'])}):")
        for w in dq_report["warnings"][:10]:
            print(f"    - {w.encode('ascii', 'replace').decode()}")
        if len(dq_report["warnings"]) > 10:
            print(f"    ... and {len(dq_report['warnings']) - 10} more")
    else:
        print(f"\n  All quality checks passed")

    # Save head/tail for inspection
    sample_path = output_dir / "sample_data.parquet"
    pd_sample = df.head(50).copy()
    pd_sample.to_parquet(sample_path)

    # Final check
    print("\n" + "=" * 70)
    critical_cols = ["nifty50_close", "nifty50_volume"]
    has_critical = all(c in df.columns for c in critical_cols)

    if has_critical and dq_report["n_rows"] > 2000:
        print("INGESTION PIPELINE: SUCCESS")
        print(f"  Output: {output_dir / 'merged_market_data.parquet'}")
        print(f"  Ready for feature engineering")
        return 0
    else:
        print("INGESTION PIPELINE: PARTIAL — check warnings")
        missing_critical = [c for c in critical_cols if c not in df.columns]
        if missing_critical:
            print(f"  Missing critical columns: {missing_critical}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
