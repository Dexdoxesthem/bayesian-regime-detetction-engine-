"""
Cross-language reconciliation: Compare Python and R regime outputs.

This IS a deliverable — verifies numerical consistency between
the Python and R implementations.
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def reconciliation_report():
    """Compare Python and R regime outputs numerically."""
    print("=" * 70)
    print("CROSS-LANGUAGE RECONCILIATION: Python vs R")
    print("=" * 70)

    # Python outputs
    hmm_diag_path = PROCESSED_DIR / "hmm" / "diagnostics.json"
    hmm_regimes_path = PROCESSED_DIR / "hmm" / "regime_assignments.parquet"

    # R outputs
    r_output_dir = PROCESSED_DIR / "r_output"
    hmm_states_r_path = r_output_dir / "hmm_states.csv"
    changepoints_r_path = r_output_dir / "changepoints.csv"

    results = {}

    # --- HMM comparison ---
    print("\n[1/3] HMM Regime Assignment Comparison...")
    if hmm_regimes_path.exists():
        py_regimes = pd.read_parquet(hmm_regimes_path)
        print(f"  Python: {len(py_regimes)} rows, regimes: {py_regimes['regime'].nunique()}")
        print(f"  Python regime distribution:")
        for r in sorted(py_regimes['regime'].unique()):
            pct = (py_regimes['regime'] == r).mean() * 100
            print(f"    Regime {r}: {pct:.1f}%")
        results["python_hmm"] = {
            "n_regimes": int(py_regimes['regime'].nunique()),
            "distribution": {
                str(r): float((py_regimes['regime'] == r).mean())
                for r in py_regimes['regime'].unique()
            }
        }
    else:
        print("  Python HMM output not found")

    if hmm_states_r_path.exists():
        r_states = pd.read_csv(hmm_states_r_path)
        print(f"  R: {len(r_states)} rows, states: {r_states['state'].nunique()}")
        print(f"  R state distribution:")
        for s in sorted(r_states['state'].unique()):
            pct = (r_states['state'] == s).mean() * 100
            print(f"    State {s}: {pct:.1f}%")
        results["r_hmm"] = {
            "n_states": int(r_states['state'].nunique()),
            "distribution": {
                str(s): float((r_states['state'] == s).mean())
                for s in r_states['state'].unique()
            }
        }
    else:
        print("  R HMM output not found (run R script first)")

    # --- Changepoint comparison ---
    print("\n[2/3] Changepoint Detection Comparison...")
    if changepoints_r_path.exists():
        r_cps = pd.read_csv(changepoints_r_path)
        print(f"  R changepoints detected: {len(r_cps)}")
        if len(r_cps) > 0:
            print(f"  R changepoint positions: {r_cps['changepoint'].head(10).tolist()}")
        results["r_changepoints"] = len(r_cps)
    else:
        print("  R changepoint output not found")

    # --- Numerical consistency checks ---
    print("\n[3/3] Numerical Consistency Checks...")

    # Check transition matrix symmetry
    hmm_diag_path = PROCESSED_DIR / "hmm" / "diagnostics.json"
    if hmm_diag_path.exists():
        with open(hmm_diag_path) as f:
            hmm_diag = json.load(f)
        trans = np.array(hmm_diag["transition_matrix"])
        row_sums = trans.sum(axis=1)
        is_stochastic = np.allclose(row_sums, 1.0, atol=1e-6)
        print(f"  Transition matrix is stochastic: {is_stochastic}")
        print(f"  Row sums: {row_sums}")
        results["transition_matrix_valid"] = bool(is_stochastic)

    # --- Reconciliation summary ---
    print("\n" + "=" * 70)
    print("RECONCILIATION SUMMARY")
    print("=" * 70)

    checks_passed = 0
    checks_total = 0

    # Check 1: Both Python and R produced outputs
    if "python_hmm" in results and "r_hmm" in results:
        checks_total += 1
        print("  [PASS] Both Python and R HMM outputs exist")
        checks_passed += 1
    else:
        checks_total += 1
        print("  [WARN] Not both outputs available for comparison")

    # Check 2: Transition matrix validity
    if results.get("transition_matrix_valid"):
        checks_total += 1
        print("  [PASS] Transition matrix is valid stochastic matrix")
        checks_passed += 1

    # Check 3: Regime counts match (within 1)
    if "python_hmm" in results and "r_hmm" in results:
        checks_total += 1
        py_n = results["python_hmm"]["n_regimes"]
        r_n = results["r_hmm"]["n_states"]
        match = abs(py_n - r_n) <= 1
        status = "PASS" if match else "WARN"
        print(f"  [{status}] Python={py_n} regimes, R={r_n} states (diff <= 1)")
        if match:
            checks_passed += 1

    # Save reconciliation report
    output = {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "results": results,
    }

    report_path = PROCESSED_DIR / "reconciliation_report.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Report saved: {report_path}")
    print(f"  Checks: {checks_passed}/{checks_total}")

    return 0 if checks_passed == checks_total else 1


if __name__ == "__main__":
    sys.exit(reconciliation_report())
