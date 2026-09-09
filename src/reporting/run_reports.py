"""
Final run script: generates all reports, model cards, and reconciliation.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ingestion.config import PROCESSED_DIR
from src.reporting.generator import generate_model_card, generate_report
from src.reporting.reconciliation import reconciliation_report


def main():
    print("=" * 70)
    print("GENERATING ALL REPORTS AND MODEL CARDS")
    print("=" * 70)

    # Load diagnostics
    results_file = PROCESSED_DIR / "all_model_results.json"
    pipeline_dir = PROCESSED_DIR / "pipeline_output"

    all_diag = {}
    if results_file.exists():
        with open(results_file) as f:
            all_diag = json.load(f)

    # Add pipeline results
    for name in ["backtest_results", "mc_var", "ic_artefact", "conformal_diagnostics"]:
        fpath = pipeline_dir / f"{name}.json"
        if fpath.exists():
            with open(fpath) as f:
                all_diag[name] = json.load(f)

    # Generate model cards
    output_dir = PROCESSED_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, diag in all_diag.items():
        if isinstance(diag, dict) and "error" not in diag:
            try:
                card_path = generate_model_card(model_name, diag, output_dir)
                print(f"  Model card: {card_path}")
            except Exception as e:
                print(f"  Model card failed for {model_name}: {e}")

    # Generate main report
    report_path = generate_report(all_diag, output_dir)
    print(f"  Main report: {report_path}")

    # Run reconciliation
    print("\nRunning cross-language reconciliation...")
    try:
        reconciliation_report()
    except Exception as e:
        print(f"  Reconciliation skipped: {e}")

    print("\n" + "=" * 70)
    print("ALL REPORTS GENERATED")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
