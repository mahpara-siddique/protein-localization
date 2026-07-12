"""
STEP 5: Generate the Final Baseline Comparison Table.
Collects output from step 3 and step 4, formats it,
and prints the markdown comparison table ready to share.
"""
import pandas as pd
from config import RESULTS_DIR


def main():
    print("=" * 60)
    print("STEP 5: Generating Final Baseline Summary")
    print("=" * 60)

    ml_path = RESULTS_DIR / "ml_baselines_results.csv"
    dl_path = RESULTS_DIR / "dl_baselines_results.csv"

    if not ml_path.exists() or not dl_path.exists():
        print(f"⚠️ Error: Missing results files. Make sure to run step3 and step4 first.")
        return

    ml_df = pd.read_csv(ml_path)
    dl_df = pd.read_csv(dl_path)

    # Combine results
    all_results = pd.concat([ml_df, dl_df], ignore_index=True)

    # Format numbers to percentages/decimals
    all_results['Accuracy'] = all_results['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    all_results['Macro-F1'] = all_results['Macro-F1'].apply(lambda x: f"{x:.4f}")
    all_results['MCC'] = all_results['MCC'].apply(lambda x: f"{x:.4f}")

    # Sort by MCC descending to see the best-performing model on top
    all_results = all_results.sort_values(by='MCC', ascending=False)

    print("\n" + "#" * 60)
    print("📋 FINAL BASELINE COMPARISON TABLE (Sorted by MCC)")
    print("#" * 60)
    
    # Print as Markdown Table
    markdown_table = all_results.to_markdown(index=False)
    print(markdown_table)
    print("#" * 60)

    # Identify best baseline
    best_model = all_results.iloc[0]
    print(f"\n⭐ Best Performing Baseline: {best_model['Model']} using {best_model['Features']} features")
    print(f"   (Accuracy: {best_model['Accuracy']}, F1: {best_model['Macro-F1']}, MCC: {best_model['MCC']})")
    
    # Save markdown summary
    with open(RESULTS_DIR / "baseline_summary.md", "w") as f:
        f.write("# Baseline Model Benchmarking Summary\n\n")
        f.write(markdown_table)
        f.write(f"\n\n**Best Baseline**: {best_model['Model']} using {best_model['Features']} (MCC: {best_model['MCC']})")
    
    print(f"\nSummary report saved → {RESULTS_DIR / 'baseline_summary.md'}")


if __name__ == "__main__":
    main()