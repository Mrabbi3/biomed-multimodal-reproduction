"""
Compare reproduction results to Med-PaLM M paper (Table 2).

This script reads saved evaluation results and generates a unified
comparison table matching the paper's format. It also creates
visualization plots for the final report.
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str = "results/tables") -> dict:
    """Load all saved evaluation metrics."""
    all_results = {}
    if not os.path.exists(results_dir):
        return all_results

    for fname in os.listdir(results_dir):
        if fname.endswith("_metrics.json"):
            dataset_name = fname.replace("_metrics.json", "")
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)

            # Map 'baseline' to 'vqa_rad' so paper baselines are found
            if dataset_name == "baseline":
                dataset_name = "vqa_rad"

            # If we already have results for this dataset, keep the fine-tuned one
            if dataset_name in all_results and all_results[dataset_name].get("mode") == "fine_tuned":
                continue
            all_results[dataset_name] = data

    return all_results


def generate_full_comparison(results_dir: str = "results/tables"):
    """Generate the complete comparison table (our Table 2 equivalent)."""
    results = load_results(results_dir)

    if not results:
        print("No results found. Run evaluation experiments first:")
        print("  python experiments/02_forward_pass_test.py")
        print("  python experiments/04_train_vqa.py")
        return

    # Build unified table
    print("\n" + "=" * 80)
    print("REPRODUCTION RESULTS vs MED-PALM M (Table 2 Comparison)")
    print("=" * 80)

    header = f"{'Task':<15} {'Dataset':<15} {'Metric':<10} {'SOTA':>10} {'PaLM-E':>10} {'MPM-12B':>10} {'MPM-562B':>10} {'Ours':>10}"
    print(header)
    print("-" * 80)

    from evaluation.evaluate import PAPER_BASELINES

    for dataset_name, metrics in results.items():
        baselines = PAPER_BASELINES.get(dataset_name, {})

        for metric_name in ["bleu_1", "f1"]:
            sota = baselines.get("prior_sota", {}).get(metric_name)
            palm_e = baselines.get("palm_e_84b", {}).get(metric_name)
            mpm_12b = baselines.get("med_palm_m_12b", {}).get(metric_name)
            mpm_562b = baselines.get("med_palm_m_562b", {}).get(metric_name)
            ours = metrics.get(metric_name)

            def fmt(v):
                return f"{v:.2f}%" if v is not None else "N/A"

            print(f"{'VQA':<15} {dataset_name:<15} {metric_name:<10} "
                  f"{fmt(sota):>10} {fmt(palm_e):>10} {fmt(mpm_12b):>10} "
                  f"{fmt(mpm_562b):>10} {fmt(ours):>10}")

    print("=" * 80)

    # Save as markdown
    _save_unified_markdown(results, results_dir)


def _save_unified_markdown(results: dict, output_dir: str):
    """Save the unified comparison as a markdown table."""
    from evaluation.evaluate import PAPER_BASELINES

    lines = [
        "# Reproduction Results vs Med-PaLM M (Table 2)",
        "",
        "| Dataset | Metric | Prior SOTA | PaLM-E 84B | Med-PaLM M 12B | Med-PaLM M 562B | **Ours** |",
        "|---------|--------|------------|------------|----------------|-----------------|----------|",
    ]

    for dataset_name, metrics in results.items():
        baselines = PAPER_BASELINES.get(dataset_name, {})

        for metric_name in ["bleu_1", "f1"]:
            def fmt(v):
                return f"{v:.2f}%" if v is not None else "N/A"

            sota = baselines.get("prior_sota", {}).get(metric_name)
            palm_e = baselines.get("palm_e_84b", {}).get(metric_name)
            mpm_12b = baselines.get("med_palm_m_12b", {}).get(metric_name)
            mpm_562b = baselines.get("med_palm_m_562b", {}).get(metric_name)
            ours = metrics.get(metric_name)

            lines.append(
                f"| {dataset_name} | {metric_name} | {fmt(sota)} | {fmt(palm_e)} | "
                f"{fmt(mpm_12b)} | {fmt(mpm_562b)} | **{fmt(ours)}** |"
            )

    path = os.path.join(output_dir, "full_comparison.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Saved comparison table to {path}")


def generate_plots(results_dir: str = "results"):
    """Generate bar chart comparing our results to paper baselines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    results = load_results(os.path.join(results_dir, "tables"))
    if not results:
        return

    from evaluation.evaluate import PAPER_BASELINES

    for dataset_name, metrics in results.items():
        baselines = PAPER_BASELINES.get(dataset_name, {})

        models = ["PaLM-E\n84B", "Med-PaLM M\n12B", "Med-PaLM M\n562B", "Ours"]
        bleu_scores = [
            baselines.get("palm_e_84b", {}).get("bleu_1", 0),
            baselines.get("med_palm_m_12b", {}).get("bleu_1", 0),
            baselines.get("med_palm_m_562b", {}).get("bleu_1", 0),
            metrics.get("bleu_1", 0),
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#95a5a6", "#3498db", "#2980b9", "#e74c3c"]
        bars = ax.bar(models, bleu_scores, color=colors, edgecolor="white", linewidth=1.5)

        # Add value labels
        for bar, score in zip(bars, bleu_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{score:.1f}%", ha="center", va="bottom", fontweight="bold")

        ax.set_ylabel("BLEU-1 (%)")
        ax.set_title(f"{dataset_name} — BLEU-1 Comparison")
        ax.set_ylim(0, max(bleu_scores) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig_path = os.path.join(results_dir, "figures", f"{dataset_name}_comparison.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"✓ Saved plot: {fig_path}")


if __name__ == "__main__":
    generate_full_comparison()
    generate_plots()
