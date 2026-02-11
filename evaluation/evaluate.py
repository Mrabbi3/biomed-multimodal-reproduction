"""
Full evaluation pipeline for comparing model outputs to Med-PaLM M baselines.

Runs model inference on test sets and produces comparison tables
matching the format of Table 2 and Table 3 from the paper.
"""

import os
import json
import time
import torch
from typing import Dict, List, Optional
from tqdm import tqdm

from evaluation.metrics import compute_batch_metrics, compute_bleu1, compute_f1_token


# Paper baselines from Table 2 (best across model scales)
PAPER_BASELINES = {
    "vqa_rad": {
        "prior_sota": {"bleu_1": 71.03, "f1": None},
        "palm_e_84b": {"bleu_1": 59.19, "f1": 38.67},
        "med_palm_m_12b": {"bleu_1": 64.02, "f1": 50.66},
        "med_palm_m_84b": {"bleu_1": 69.38, "f1": 59.90},
        "med_palm_m_562b": {"bleu_1": 71.27, "f1": 62.06},
    },
    "slake_vqa": {
        "prior_sota": {"bleu_1": 78.60, "f1": 78.10},
        "palm_e_84b": {"bleu_1": 52.65, "f1": 24.53},
        "med_palm_m_12b": {"bleu_1": 90.77, "f1": 86.22},
        "med_palm_m_84b": {"bleu_1": 92.70, "f1": 89.28},
        "med_palm_m_562b": {"bleu_1": 91.64, "f1": 87.50},
    },
    "path_vqa": {
        "prior_sota": {"bleu_1": 70.30, "f1": 58.40},
        "palm_e_84b": {"bleu_1": 54.92, "f1": 29.68},
        "med_palm_m_12b": {"bleu_1": 68.97, "f1": 57.24},
        "med_palm_m_84b": {"bleu_1": 70.16, "f1": 59.51},
        "med_palm_m_562b": {"bleu_1": 72.27, "f1": 62.69},
    },
}


def evaluate_model(
    model,
    dataset,
    dataset_name: str,
    max_samples: int = None,
    output_dir: str = "results",
    save_predictions: bool = True,
) -> Dict:
    """
    Run full evaluation on a dataset and compare to paper baselines.

    Args:
        model: Model wrapper (BLIP2Wrapper or LLaVAMedWrapper)
        dataset: Test dataset
        dataset_name: Name for baseline lookup ('vqa_rad', 'slake_vqa', 'path_vqa')
        max_samples: Limit evaluation to N samples (for quick testing)
        output_dir: Directory to save results
        save_predictions: Whether to save individual predictions

    Returns:
        Dict with metrics and comparison
    """
    from training.prompts import build_vqa_prompt

    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")

    predictions = []
    references = []
    detailed_results = []

    n_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    start_time = time.time()

    for i in tqdm(range(n_samples), desc="Evaluating"):
        sample = dataset[i]

        prompt = build_vqa_prompt(question=sample["question"])
        image = sample.get("image_processed") or sample.get("image")

        if image is None:
            continue

        try:
            prediction = model.generate(image=image, prompt=prompt)
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            prediction = ""

        predictions.append(prediction)
        references.append(sample["answer"])

        detailed_results.append({
            "index": i,
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "prediction": prediction,
            "bleu_1": compute_bleu1(prediction, sample["answer"]) * 100,
            "f1": compute_f1_token(prediction, sample["answer"]) * 100,
        })

    elapsed = time.time() - start_time

    # Compute aggregate metrics
    metrics = compute_batch_metrics(predictions, references)
    metrics["dataset"] = dataset_name
    metrics["elapsed_seconds"] = elapsed
    metrics["samples_per_second"] = n_samples / elapsed

    # Compare to paper baselines
    comparison = _build_comparison_table(dataset_name, metrics)
    metrics["comparison"] = comparison

    # Print results
    _print_results(dataset_name, metrics, comparison)

    # Save results
    if save_predictions:
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

        # Save detailed predictions
        pred_path = os.path.join(output_dir, "logs", f"{dataset_name}_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save summary metrics
        metrics_path = os.path.join(output_dir, "tables", f"{dataset_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save comparison table as markdown
        table_path = os.path.join(output_dir, "tables", f"{dataset_name}_comparison.md")
        _save_comparison_markdown(table_path, dataset_name, metrics, comparison)

        print(f"\n  Results saved to {output_dir}/")

    return metrics


def _build_comparison_table(dataset_name: str, our_metrics: Dict) -> List[Dict]:
    """Build comparison table between our results and paper baselines."""
    baselines = PAPER_BASELINES.get(dataset_name, {})
    table = []

    for model_name, baseline_metrics in baselines.items():
        table.append({
            "model": model_name,
            "bleu_1": baseline_metrics.get("bleu_1"),
            "f1": baseline_metrics.get("f1"),
            "source": "paper",
        })

    table.append({
        "model": "Ours",
        "bleu_1": round(our_metrics["bleu_1"], 2),
        "f1": round(our_metrics["f1"], 2),
        "source": "reproduction",
    })

    return table


def _print_results(dataset_name: str, metrics: Dict, comparison: List[Dict]):
    """Print formatted results table."""
    print(f"\n  Results on {dataset_name}:")
    print(f"  {'Model':<25} {'BLEU-1':>10} {'F1':>10}")
    print(f"  {'-'*45}")

    for row in comparison:
        bleu = f"{row['bleu_1']:.2f}%" if row['bleu_1'] is not None else "N/A"
        f1 = f"{row['f1']:.2f}%" if row['f1'] is not None else "N/A"
        marker = " ← ours" if row["source"] == "reproduction" else ""
        print(f"  {row['model']:<25} {bleu:>10} {f1:>10}{marker}")

    print(f"\n  Inference: {metrics['samples_per_second']:.1f} samples/sec")


def _save_comparison_markdown(path: str, dataset_name: str, metrics: Dict, comparison: List[Dict]):
    """Save comparison table as a markdown file."""
    lines = [
        f"# {dataset_name} — Results Comparison",
        "",
        f"| Model | BLEU-1 | F1 |",
        f"|-------|--------|-----|",
    ]

    for row in comparison:
        bleu = f"{row['bleu_1']:.2f}%" if row['bleu_1'] is not None else "N/A"
        f1 = f"{row['f1']:.2f}%" if row['f1'] is not None else "N/A"
        bold = "**" if row["source"] == "reproduction" else ""
        lines.append(f"| {bold}{row['model']}{bold} | {bold}{bleu}{bold} | {bold}{f1}{bold} |")

    lines.extend([
        "",
        f"Evaluated on {metrics['num_samples']} samples.",
        f"Inference speed: {metrics['samples_per_second']:.1f} samples/sec.",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))
