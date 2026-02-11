"""
Experiment 05: Zero-Shot Generalization Evaluation
===================================================

Replicates key findings from Med-PaLM M paper Section 6.2:

1. Cross-dataset transfer: Train on VQA-RAD, test on Slake-VQA
   (tests if model generalizes to unseen medical images/questions)

2. One-shot exemplar ablation: Compare with vs without exemplar
   (tests the paper's prompting strategy)

3. Positive task transfer: Compare single-task vs multi-task training
   (replicates Table 6 from the paper)

Usage:
    python experiments/05_zero_shot_eval.py --experiment cross_dataset
    python experiments/05_zero_shot_eval.py --experiment exemplar_ablation
    python experiments/05_zero_shot_eval.py --experiment all
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cross_dataset_evaluation(model, args):
    """
    Test 1: Train on VQA-RAD, evaluate on Slake-VQA (zero-shot).

    Paper reference (Section 6.2.1):
    "Med-PaLM M can generalize to novel medical concepts and unseen
     tasks in a zero-shot fashion."
    """
    print("\n" + "=" * 60)
    print("Test 1: Cross-Dataset Zero-Shot Generalization")
    print("  Train: VQA-RAD → Test: Slake-VQA")
    print("=" * 60)

    from evaluation.evaluate import evaluate_model

    # Load Slake-VQA test set
    try:
        from data.slake_loader import SlakeVQADataset
        slake_test = SlakeVQADataset(data_dir="data/slake_vqa", split="test")
        print(f"  ✓ Loaded Slake-VQA test: {len(slake_test)} samples")
    except Exception as e:
        print(f"  ✗ Slake-VQA not found: {e}")
        print("  Run: python data/download.py --dataset slake_vqa")
        return None

    # Evaluate (model already fine-tuned on VQA-RAD from experiment 04)
    metrics = evaluate_model(
        model=model,
        dataset=slake_test,
        dataset_name="slake_vqa",
        max_samples=args.max_samples,
        output_dir="results",
    )

    return metrics


def exemplar_ablation(model, args):
    """
    Test 2: Compare performance with vs without one-shot exemplar.

    Paper reference (Section 4.2):
    "We added a text-only 'one-shot exemplar' to the task prompt
     to condition the language model's prediction."
    """
    print("\n" + "=" * 60)
    print("Test 2: One-Shot Exemplar Ablation")
    print("  Comparing: with exemplar vs without exemplar")
    print("=" * 60)

    from data.vqa_rad_loader import VQARadDataset
    from training.prompts import build_vqa_prompt
    from evaluation.metrics import compute_batch_metrics, compute_bleu1, compute_f1_token

    try:
        test_ds = VQARadDataset(data_dir="data/vqa_rad", split="test")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

    n = min(args.max_samples, len(test_ds))
    results = {}

    for use_exemplar in [True, False]:
        mode = "with_exemplar" if use_exemplar else "without_exemplar"
        print(f"\n  Running: {mode}...")

        predictions = []
        references = []

        for i in range(n):
            sample = test_ds[i]
            image = sample.get("image_processed") or sample.get("image")
            if image is None:
                continue

            if use_exemplar:
                prompt = build_vqa_prompt(
                    question=sample["question"],
                    exemplar_q="Is this a normal chest x-ray?",
                    exemplar_a="No.",
                )
            else:
                prompt = build_vqa_prompt(question=sample["question"])

            pred = model.generate(image=image, prompt=prompt, max_new_tokens=64)
            predictions.append(pred)
            references.append(sample["answer"])

        metrics = compute_batch_metrics(predictions, references)
        results[mode] = metrics
        print(f"    BLEU-1: {metrics['bleu_1']:.2f}% | F1: {metrics['f1']:.2f}%")

    # Compare
    print(f"\n  Exemplar ablation results:")
    print(f"  {'Mode':<25} {'BLEU-1':>10} {'F1':>10}")
    print(f"  {'-'*45}")
    for mode, m in results.items():
        print(f"  {mode:<25} {m['bleu_1']:>9.2f}% {m['f1']:>9.2f}%")

    diff_bleu = results["with_exemplar"]["bleu_1"] - results["without_exemplar"]["bleu_1"]
    diff_f1 = results["with_exemplar"]["f1"] - results["without_exemplar"]["f1"]
    print(f"\n  Exemplar effect: BLEU-1 {diff_bleu:+.2f}%, F1 {diff_f1:+.2f}%")

    # Save
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/exemplar_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main(args):
    print("🧬 Experiment 05: Zero-Shot Generalization Evaluation")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    from models.blip2_wrapper import BLIP2Wrapper
    model = BLIP2Wrapper(
        model_name="Salesforce/blip2-flan-t5-xl",
        load_in_8bit=args.quantize,
    )
    model.load_model()

    # Check if fine-tuned checkpoint exists
    checkpoint_path = "results/checkpoint_best"
    if os.path.exists(checkpoint_path):
        print(f"  Loading fine-tuned checkpoint from {checkpoint_path}")
        try:
            from peft import PeftModel
            model.model = PeftModel.from_pretrained(model.model, checkpoint_path)
            print("  ✓ Fine-tuned weights loaded")
        except Exception as e:
            print(f"  ⚠ Could not load checkpoint: {e}")
            print("  Using base model (zero-shot)")
    else:
        print("  No checkpoint found. Using base model (zero-shot)")
        print("  Run experiment 04 first for fine-tuned evaluation")

    # Run requested experiments
    if args.experiment in ["cross_dataset", "all"]:
        cross_dataset_evaluation(model, args)

    if args.experiment in ["exemplar_ablation", "all"]:
        exemplar_ablation(model, args)

    print(f"\n{'='*60}")
    print("All experiments complete. Results saved to results/")
    print("Run: python evaluation/compare_to_paper.py  — for full comparison")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["cross_dataset", "exemplar_ablation", "all"])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    main(args)
