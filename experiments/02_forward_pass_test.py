"""
Experiment 02: Forward Pass Test
================================

Purpose:
    Verify that the model loads and can produce text output
    from an image + prompt WITHOUT any training (baseline).

What this tests:
    - Model downloads and loads correctly
    - Image preprocessing pipeline works end-to-end
    - Model generates coherent text (even if medically wrong)
    - Establishes zero-shot baseline metrics

Usage:
    python experiments/02_forward_pass_test.py
    python experiments/02_forward_pass_test.py --model blip2
    python experiments/02_forward_pass_test.py --max_samples 20
"""

import sys
import os
import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(args):
    print("🧬 Experiment 02: Forward Pass Test")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/4] Loading dataset...")
    try:
        from data.vqa_rad_loader import VQARadDataset
        dataset = VQARadDataset(data_dir="data/vqa_rad", split="test")
        print(f"  ✓ Loaded {len(dataset)} test samples")
    except Exception as e:
        print(f"  ✗ Dataset error: {e}")
        print("  Run: python data/download.py --dataset vqa_rad")
        sys.exit(1)

    # Step 2: Load model
    print(f"\n[2/4] Loading model ({args.model})...")
    if args.model == "blip2":
        from models.blip2_wrapper import BLIP2Wrapper
        model = BLIP2Wrapper(
            model_name="Salesforce/blip2-flan-t5-xl",
            load_in_8bit=args.quantize,
        )
    elif args.model == "llava_med":
        from models.llava_med_wrapper import LLaVAMedWrapper
        model = LLaVAMedWrapper(load_in_4bit=args.quantize)
    else:
        print(f"  Unknown model: {args.model}")
        sys.exit(1)

    model.load_model()

    # Step 3: Run inference
    print(f"\n[3/4] Running inference on {min(args.max_samples, len(dataset))} samples...")
    from training.prompts import build_vqa_prompt
    from evaluation.metrics import compute_bleu1, compute_f1_token

    predictions = []
    references = []
    n = min(args.max_samples, len(dataset))
    start = time.time()

    for i in range(n):
        sample = dataset[i]
        image = sample.get("image_processed") or sample.get("image")
        if image is None:
            continue

        prompt = build_vqa_prompt(question=sample["question"])
        prediction = model.generate(image=image, prompt=prompt, max_new_tokens=64)

        predictions.append(prediction)
        references.append(sample["answer"])

        if i < 5:
            print(f"\n  Sample {i}:")
            print(f"    Q: {sample['question']}")
            print(f"    Ground truth: {sample['answer']}")
            print(f"    Prediction:   {prediction}")
            bleu = compute_bleu1(prediction, sample["answer"]) * 100
            f1 = compute_f1_token(prediction, sample["answer"]) * 100
            print(f"    BLEU-1: {bleu:.1f}% | F1: {f1:.1f}%")

    elapsed = time.time() - start

    # Step 4: Compute baseline
    print(f"\n[4/4] Computing baseline metrics...")
    from evaluation.metrics import compute_batch_metrics
    metrics = compute_batch_metrics(predictions, references)

    print(f"\n{'='*60}")
    print(f"ZERO-SHOT BASELINE RESULTS ({args.model})")
    print(f"{'='*60}")
    print(f"  BLEU-1: {metrics['bleu_1']:.2f}%")
    print(f"  F1:     {metrics['f1']:.2f}%")
    print(f"  Samples: {metrics['num_samples']}")
    print(f"  Speed: {n/elapsed:.1f} samples/sec")
    print(f"\n  Paper comparison (VQA-RAD):")
    print(f"    PaLM-E 84B (no finetune): BLEU-1=59.19%, F1=38.67%")
    print(f"    Med-PaLM M 562B:          BLEU-1=71.27%, F1=62.06%")
    print(f"{'='*60}")

    os.makedirs("results/tables", exist_ok=True)
    # Save as both baseline and vqa_rad so comparison chart picks it up
    result_data = {"model": args.model, "mode": "zero_shot", **metrics,
                   "elapsed_seconds": elapsed}
    with open("results/tables/baseline_metrics.json", "w") as f:
        json.dump(result_data, f, indent=2)
    with open("results/tables/vqa_rad_metrics.json", "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n✓ Saved to results/tables/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="blip2", choices=["blip2", "llava_med"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    main(args)
