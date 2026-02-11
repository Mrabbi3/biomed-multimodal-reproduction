"""
Experiment 03: Overfit Single Batch ("Golden Test")
====================================================

Purpose:
    Train on exactly 5 examples for many iterations.
    If the model can't memorize 5 examples, there's a code bug.

Expected outcome:
    - Loss decreases steadily toward ~0
    - Model reproduces training answers perfectly

This is a DEBUGGING tool, not a real experiment. Run this before
any real training to verify the pipeline works end-to-end.

Usage:
    python experiments/03_overfit_single_batch.py
    python experiments/03_overfit_single_batch.py --num_samples 10 --epochs 50
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(args):
    print("🧬 Experiment 03: Overfit Single Batch (Golden Test)")
    print("=" * 60)
    print(f"  Samples: {args.num_samples}")
    print(f"  Epochs:  {args.epochs}")
    print("=" * 60)

    # Step 1: Load a tiny subset
    print("\n[1/5] Loading tiny dataset...")
    try:
        from data.vqa_rad_loader import VQARadDataset
        full_dataset = VQARadDataset(data_dir="data/vqa_rad", split="train")
        # Take only N samples
        tiny_samples = [full_dataset[i] for i in range(min(args.num_samples, len(full_dataset)))]
        print(f"  ✓ Selected {len(tiny_samples)} samples for overfitting")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)

    # Show what we're trying to memorize
    print("\n  Memorization targets:")
    for i, s in enumerate(tiny_samples):
        print(f"    {i}: Q='{s['question'][:50]}...' → A='{s['answer']}'")

    # Step 2: Load model
    print(f"\n[2/5] Loading model...")
    from models.blip2_wrapper import BLIP2Wrapper
    model = BLIP2Wrapper(
        model_name="Salesforce/blip2-flan-t5-xl",
        load_in_8bit=args.quantize,
    )
    model.load_model()

    # Step 3: Create training dataset
    print("\n[3/5] Setting up training...")
    from training.trainer import VQATrainingDataset, MedVQATrainer

    # Wrap samples into training format
    train_samples = []
    for s in tiny_samples:
        train_samples.append({
            "image": s.get("image_processed") or s.get("image"),
            "question": s["question"],
            "answer": s["answer"],
        })

    train_dataset = VQATrainingDataset(
        samples=train_samples,
        processor=model.processor,
        use_exemplar=False,  # Keep it simple for overfit test
    )

    # Step 4: Train
    print(f"\n[4/5] Training for {args.epochs} epochs on {len(train_dataset)} samples...")
    trainer = MedVQATrainer(
        model=model.model,
        processor=model.processor,
        train_dataset=train_dataset,
        output_dir="results",
        learning_rate=1e-4,  # Higher LR for fast memorization
        batch_size=min(args.num_samples, 4),
        num_epochs=args.epochs,
        use_lora=True,
        gradient_accumulation_steps=1,
    )
    trainer.train()

    # Step 5: Verify memorization
    print(f"\n[5/5] Verifying memorization...")
    from training.prompts import build_vqa_prompt
    from evaluation.metrics import compute_bleu1, compute_f1_token

    model.model.eval()
    perfect = 0

    for i, s in enumerate(tiny_samples):
        image = s.get("image_processed") or s.get("image")
        if image is None:
            continue

        prompt = build_vqa_prompt(question=s["question"])
        prediction = model.generate(image=image, prompt=prompt, max_new_tokens=32)

        exact_match = prediction.strip().lower() == s["answer"].strip().lower()
        f1 = compute_f1_token(prediction, s["answer"]) * 100

        status = "✓" if exact_match else "✗"
        print(f"  {status} Q: {s['question'][:40]}...")
        print(f"    Expected: '{s['answer']}' | Got: '{prediction}' | F1: {f1:.0f}%")

        if exact_match:
            perfect += 1

    print(f"\n{'='*60}")
    print(f"MEMORIZATION RESULT: {perfect}/{len(tiny_samples)} exact matches")
    if perfect == len(tiny_samples):
        print("✓ PASS — Pipeline is working correctly!")
    elif perfect > 0:
        print("⚠ PARTIAL — Model is learning but may need more epochs")
    else:
        print("✗ FAIL — Possible bug in training pipeline")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    main(args)
