"""
Experiment 04: Full VQA Training
================================

Purpose:
    Fine-tune on VQA-RAD training set, evaluate on test set,
    and compare to Med-PaLM M baselines from Table 2.

This is the main experiment that produces the reproduction results.

Usage:
    python experiments/04_train_vqa.py
    python experiments/04_train_vqa.py --dataset slake_vqa --epochs 15
    python experiments/04_train_vqa.py --multitask  # Train on VQA-RAD + Slake together
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dataset_splits(dataset_name: str):
    """Load train and test splits for a given dataset."""
    if dataset_name == "vqa_rad":
        from data.vqa_rad_loader import VQARadDataset
        train = VQARadDataset(data_dir="data/vqa_rad", split="train")
        test = VQARadDataset(data_dir="data/vqa_rad", split="test")
    elif dataset_name == "slake_vqa":
        from data.slake_loader import SlakeVQADataset
        train = SlakeVQADataset(data_dir="data/slake_vqa", split="train")
        test = SlakeVQADataset(data_dir="data/slake_vqa", split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train, test


def prepare_samples(dataset):
    """Extract samples from dataset into training format."""
    samples = []
    for i in range(len(dataset)):
        s = dataset[i]
        samples.append({
            "image": s.get("image_processed") or s.get("image"),
            "question": s["question"],
            "answer": s["answer"],
        })
    return samples


def main(args):
    print("🧬 Experiment 04: Full VQA Training")
    print("=" * 60)

    # Step 1: Load data
    print(f"\n[1/5] Loading {args.dataset} dataset...")
    try:
        train_ds, test_ds = load_dataset_splits(args.dataset)
        print(f"  ✓ Train: {len(train_ds)} | Test: {len(test_ds)}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print(f"  Run: python data/download.py --dataset {args.dataset}")
        sys.exit(1)

    # Step 2: Load model
    print(f"\n[2/5] Loading BLIP-2 model...")
    from models.blip2_wrapper import BLIP2Wrapper
    model = BLIP2Wrapper(
        model_name="Salesforce/blip2-flan-t5-xl",
        load_in_8bit=args.quantize,
    )
    model.load_model()

    # Step 3: Prepare training
    print(f"\n[3/5] Preparing training pipeline...")
    from training.trainer import VQATrainingDataset, MedVQATrainer

    train_samples = prepare_samples(train_ds)
    val_samples = prepare_samples(test_ds)

    train_dataset = VQATrainingDataset(
        samples=train_samples,
        processor=model.processor,
        use_exemplar=args.use_exemplar,
    )
    val_dataset = VQATrainingDataset(
        samples=val_samples[:100],  # Use subset for validation speed
        processor=model.processor,
        use_exemplar=args.use_exemplar,
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  One-shot exemplar: {args.use_exemplar}")

    # Step 4: Train
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    trainer = MedVQATrainer(
        model=model.model,
        processor=model.processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir="results",
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_lora=True,
        lora_rank=args.lora_rank,
        gradient_accumulation_steps=args.grad_accum,
    )
    trainer.train()

    # Step 5: Full evaluation
    print(f"\n[5/5] Running full evaluation on test set...")
    from evaluation.evaluate import evaluate_model

    model.model.eval()
    metrics = evaluate_model(
        model=model,
        dataset=test_ds,
        dataset_name=args.dataset,
        output_dir="results",
    )

    # Generate comparison
    print(f"\n{'='*60}")
    print("To see full comparison with paper baselines, run:")
    print("  python evaluation/compare_to_paper.py")
    print(f"{'='*60}")


def main_multitask(args):
    """Multi-task training on VQA-RAD + Slake-VQA simultaneously."""
    print("🧬 Experiment 04: Multi-Task VQA Training")
    print("=" * 60)

    # Load both datasets
    print("\n[1/5] Loading multiple datasets...")
    datasets_train = {}
    datasets_test = {}

    for ds_name in ["vqa_rad", "slake_vqa"]:
        try:
            train, test = load_dataset_splits(ds_name)
            datasets_train[ds_name] = train
            datasets_test[ds_name] = test
            print(f"  ✓ {ds_name}: {len(train)} train, {len(test)} test")
        except Exception as e:
            print(f"  ✗ {ds_name}: {e}")

    if not datasets_train:
        print("No datasets loaded. Download at least one:")
        print("  python data/download.py --dataset vqa_rad")
        sys.exit(1)

    # Load model
    print(f"\n[2/5] Loading BLIP-2...")
    from models.blip2_wrapper import BLIP2Wrapper
    model = BLIP2Wrapper(model_name="Salesforce/blip2-flan-t5-xl", load_in_8bit=args.quantize)
    model.load_model()

    # Create multi-task mixer
    print(f"\n[3/5] Creating multi-task mixer...")
    from training.multitask_mixer import MultiTaskMixer
    from training.trainer import VQATrainingDataset, MedVQATrainer

    # Wrap each dataset
    wrapped = {}
    for name, ds in datasets_train.items():
        samples = prepare_samples(ds)
        wrapped[name] = VQATrainingDataset(samples, model.processor, use_exemplar=True)

    mixer = MultiTaskMixer(wrapped)

    # Train
    print(f"\n[4/5] Training multi-task model...")
    trainer = MedVQATrainer(
        model=model.model,
        processor=model.processor,
        train_dataset=mixer,
        output_dir="results",
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_lora=True,
    )
    trainer.train()

    # Evaluate on each test set
    print(f"\n[5/5] Evaluating on each test set...")
    from evaluation.evaluate import evaluate_model
    model.model.eval()

    for ds_name, test_ds in datasets_test.items():
        evaluate_model(model=model, dataset=test_ds, dataset_name=ds_name, output_dir="results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vqa_rad", choices=["vqa_rad", "slake_vqa"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--use_exemplar", action="store_true", default=True)
    parser.add_argument("--no_exemplar", action="store_false", dest="use_exemplar")
    parser.add_argument("--multitask", action="store_true",
                        help="Train on VQA-RAD + Slake-VQA together (paper Section 6.2.4)")
    args = parser.parse_args()

    if args.multitask:
        main_multitask(args)
    else:
        main(args)
