"""
Experiment 01: Data Loading Sanity Check
========================================

Purpose:
    Verify that datasets load correctly and images match their questions.
    This is the FIRST thing to run before any model work.

What to check:
    ✓ Dataset downloads and loads without errors
    ✓ Images display correctly
    ✓ Questions are coherent and match the image content
    ✓ Preprocessing (resize to 224x224) works correctly

Usage:
    python experiments/01_data_sanity_check.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def check_vqa_rad():
    """Sanity check VQA-RAD dataset."""
    print("\n" + "=" * 60)
    print("VQA-RAD Dataset Sanity Check")
    print("=" * 60)

    try:
        from data.vqa_rad_loader import VQARadDataset

        dataset = VQARadDataset(data_dir="data/vqa_rad", split="train")
        print(f"✓ Loaded {len(dataset)} training samples")

        # Display first 5 samples
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"\n  Sample {i}:")
            print(f"    Q: {sample['question']}")
            print(f"    A: {sample['answer']}")
            print(f"    Image: {'Present' if sample['image'] else 'MISSING'}")

            if sample["image_processed"] is not None:
                axes[i].imshow(sample["image_processed"])
                axes[i].set_title(f"Q: {sample['question'][:30]}...", fontsize=8)
            axes[i].axis("off")

        os.makedirs("results/figures", exist_ok=True)
        plt.tight_layout()
        plt.savefig("results/figures/vqa_rad_sanity_check.png", dpi=150)
        print(f"\n✓ Saved visualization to results/figures/vqa_rad_sanity_check.png")

        # Check preprocessing
        sample = dataset[0]
        if sample["image_processed"] is not None:
            img_arr = np.array(sample["image_processed"])
            assert img_arr.shape == (224, 224, 3), f"Bad shape: {img_arr.shape}"
            print(f"✓ Image shape after preprocessing: {img_arr.shape}")
            print(f"✓ Pixel range: [{img_arr.min()}, {img_arr.max()}]")

        return True

    except FileNotFoundError:
        print("✗ VQA-RAD not downloaded. Run: python data/download.py --dataset vqa_rad")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_preprocessing():
    """Test image preprocessing independently."""
    print("\n" + "=" * 60)
    print("Preprocessing Sanity Check")
    print("=" * 60)

    from data.preprocessing import resize_and_pad, normalize_for_model
    from PIL import Image

    # Test with a synthetic image (no dataset needed)
    test_sizes = [(100, 200), (300, 150), (50, 50), (500, 500)]

    for w, h in test_sizes:
        img = Image.new("RGB", (w, h), color=(128, 64, 200))
        result = resize_and_pad(img, (224, 224))
        assert result.size == (224, 224), f"Expected (224,224), got {result.size}"

    print("✓ Resize + pad works for various aspect ratios")

    # Test grayscale conversion
    gray_img = Image.new("L", (100, 100), color=128)
    result = resize_and_pad(gray_img, (224, 224))
    assert result.mode == "RGB", f"Expected RGB, got {result.mode}"
    print("✓ Grayscale → RGB conversion works")

    # Test normalization
    arr = np.array(result)
    normalized = normalize_for_model(arr)
    assert normalized.dtype == np.float32
    assert 0 <= normalized.min() and normalized.max() <= 1.0
    print("✓ Normalization to [0, 1] works")

    return True


def check_metrics():
    """Test evaluation metrics with known inputs."""
    print("\n" + "=" * 60)
    print("Metrics Sanity Check")
    print("=" * 60)

    from evaluation.metrics import compute_bleu1, compute_f1_token

    # Identical strings should give perfect score
    assert compute_bleu1("yes", "yes") == 1.0, "BLEU-1 identity check failed"
    assert compute_f1_token("yes", "yes") == 1.0, "F1 identity check failed"
    print("✓ Identical strings → 1.0")

    # Completely different strings should give zero
    assert compute_bleu1("cat", "dog") == 0.0, "BLEU-1 disjoint check failed"
    assert compute_f1_token("cat", "dog") == 0.0, "F1 disjoint check failed"
    print("✓ Disjoint strings → 0.0")

    # Partial overlap
    bleu = compute_bleu1("the cat sat", "the dog sat")
    assert 0 < bleu < 1, f"Expected partial score, got {bleu}"
    print(f"✓ Partial overlap BLEU-1: {bleu:.3f}")

    f1 = compute_f1_token("the cat sat on mat", "the cat sat")
    assert 0 < f1 <= 1, f"Expected partial score, got {f1}"
    print(f"✓ Partial overlap F1: {f1:.3f}")

    return True


if __name__ == "__main__":
    print("🧬 Biomedical Multimodal Reproduction — Sanity Checks")
    print("=" * 60)

    results = {}

    # Always run these (no dataset needed)
    results["preprocessing"] = check_preprocessing()
    results["metrics"] = check_metrics()

    # Try dataset loading (may fail if not downloaded yet)
    results["vqa_rad"] = check_vqa_rad()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} — {name}")

    all_passed = all(results.values())
    print(f"\n{'All checks passed!' if all_passed else 'Some checks failed — see above.'}")
    sys.exit(0 if all_passed else 1)
