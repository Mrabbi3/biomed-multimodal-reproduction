"""
Dataset download utilities for MultiMedBench datasets.

Usage:
    python data/download.py --dataset vqa_rad
    python data/download.py --dataset slake_vqa
    python data/download.py --dataset path_vqa
    python data/download.py --dataset all
"""

import argparse
import os
import sys
from pathlib import Path

# Dataset URLs and info
DATASETS = {
    "vqa_rad": {
        "description": "VQA-RAD: Radiology Visual Question Answering",
        "size": "~50 MB",
        "qa_pairs": 3515,
        "images": 315,
        "source": "https://huggingface.co/datasets/flaviagiammarino/vqa-rad",
        "method": "huggingface",
        "hf_name": "flaviagiammarino/vqa-rad",
    },
    "slake_vqa": {
        "description": "Slake-VQA: Bilingual Medical VQA",
        "size": "~400 MB",
        "qa_pairs": 14028,
        "images": 642,
        "source": "https://huggingface.co/datasets/BoKelworworworworworworworworworwor/SLAKE",
        "method": "huggingface",
        "hf_name": "mdwiratathya/SLAKE",
    },
    "path_vqa": {
        "description": "Path-VQA: Pathology Visual Question Answering",
        "size": "~500 MB",
        "qa_pairs": 32799,
        "images": 4998,
        "source": "https://huggingface.co/datasets/flaviagiammarino/path-vqa",
        "method": "huggingface",
        "hf_name": "flaviagiammarino/path-vqa",
    },
}


def download_from_huggingface(dataset_name: str, hf_name: str, save_dir: str) -> None:
    """Download a dataset from Hugging Face Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Downloading {dataset_name} from Hugging Face: {hf_name}")
    print(f"Saving to: {save_dir}")

    dataset = load_dataset(hf_name)
    dataset.save_to_disk(save_dir)

    print(f"✓ {dataset_name} downloaded successfully!")
    print(f"  Splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")


def download_dataset(name: str) -> None:
    """Download a single dataset by name."""
    if name not in DATASETS:
        print(f"Error: Unknown dataset '{name}'")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    info = DATASETS[name]
    save_dir = os.path.join("data", name)

    print(f"\n{'='*60}")
    print(f"Dataset: {info['description']}")
    print(f"Size: {info['size']}")
    print(f"QA Pairs: {info['qa_pairs']} | Images: {info['images']}")
    print(f"{'='*60}\n")

    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"Directory {save_dir} already exists and is not empty.")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != "y":
            print("Skipping.")
            return

    os.makedirs(save_dir, exist_ok=True)

    if info["method"] == "huggingface":
        download_from_huggingface(name, info["hf_name"], save_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for biomed-multimodal-reproduction"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to download (or 'all')",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            download_dataset(name)
    else:
        download_dataset(args.dataset)


if __name__ == "__main__":
    main()
