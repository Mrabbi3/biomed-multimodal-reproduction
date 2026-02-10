"""
VQA-RAD Dataset Loader.

VQA-RAD: 315 radiology images, 3,515 QA pairs
- Modalities: CT, MRI, X-ray
- Regions: Head, Abdominal, Chest
- Question types: 58% closed-ended, 42% open-ended
"""

import os
from typing import Dict, List, Optional, Tuple

from PIL import Image
from .preprocessing import resize_and_pad


class VQARadDataset:
    """
    VQA-RAD dataset wrapper.

    Paper reference (Section A.1):
    "The training set contains 1,797 QA pairs (only free-form and
     paraphrased questions) and the test set contains 451 QA pairs."
    """

    def __init__(
        self,
        data_dir: str = "data/vqa_rad",
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.samples = []
        self._load_data()

    def _load_data(self) -> None:
        """Load dataset from saved Hugging Face format."""
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(self.data_dir)

            if self.split in dataset:
                split_data = dataset[self.split]
            else:
                print(f"Available splits: {list(dataset.keys())}")
                raise KeyError(f"Split '{self.split}' not found")

            for idx in range(len(split_data)):
                sample = split_data[idx]
                self.samples.append({
                    "image": sample.get("image"),
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                })

        except FileNotFoundError:
            print(f"Dataset not found at {self.data_dir}")
            print("Run: python data/download.py --dataset vqa_rad")
            raise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dict with keys: 'image' (PIL), 'image_processed' (PIL 224x224),
                           'question' (str), 'answer' (str)
        """
        sample = self.samples[idx]
        image = sample["image"]

        # Apply paper's preprocessing
        if image is not None:
            image_processed = resize_and_pad(image, self.image_size)
        else:
            image_processed = None

        return {
            "image": image,
            "image_processed": image_processed,
            "question": sample["question"],
            "answer": sample["answer"],
        }

    def get_sample_info(self, idx: int) -> str:
        """Get a human-readable summary of a sample."""
        sample = self[idx]
        return (
            f"Sample {idx}:\n"
            f"  Q: {sample['question']}\n"
            f"  A: {sample['answer']}\n"
            f"  Image: {'Present' if sample['image'] else 'Missing'}"
        )
