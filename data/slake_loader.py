"""
Slake-VQA Dataset Loader.

Slake-VQA: 642 annotated images, 14,028 QA pairs
- Bilingual: English and Chinese
- Modalities: CT, MRI, Chest X-ray
- 12 diseases, 39 organ systems
"""

import os
from typing import Dict, Tuple

from PIL import Image
from .preprocessing import resize_and_pad


class SlakeVQADataset:
    """
    Slake-VQA dataset wrapper.

    Paper reference (Section A.1):
    "Training, validation, and test sets contain 9,849, 2,109,
     and 2,070 samples, respectively."
    """

    def __init__(
        self,
        data_dir: str = "data/slake_vqa",
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        language: str = "en",  # 'en' or 'zh'
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.language = language
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
            print("Run: python data/download.py --dataset slake_vqa")
            raise

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = sample["image"]

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
