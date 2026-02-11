"""
Multi-task data mixer following Med-PaLM M training strategy.

Paper (Section 4.2, Table A.1):
"We trained the model with a mixture of distinct tasks simultaneously
 via instruction tuning. Mixture ratios were empirically determined
 such that they are approximately proportional to the number of
 training samples in each dataset."

This module handles sampling from multiple VQA datasets with
configurable mixture ratios for multi-task training.
"""

import random
from torch.utils.data import Dataset
from typing import Dict, List


class MultiTaskMixer(Dataset):
    """
    Combines multiple VQA datasets into a single training set
    with configurable sampling ratios.
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        mixture_ratios: Dict[str, float] = None,
        total_samples_per_epoch: int = None,
    ):
        """
        Args:
            datasets: Dict mapping dataset name → Dataset object
            mixture_ratios: Dict mapping dataset name → sampling weight
                           (defaults to proportional to dataset size)
            total_samples_per_epoch: Fixed epoch size (default: sum of all datasets)
        """
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())

        # Calculate mixture ratios
        if mixture_ratios is None:
            total = sum(len(d) for d in datasets.values())
            self.mixture_ratios = {
                name: len(ds) / total for name, ds in datasets.items()
            }
        else:
            # Normalize ratios
            total_ratio = sum(mixture_ratios.values())
            self.mixture_ratios = {
                k: v / total_ratio for k, v in mixture_ratios.items()
            }

        if total_samples_per_epoch is None:
            self.total_samples = sum(len(d) for d in datasets.values())
        else:
            self.total_samples = total_samples_per_epoch

        # Pre-compute the sampling schedule for one epoch
        self._build_schedule()

        print(f"MultiTaskMixer initialized:")
        for name in self.dataset_names:
            count = len(datasets[name])
            ratio = self.mixture_ratios[name]
            print(f"  {name}: {count} samples, {ratio*100:.1f}% mixture")
        print(f"  Total samples per epoch: {self.total_samples}")

    def _build_schedule(self):
        """Build a shuffled sampling schedule for one epoch."""
        self.schedule = []
        for name in self.dataset_names:
            n_samples = int(self.total_samples * self.mixture_ratios[name])
            ds_len = len(self.datasets[name])
            indices = [i % ds_len for i in range(n_samples)]
            self.schedule.extend([(name, idx) for idx in indices])
        random.shuffle(self.schedule)

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, idx):
        dataset_name, sample_idx = self.schedule[idx]
        sample = self.datasets[dataset_name][sample_idx]
        sample["dataset_name"] = dataset_name
        return sample

    def reshuffle(self):
        """Reshuffle schedule (call between epochs)."""
        self._build_schedule()
