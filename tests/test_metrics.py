"""
Unit tests for evaluation metrics.

Run: pytest tests/test_metrics.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import compute_bleu1, compute_f1_token, compute_batch_metrics


class TestBLEU1:
    def test_identical(self):
        assert compute_bleu1("yes", "yes") == 1.0

    def test_disjoint(self):
        assert compute_bleu1("cat", "dog") == 0.0

    def test_empty_prediction(self):
        assert compute_bleu1("", "yes") == 0.0

    def test_partial_overlap(self):
        score = compute_bleu1("the cat sat", "the dog sat")
        assert 0.5 < score < 1.0  # 2/3 match

    def test_case_insensitive(self):
        assert compute_bleu1("Yes", "yes") == 1.0


class TestF1Token:
    def test_identical(self):
        assert compute_f1_token("yes", "yes") == 1.0

    def test_disjoint(self):
        assert compute_f1_token("cat", "dog") == 0.0

    def test_both_empty(self):
        assert compute_f1_token("", "") == 1.0

    def test_one_empty(self):
        assert compute_f1_token("", "yes") == 0.0
        assert compute_f1_token("yes", "") == 0.0

    def test_superset_prediction(self):
        """Prediction has extra tokens → recall=1 but precision<1"""
        score = compute_f1_token("the big cat sat", "the cat sat")
        assert 0 < score < 1.0

    def test_subset_prediction(self):
        """Prediction missing tokens → precision=1 but recall<1"""
        score = compute_f1_token("the cat", "the cat sat")
        assert 0 < score < 1.0


class TestBatchMetrics:
    def test_perfect_batch(self):
        preds = ["yes", "no", "axial"]
        refs = ["yes", "no", "axial"]
        result = compute_batch_metrics(preds, refs)
        assert result["bleu_1"] == 100.0
        assert result["f1"] == 100.0
        assert result["num_samples"] == 3

    def test_mixed_batch(self):
        preds = ["yes", "cat"]
        refs = ["yes", "dog"]
        result = compute_batch_metrics(preds, refs)
        assert 0 < result["bleu_1"] < 100
        assert 0 < result["f1"] < 100
