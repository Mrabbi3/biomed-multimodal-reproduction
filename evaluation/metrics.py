"""
Evaluation metrics for Medical VQA reproduction.

Paper (Section A.3, Performance on medical VQA):
"We used BLEU-1 and token-level F1 scores to assess the performance
 of Med-PaLM M. This is in contrast with many prior works which used
 a string-level accuracy evaluation metric."
"""

from typing import List, Dict
from collections import Counter


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization."""
    return text.lower().strip().split()


def compute_bleu1(prediction: str, reference: str) -> float:
    """
    Compute BLEU-1 (unigram precision) between prediction and reference.

    Args:
        prediction: Model-generated answer
        reference: Ground truth answer

    Returns:
        BLEU-1 score in [0, 1]
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)

    # Clipped counts: min of predicted count and reference count
    clipped = 0
    for token, count in pred_counts.items():
        clipped += min(count, ref_counts.get(token, 0))

    return clipped / len(pred_tokens)


def compute_f1_token(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score.

    This is the harmonic mean of token-level precision and recall,
    which the paper uses instead of string-level accuracy.

    Args:
        prediction: Model-generated answer
        reference: Ground truth answer

    Returns:
        F1 score in [0, 1]
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_batch_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute average metrics over a batch of predictions.

    Args:
        predictions: List of model outputs
        references: List of ground truth answers

    Returns:
        Dict with 'bleu_1' and 'f1' average scores
    """
    assert len(predictions) == len(references), "Length mismatch"

    bleu_scores = []
    f1_scores = []

    for pred, ref in zip(predictions, references):
        bleu_scores.append(compute_bleu1(pred, ref))
        f1_scores.append(compute_f1_token(pred, ref))

    return {
        "bleu_1": sum(bleu_scores) / len(bleu_scores) * 100,
        "f1": sum(f1_scores) / len(f1_scores) * 100,
        "num_samples": len(predictions),
    }
