"""Binary PRM metrics and Dev-threshold selection."""

from __future__ import annotations

from typing import Any, Dict, Sequence


def binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    predictions = [int(score >= threshold) for score in scores]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    bins = [[] for _ in range(10)]
    for label, score in zip(labels, scores):
        bins[min(int(score * 10), 9)].append((label, score))
    ece = sum(
        len(items) / len(labels)
        * abs(
            sum(score for _, score in items) / len(items)
            - sum(label for label, _ in items) / len(items)
        )
        for items in bins
        if items
    )
    return {
        "threshold": threshold,
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "per_class": {
            str(label): {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in (0, 1)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "auroc_correct": roc_auc_score(labels, scores),
        "auprc_correct": average_precision_score(labels, scores),
        "auprc_error": average_precision_score(
            [1 - label for label in labels], [1 - score for score in scores]
        ),
        "brier": brier_score_loss(labels, scores),
        "ece_10_bin": ece,
    }


def select_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Choose the Macro-F1-optimal threshold, preferring proximity to 0.5."""
    candidates = sorted(set([0.0, 0.5, 1.0, *scores]))
    best = None
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, threshold)
        key = (metrics["macro_f1"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold)
    return best[1]
