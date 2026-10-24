"""Evaluation metrics for NeuroGuard-X (pure numpy / scikit-learn, no torch).

Three levels:
  * binary_metrics  - works for graph-level (clean vs infected) and node-level (Trojan node vs not)
  * topk_metrics    - localization quality: does the top-k suspicious list contain real Trojan nodes?
  * best_f1_threshold - pick a decision threshold on validation data
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

NAN = float("nan")


def _has_both_classes(y):
    return len(np.unique(y)) > 1


def best_f1_threshold(y_true, y_prob):
    """Threshold that maximises F1 (use on VALIDATION data, then freeze it for test)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if not _has_both_classes(y_true):
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def binary_metrics(y_true, y_prob, threshold=0.5):
    """Standard binary classification metrics at a given threshold.

    ROC-AUC and PR-AUC do not depend on the threshold. For heavily imbalanced
    data (node-level) look at PR-AUC / precision / recall / F1, NOT accuracy.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    both = _has_both_classes(y_true)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else NAN,
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else NAN,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else NAN,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else NAN,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both else NAN,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def random_hit_at_k(n_nodes, n_trojan, k):
    """Probability that k randomly chosen nodes contain at least one Trojan node."""
    k = min(k, n_nodes)
    miss = 1.0
    for i in range(k):
        remaining_clean = n_nodes - n_trojan - i
        if remaining_clean <= 0:
            return 1.0
        miss *= remaining_clean / (n_nodes - i)
    return 1.0 - miss


def topk_metrics(per_graph, ks=(1, 5, 10, 20)):
    """Localization metrics over infected graphs.

    per_graph: list of (node_scores, node_labels) numpy arrays, one pair per graph.
    Only graphs that really contain Trojan nodes are used.

      hit@k        fraction of infected graphs whose top-k nodes contain >= 1 Trojan node
      precision@k  mean fraction of the top-k nodes that are Trojan nodes
      recall@k     mean fraction of a graph's Trojan nodes found within the top-k
      random_hit@k what hit@k would be for random guessing (the baseline to beat)
    """
    infected = [(np.asarray(s), np.asarray(l)) for s, l in per_graph if np.asarray(l).sum() > 0]
    out = {"num_infected_graphs": len(infected)}
    for k in ks:
        if not infected:
            out.update({f"hit@{k}": NAN, f"precision@{k}": NAN, f"recall@{k}": NAN, f"random_hit@{k}": NAN})
            continue
        hits, precs, recs, rnd = [], [], [], []
        for scores, labels in infected:
            kk = min(k, len(scores))
            top = np.argsort(-scores, kind="stable")[:kk]
            found = labels[top].sum()
            hits.append(1.0 if found > 0 else 0.0)
            precs.append(found / kk)
            recs.append(found / labels.sum())
            rnd.append(random_hit_at_k(len(scores), int(labels.sum()), kk))
        out[f"hit@{k}"] = float(np.mean(hits))
        out[f"precision@{k}"] = float(np.mean(precs))
        out[f"recall@{k}"] = float(np.mean(recs))
        out[f"random_hit@{k}"] = float(np.mean(rnd))
    return out
