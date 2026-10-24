"""Figures for the report (matplotlib, no display needed)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def _col(history, key):
    return [row.get(key, float("nan")) for row in history]


def plot_history(history, out_dir):
    """Training curves: loss, node-level quality, graph-level quality per epoch."""
    out_dir = Path(out_dir)
    epochs = _col(history, "epoch")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.plot(epochs, _col(history, "train_loss"), label="train")
    ax.plot(epochs, _col(history, "val_loss"), label="validation")
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.set_ylabel("weighted cross-entropy"); ax.legend()

    ax = axes[1]
    for key, name in [("val_node_pr_auc", "PR-AUC"), ("val_node_roc_auc", "ROC-AUC"), ("val_node_f1", "F1")]:
        ax.plot(epochs, _col(history, key), label=name)
    ax.set_title("Node-level (validation)"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1.02); ax.legend()

    ax = axes[2]
    for key, name in [("val_graph_auc", "ROC-AUC (graph head)"),
                      ("val_graph_auc_from_nodes", "ROC-AUC (max node score)"),
                      ("val_graph_f1", "F1 (graph head)")]:
        ax.plot(epochs, _col(history, key), label=name)
    ax.set_title("Graph-level (validation)"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1.02); ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def _confusion(ax, m, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(title)


def plot_test_report(node_y, node_p, graph_y, graph_p, node_m, graph_m, out_dir):
    """ROC + PR curves (node and graph level) and confusion matrices on the test set."""
    out_dir = Path(out_dir)
    node_y, node_p = np.asarray(node_y), np.asarray(node_p)
    graph_y, graph_p = np.asarray(graph_y), np.asarray(graph_p)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, (name, y, p) in enumerate([("node", node_y, node_p), ("graph", graph_y, graph_p)]):
        ax_roc, ax_pr = axes[row, 0], axes[row, 1]
        if len(np.unique(y)) > 1:
            fpr, tpr, _ = roc_curve(y, p)
            ax_roc.plot(fpr, tpr)
            prec, rec, _ = precision_recall_curve(y, p)
            ax_pr.plot(rec, prec)
            ax_pr.axhline(y.mean(), ls="--", color="gray", label="random")
            ax_pr.legend()
        else:
            for ax in (ax_roc, ax_pr):
                ax.text(0.5, 0.5, "only one class in test set", ha="center", va="center")
        ax_roc.plot([0, 1], [0, 1], ls="--", color="gray")
        ax_roc.set_title(f"{name}-level ROC (test)"); ax_roc.set_xlabel("false positive rate"); ax_roc.set_ylabel("true positive rate")
        ax_pr.set_title(f"{name}-level precision-recall (test)"); ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision")

    _confusion(axes[0, 2], node_m, "Node confusion matrix (test)")
    _confusion(axes[1, 2], graph_m, "Graph confusion matrix (test)")

    fig.tight_layout()
    fig.savefig(out_dir / "test_report.png", dpi=150)
    plt.close(fig)
