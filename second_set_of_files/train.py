"""Train the multi-view GNN on the labelled graphs and report metrics.

Usage:
    python train.py graphs --out-dir runs/exp1
    python train.py graphs --out-dir runs/exp2 --epochs 100 --lr 5e-4 --node-weight 2.0

Outputs (in --out-dir):
    history.csv            per-epoch losses and validation metrics
    training_curves.png    loss / node-level / graph-level curves
    metrics.json           final metrics on train / val / test + thresholds + config
    test_predictions.csv   one row per test graph (score, hit@k, rank of first Trojan node)
    test_report.png        ROC, precision-recall, confusion matrices on the test set
    model_best.pt          best checkpoint (by validation node PR-AUC)
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_model import MultiViewTrojanGNN
from metrics import best_f1_threshold, binary_metrics, topk_metrics
from plots import plot_history, plot_test_report
from splits import assign_splits

KS = (1, 5, 10, 20)


# ----------------------------------------------------------------------------- data
def load_dataset(graphs_dir):
    """Load every labelled graph; keep only the tensors needed for training."""
    graphs_dir = Path(graphs_dir)
    data_list, skipped = [], {"no_label": 0, "no_node_labels": 0, "old_format": 0}
    for pt in sorted(graphs_dir.rglob("*.pt")):
        d = torch.load(pt, weights_only=False)
        if getattr(d, "y", None) is None or int(d.y.view(-1)[0]) < 0:
            skipped["no_label"] += 1
            continue
        if getattr(d, "node_y", None) is None:
            skipped["no_node_labels"] += 1
            continue
        if getattr(d, "edge_type", None) is None:
            skipped["old_format"] += 1
            continue
        data_list.append(Data(
            x=d.x, edge_index=d.edge_index, edge_type=d.edge_type,
            y=d.y.view(-1).long(), node_y=d.node_y.long(),
            name=str(pt.relative_to(graphs_dir)),
        ))
    print(f"Loaded {len(data_list)} labelled graphs. Skipped: {skipped}")
    if skipped["no_node_labels"]:
        print("  -> run label_nodes.py first to add node-level labels")
    if skipped["old_format"]:
        print("  -> re-run main.py (Module 2) to regenerate graphs with edge types")
    return data_list


def make_weights(train_data, max_pos_weight, device):
    """Class weights, because both levels are imbalanced (esp. Trojan nodes)."""
    g = torch.tensor([int(d.y.item()) for d in train_data])
    n0, n1 = int((g == 0).sum()), int((g == 1).sum())
    graph_w = torch.tensor([(n0 + n1) / (2 * max(n0, 1)), (n0 + n1) / (2 * max(n1, 1))])
    nodes = torch.cat([d.node_y for d in train_data])
    pos = int(nodes.sum())
    neg = nodes.numel() - pos
    node_pos_w = min(neg / max(pos, 1), max_pos_weight)
    node_w = torch.tensor([1.0, node_pos_w])
    info = {"train_clean_graphs": n0, "train_infected_graphs": n1,
            "train_trojan_nodes": pos, "train_total_nodes": nodes.numel(),
            "node_pos_weight": node_pos_w, "graph_weights": graph_w.tolist()}
    return graph_w.float().to(device), node_w.float().to(device), info


# ------------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, loader, device, node_w, graph_w, node_weight):
    model.eval()
    loss_sum, n_graphs = 0.0, 0
    out = {"graph_y": [], "graph_p": [], "graph_pmax": [], "node_y": [], "node_p": [],
           "per_graph": [], "names": []}
    for batch in loader:
        batch = batch.to(device)
        node_logits, graph_logits = model(batch)
        nloss = F.cross_entropy(node_logits, batch.node_y, weight=node_w)
        gloss = F.cross_entropy(graph_logits, batch.y, weight=graph_w)
        loss_sum += float(gloss + node_weight * nloss) * batch.num_graphs
        n_graphs += batch.num_graphs

        node_p = torch.softmax(node_logits, dim=-1)[:, 1].cpu().numpy()
        node_y = batch.node_y.cpu().numpy()
        ptr = batch.ptr.cpu().numpy()
        for i in range(batch.num_graphs):
            s, l = node_p[ptr[i]:ptr[i + 1]], node_y[ptr[i]:ptr[i + 1]]
            out["per_graph"].append((s, l))
            out["graph_pmax"].append(float(s.max()) if len(s) else 0.0)
            out["names"].append(batch.name[i])
        out["graph_y"] += batch.y.cpu().tolist()
        out["graph_p"] += torch.softmax(graph_logits, dim=-1)[:, 1].cpu().tolist()
        out["node_y"].append(node_y)
        out["node_p"].append(node_p)
    out["loss"] = loss_sum / max(n_graphs, 1)
    out["node_y"] = np.concatenate(out["node_y"]) if out["node_y"] else np.array([])
    out["node_p"] = np.concatenate(out["node_p"]) if out["node_p"] else np.array([])
    return out


def summarize(ev, thr):
    """All metrics for one evaluation pass, with frozen thresholds."""
    return {
        "loss": ev["loss"],
        "graph_head": binary_metrics(ev["graph_y"], ev["graph_p"], thr["graph"]),
        "graph_from_nodes": binary_metrics(ev["graph_y"], ev["graph_pmax"], thr["graph_from_nodes"]),
        "node": binary_metrics(ev["node_y"], ev["node_p"], thr["node"]),
        "localization": topk_metrics(ev["per_graph"], KS),
    }


def clean(obj):
    """Make nested metric dicts JSON-friendly."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def print_block(title, m):
    g, gn, n, loc = m["graph_head"], m["graph_from_nodes"], m["node"], m["localization"]
    print(f"\n=== {title} ===")
    print(f"Graph-level (graph head)     acc {g['accuracy']:.3f} | bal-acc {g['balanced_accuracy']:.3f} | "
          f"P {g['precision']:.3f} | R {g['recall']:.3f} | F1 {g['f1']:.3f} | ROC-AUC {g['roc_auc']:.3f}")
    print(f"Graph-level (max node score) F1 {gn['f1']:.3f} | ROC-AUC {gn['roc_auc']:.3f}")
    print(f"Node-level                   P {n['precision']:.3f} | R {n['recall']:.3f} | F1 {n['f1']:.3f} | "
          f"ROC-AUC {n['roc_auc']:.3f} | PR-AUC {n['pr_auc']:.3f} (random = {n['positive_rate']:.4f})")
    print("Localization (infected graphs: {})".format(loc["num_infected_graphs"]))
    for k in KS:
        print(f"   top-{k:<2d}  hit {loc[f'hit@{k}']:.3f}  (random {loc[f'random_hit@{k}']:.3f}) | "
              f"precision {loc[f'precision@{k}']:.3f} | recall {loc[f'recall@{k}']:.3f}")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs_dir", nargs="?", default="graphs")
    ap.add_argument("--out-dir", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience (epochs)")
    ap.add_argument("--node-weight", type=float, default=1.0, help="weight of node loss vs graph loss")
    ap.add_argument("--max-pos-weight", type=float, default=50.0, help="cap for Trojan-node class weight")
    ap.add_argument("--split", choices=["group", "random"], default="group")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    data_list = load_dataset(args.graphs_dir)
    if not data_list:
        raise SystemExit("No usable graphs found.")

    split_names = assign_splits([(d.name, int(d.y.item())) for d in data_list], args.split, seed=args.seed)
    parts = {s: [d for d, n in zip(data_list, split_names) if n == s] for s in ("train", "val", "test")}
    print(f"\nSplit mode: {args.split}")
    for s, ds in parts.items():
        inf = sum(int(d.y.item()) for d in ds)
        troj = sum(int(d.node_y.sum()) for d in ds)
        print(f"  {s:5s}: {len(ds):4d} graphs ({len(ds) - inf} clean, {inf} infected), {troj} Trojan nodes")
    if any(len(v) == 0 for v in parts.values()):
        raise SystemExit("A split is empty - not enough circuit families for this split mode.")
    if args.split == "random":
        print("  WARNING: random split puts variants of the same circuit in train AND test (leaky).")

    loaders = {
        "train": DataLoader(parts["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(parts["val"], batch_size=args.batch_size),
        "test": DataLoader(parts["test"], batch_size=args.batch_size),
        "train_eval": DataLoader(parts["train"], batch_size=args.batch_size),
    }

    graph_w, node_w, weight_info = make_weights(parts["train"], args.max_pos_weight, device)
    print(f"\nClass weights: {weight_info}")

    model = MultiViewTrojanGNN(input_dim=data_list[0].x.size(1), hidden_dim=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    history, best_score, best_state, bad_epochs = [], -float("inf"), None, 0
    neutral = {"graph": 0.5, "graph_from_nodes": 0.5, "node": 0.5}

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = nloss_sum = gloss_sum = 0.0
        n_graphs = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            node_logits, graph_logits = model(batch)
            nloss = F.cross_entropy(node_logits, batch.node_y, weight=node_w)
            gloss = F.cross_entropy(graph_logits, batch.y, weight=graph_w)
            loss = gloss + args.node_weight * nloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            b = batch.num_graphs
            loss_sum += float(loss) * b; nloss_sum += float(nloss) * b; gloss_sum += float(gloss) * b
            n_graphs += b

        ev = evaluate(model, loaders["val"], device, node_w, graph_w, args.node_weight)
        node_thr = best_f1_threshold(ev["node_y"], ev["node_p"])
        vm = {
            "node": binary_metrics(ev["node_y"], ev["node_p"], node_thr),
            "graph": binary_metrics(ev["graph_y"], ev["graph_p"], neutral["graph"]),
            "from_nodes": binary_metrics(ev["graph_y"], ev["graph_pmax"],
                                         best_f1_threshold(ev["graph_y"], ev["graph_pmax"])),
            "loc": topk_metrics(ev["per_graph"], KS),
        }
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / n_graphs, "train_node_loss": nloss_sum / n_graphs,
            "train_graph_loss": gloss_sum / n_graphs, "val_loss": ev["loss"],
            "val_node_pr_auc": vm["node"]["pr_auc"], "val_node_roc_auc": vm["node"]["roc_auc"],
            "val_node_f1": vm["node"]["f1"],
            "val_graph_acc": vm["graph"]["accuracy"], "val_graph_f1": vm["graph"]["f1"],
            "val_graph_auc": vm["graph"]["roc_auc"],
            "val_graph_auc_from_nodes": vm["from_nodes"]["roc_auc"],
            "val_hit@5": vm["loc"]["hit@5"], "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f} | "
              f"node PR-AUC {row['val_node_pr_auc']:.3f} F1 {row['val_node_f1']:.3f} | "
              f"graph AUC {row['val_graph_auc']:.3f} | hit@5 {row['val_hit@5']:.3f} | {row['seconds']:.0f}s")

        score = row["val_node_pr_auc"]
        if math.isnan(score):
            score = -row["val_loss"]
        if score > best_score + 1e-4:
            best_score, bad_epochs = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    # ---- history + curves
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    plot_history(history, out_dir)

    # ---- final evaluation with the best checkpoint; thresholds frozen from validation
    model.load_state_dict(best_state)
    evs = {s: evaluate(model, loaders[k], device, node_w, graph_w, args.node_weight)
           for s, k in (("train", "train_eval"), ("val", "val"), ("test", "test"))}
    thr = {
        "node": best_f1_threshold(evs["val"]["node_y"], evs["val"]["node_p"]),
        "graph": best_f1_threshold(evs["val"]["graph_y"], evs["val"]["graph_p"]),
        "graph_from_nodes": best_f1_threshold(evs["val"]["graph_y"], evs["val"]["graph_pmax"]),
    }
    results = {s: summarize(evs[s], thr) for s in evs}
    for s in ("train", "val", "test"):
        print_block(f"{s.upper()}" + (" (unseen circuit families)" if s == "test" and args.split == "group" else ""),
                    results[s])

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(clean({"config": vars(args), "weights": weight_info, "thresholds": thr,
                         "best_val_node_pr_auc": best_score, **results}), f, indent=2)

    # ---- per-graph test predictions
    ev = evs["test"]
    with open(out_dir / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "true_label", "graph_head_prob", "max_node_prob", "nodes", "trojan_nodes",
                    "first_trojan_rank", "hit@1", "hit@5", "hit@10"])
        for name, yt, gp, gm, (s, l) in zip(ev["names"], ev["graph_y"], ev["graph_p"],
                                             ev["graph_pmax"], ev["per_graph"]):
            order = np.argsort(-s, kind="stable")
            hits = np.nonzero(l[order])[0]
            first = int(hits[0]) + 1 if len(hits) else ""
            w.writerow([name, yt, f"{gp:.4f}", f"{gm:.4f}", len(s), int(l.sum()), first,
                        *[int(first != "" and first <= k) for k in (1, 5, 10)]])

    plot_test_report(ev["node_y"], ev["node_p"], ev["graph_y"], ev["graph_p"],
                     results["test"]["node"], results["test"]["graph_head"], out_dir)
    torch.save({"state_dict": best_state, "thresholds": thr, "config": vars(args),
                "input_dim": data_list[0].x.size(1)}, out_dir / "model_best.pt")
    print(f"\nSaved everything to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
