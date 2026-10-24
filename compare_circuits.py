"""
Comparative Analysis: Clean (Trojan-free) vs. Trojan-Infected Hardware Circuits
Usage:
    ./.venv/bin/python3 compare_circuits.py [clean_graph_path] [trojan_graph_path]
"""

import sys
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

from gnn_model import MultiViewTrojanGNN, suspicious_region


def compare_pair(clean_path_str: str, trojan_path_str: str):
    clean_p = Path(clean_path_str)
    trojan_p = Path(trojan_path_str)

    if not clean_p.is_file() or not trojan_p.is_file():
        print(f"Error: Could not find files: {clean_p} or {trojan_p}")
        return

    g_clean = torch.load(clean_p, weights_only=False)
    g_trojan = torch.load(trojan_p, weights_only=False)

    print("=" * 80)
    print(f"🔬 COMPARATIVE ANALYSIS: CLEAN vs. TROJAN-INFECTED CIRCUIT")
    print("=" * 80)
    print(f"  Clean (Golden) Benchmark   : {clean_p.name}")
    print(f"  Trojan-Infected Benchmark  : {trojan_p.name}")
    print("-" * 80)

    # 1. Structural Comparison
    clean_nodes = g_clean.x.size(0)
    clean_edges = g_clean.edge_index.size(1)
    trojan_nodes = g_trojan.x.size(0)
    trojan_edges = g_trojan.edge_index.size(1)

    node_diff = trojan_nodes - clean_nodes
    edge_diff = trojan_edges - clean_edges

    print("📊 1. CIRCUIT TOPOLOGY & NETLIST METRICS:")
    print(f"  • Nodes (Gates + Nets)    : Clean = {clean_nodes:,}  |  Trojan = {trojan_nodes:,}  (Diff: +{node_diff} nodes)")
    print(f"  • Connections (Edges)     : Clean = {clean_edges:,}  |  Trojan = {trojan_edges:,}  (Diff: +{edge_diff} edges)")
    print(f"  • Module Types (Cell Lib) : Clean = {len(g_clean.module_types)}  |  Trojan = {len(g_trojan.module_types)}")
    extra_modules = set(g_trojan.module_types) - set(g_clean.module_types)
    if extra_modules:
        print(f"    ⚠️ Extra Gate Types Introduced by Trojan: {', '.join(extra_modules)}")
    else:
        print(f"    ℹ️ Trojan uses existing standard library cells (stealthy).")

    # 2. GNN Evaluation
    # Note: Using consistent hidden_dim across multi-views
    def eval_graph(data):
        torch.manual_seed(42)  # Fixed seed for consistent comparison weights
        model = MultiViewTrojanGNN(input_dim=data.x.size(1), hidden_dim=64, num_classes=2)
        for ckpt in [Path("runs/exp1/model_best.pt"), Path("runs/test_run/model_best.pt")]:
            if ckpt.is_file():
                try:
                    model.load_state_dict(torch.load(ckpt, weights_only=False, map_location="cpu"))
                    break
                except Exception:
                    pass
        model.eval()
        with torch.no_grad():
            node_logits, graph_logits = model(data)
            node_scores = torch.softmax(node_logits, dim=-1)[:, 1]
            graph_score = torch.softmax(graph_logits, dim=-1)[0, 1].item()
        return graph_score, node_scores

    score_clean, nscores_clean = eval_graph(g_clean)
    score_trojan, nscores_trojan = eval_graph(g_trojan)

    print("\n🧠 2. MULTI-VIEW GNN RISK ASSESSMENT:")
    print(f"  • Clean Circuit Trojan Risk Score   : {score_clean:.4f}")
    print(f"  • Trojan Circuit Risk Score        : {score_trojan:.4f}")
    if score_trojan > score_clean:
        print(f"  • Classification Shift              : 🔴 RISK INCREASED by +{(score_trojan - score_clean):.4f}")
    else:
        print(f"  • Classification Shift              : ⚪ Risk Score Difference: {(score_trojan - score_clean):.4f}")

    # 3. Top Suspicious Nodes in Infected Circuit
    print("\n🎯 3. LOCALIZATION: TOP SUSPICIOUS COMPONENTS IN INFECTED CIRCUIT:")
    regions = suspicious_region(g_trojan, nscores_trojan, top_k=5, hops=2)
    for rank, r in enumerate(regions, 1):
        is_trojan_named = "troj" in r["name"].lower() or "trigger" in r["name"].lower()
        tag = " 🚨 [TROJAN COMPONENT DETECTED]" if is_trojan_named else ""
        print(f"  [{rank}] {r['name']} ({r['type']}) — Suspicion Score: {r['score']:.4f}{tag}")
        troj_in_context = [c['name'] for c in r['context'] if "troj" in c['name'].lower() or "trigger" in c['name'].lower()]
        if troj_in_context:
            print(f"       -> Connected to Trojan Signals in 2-hop context: {', '.join(troj_in_context)}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    pairs = [
        ("graphs/TRIT-TC/c2670_T000/c2670_T000.v.pt", "graphs/TRIT-TC/c2670_T050/c2670_T050.v.pt"),
        ("graphs/TRIT-TC/c2670_T000.v.pt", "graphs/TRIT-TC/c2670_T050.v.pt"),
        ("graphs/TRIT-TC/c5315_T000/c5315_T000.v.pt", "graphs/TRIT-TC/c5315_T052/c5315_T052.v.pt"),
        ("graphs/TRIT-TC/s15850_T000/s15850_T000.v.pt", "graphs/TRIT-TC/s15850_T206/s15850_T206.v.pt"),
    ]

    if len(sys.argv) == 3:
        pairs = [(sys.argv[1], sys.argv[2])]

    for clean, trojan in pairs:
        if Path(clean).is_file() and Path(trojan).is_file():
            compare_pair(clean, trojan)

