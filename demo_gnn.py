"""
Demo and test script for inspecting generated hardware graphs and running GNN inference.
Usage:
    ./.venv/bin/python3 demo_gnn.py [path_to_graph_file.pt]
"""

import sys
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore")

from gnn_model import MultiViewTrojanGNN, suspicious_region


def analyze_graph(graph_path: str):
    path = Path(graph_path)
    if not path.is_file():
        print(f"Error: File not found: {path}")
        return

    print("=" * 65)
    print(f"Hardware Graph Analysis: {path.name}")
    print("=" * 65)

    # 1. Load Graph
    data = torch.load(path, weights_only=False)
    print(f"  • Total Circuit Nodes (Instances + Nets) : {data.x.size(0):,}")
    print(f"  • Total Connections (Edges)              : {data.edge_index.size(1):,}")
    print(f"  • Feature Width                          : {data.x.size(1)}")
    if hasattr(data, "module_types"):
        print(f"  • Distinct Module Types                  : {len(data.module_types)}")
        if len(data.module_types) <= 10:
            print(f"    Types: {', '.join(data.module_types)}")

    # 2. Initialize Multi-View GNN
    input_dim = data.x.size(1)
    hidden_dim = 64
    model = MultiViewTrojanGNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2)
    for ckpt in [Path("runs/exp1/model_best.pt"), Path("runs/test_run/model_best.pt")]:
        if ckpt.is_file():
            try:
                model.load_state_dict(torch.load(ckpt, weights_only=False, map_location="cpu"))
                print(f"  • Loaded trained checkpoint: {ckpt}")
                break
            except Exception:
                pass
    model.eval()

    # 3. Model Inference (Multi-View: RGCN + GraphSAGE + GAT)
    with torch.no_grad():
        node_logits, graph_logits = model(data)
        # Trojan probability score per node (class 1)
        node_trojan_scores = torch.softmax(node_logits, dim=-1)[:, 1]
        graph_trojan_prob = torch.softmax(graph_logits, dim=-1)[0, 1].item()

    print("\n" + "-" * 65)
    print("GNN Inference Results (Multi-View RGCN + SAGE + GAT):")
    print(f"  • Overall Design Trojan Risk Score : {graph_trojan_prob:.4f}")
    print("-" * 65)

    # 4. Extract Top Suspicious Hardware Regions (k-hop context)
    top_k = min(5, data.x.size(0))
    hops = 2
    regions = suspicious_region(data, node_trojan_scores, top_k=top_k, hops=hops)

    print(f"\nTop {top_k} Most Suspicious Nodes & Their {hops}-Hop Circuit Context:")
    for rank, r in enumerate(regions, 1):
        print(f"\n[{rank}] Node #{r['index']} ('{r['name']}', type: {r['type']})")
        print(f"    Suspicion Score : {r['score']:.4f}")
        print(f"    {hops}-Hop Neighborhood: {len(r['context'])} interconnected circuit components")
        sample_context = [f"{c['name']} ({c['type']})" for c in r['context'][:6]]
        if len(r['context']) > 6:
            sample_context.append(f"... +{len(r['context']) - 6} more")
        print(f"    Connected Components: {', '.join(sample_context)}")

    print("\n" + "=" * 65)
    print("Analysis complete.")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default sample graph
        target_path = "graphs/TRIT-TC/c2670_T000/c2670_T000.v.pt"
    analyze_graph(target_path)

