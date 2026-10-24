"""Add node-level Trojan labels (`node_y`) to the saved graphs.

Method: for an infected graph  X_T0nn , find its golden twin  X_T000  and mark every
node (gate or net) whose (type, name) does NOT exist in the golden circuit as Trojan.
Clean graphs get all zeros.

Names are used ONLY to create labels here. They are never used as model features.

Usage:
    python label_nodes.py graphs
    python label_nodes.py graphs --pattern "troj|trigger"   # fallback when no golden twin exists
"""

import argparse
import re
from pathlib import Path

import torch


def golden_path(pt_path: Path):
    m = re.match(r"(.+)_T\d+\.v\.pt$", pt_path.name, re.IGNORECASE)
    if not m:
        return None
    base = m.group(1)
    # Check nested hierarchy: graphs/TRIT-TC/c2670_T000/c2670_T000.v.pt
    nested = pt_path.parent.parent / f"{base}_T000" / f"{base}_T000.v.pt"
    if nested.is_file():
        return nested
    # Check same directory flat structure: graphs/TRIT-TC/c2670_T000.v.pt
    flat = pt_path.parent / f"{base}_T000.v.pt"
    if flat.is_file():
        return flat
    # Fallback search anywhere in the root directory
    for candidate in pt_path.parents[-1].rglob(f"{base}_T000.v.pt"):
        if candidate.is_file():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs_dir", nargs="?", default="graphs")
    ap.add_argument("--pattern", default=None,
                    help="regex on node names, used only if no golden twin is found")
    args = ap.parse_args()

    done = warned = 0
    for pt in sorted(Path(args.graphs_dir).rglob("*.pt")):
        data = torch.load(pt, weights_only=False)
        y = int(data.y.item()) if getattr(data, "y", None) is not None else -1
        n = data.x.size(0)
        if y == -1:
            continue

        if y == 0:
            node_y = torch.zeros(n, dtype=torch.long)
        else:
            gold = golden_path(pt)
            if gold is not None and gold.is_file():
                g = torch.load(gold, weights_only=False)
                known = set(zip(g.node_types, g.node_names))
                flags = [0 if (t, nm) in known else 1 for t, nm in zip(data.node_types, data.node_names)]
            elif args.pattern:
                rx = re.compile(args.pattern, re.IGNORECASE)
                flags = [1 if rx.search(nm) else 0 for nm in data.node_names]
            else:
                print(f"⚠️  {pt.name}: no golden twin found and no --pattern given; skipped")
                warned += 1
                continue
            node_y = torch.tensor(flags, dtype=torch.long)
            frac = node_y.sum().item() / max(1, n)
            sample = [data.node_names[i] for i in node_y.nonzero().flatten().tolist()[:5]]
            print(f"{pt.name}: {node_y.sum().item()} / {n} nodes labelled Trojan  e.g. {sample}")
            if frac > 0.25:
                print(f"   ⚠️ {frac:.0%} of nodes flagged - names probably changed between versions; check this one")

        data.node_y = node_y
        torch.save(data, pt)
        done += 1

    print(f"\nLabelled {done} graphs ({warned} skipped with warnings).")


if __name__ == "__main__":
    main()
