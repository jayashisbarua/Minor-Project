"""Convert a tree of Verilog sources into PyTorch Geometric graph files.

Two passes:
  1. parse every file and collect the set of gate/cell types (global vocabulary)
  2. build graphs using that shared vocabulary, attach graph labels, save .pt files
"""

import argparse
import json
from pathlib import Path

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Install torch before running this script.") from exc

from file_collector import collect_verilog_files
from verilog_parser import parse_verilog, extract_netlist
from graph_builder import build_graph, nx_to_pyg, build_vocab
from labels import graph_label_from_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", default="data", help="directory containing Verilog sources")
    parser.add_argument("--output-dir", default="graphs", help="directory for serialized graph files")
    parser.add_argument("--vocab", default=None,
                        help="reuse an existing vocab.json instead of building a new one "
                             "(use this when converting new designs for a trained model)")
    args = parser.parse_args()

    base_dir = Path(args.input_dir).expanduser().resolve()
    save_dir = Path(args.output_dir).expanduser().resolve()
    if not base_dir.is_dir():
        parser.error(f"input directory does not exist: {base_dir}")

    verilog_files = sorted(collect_verilog_files(str(base_dir)))
    print(f"Found {len(verilog_files)} Verilog files under {base_dir}")

    # ---- Pass 1: parse everything ------------------------------------------
    parsed, skipped = [], 0
    for idx, source in enumerate(verilog_files, 1):
        print(f"[{idx}/{len(verilog_files)}]", end=" ")
        netlist = extract_netlist(parse_verilog(source))
        if not netlist["connections"]:
            print("No instance port connections found; skipping.")
            skipped += 1
            continue
        parsed.append((source, netlist))

    # ---- Global vocabulary ---------------------------------------------------
    save_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = save_dir / "vocab.json"
    if args.vocab:
        vocab = json.loads(Path(args.vocab).read_text())
        print(f"Loaded vocabulary with {len(vocab)} entries from {args.vocab}")
    else:
        vocab = build_vocab(c[2] for _, nl in parsed for c in nl["connections"])
        print(f"Built global vocabulary with {len(vocab)} entries (incl. <UNK>)")
    vocab_path.write_text(json.dumps(vocab, indent=2))

    # ---- Pass 2: build + save graphs -----------------------------------------
    saved, label_counts = 0, {0: 0, 1: 0, -1: 0}
    for source, netlist in parsed:
        label = graph_label_from_path(source)
        label_counts[label] += 1
        graph = build_graph(netlist)
        pyg_graph = nx_to_pyg(graph, vocab, label=label)
        relative = Path(source).resolve().relative_to(base_dir)
        out_path = save_dir / relative.parent / f"{relative.name}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pyg_graph, out_path)
        saved += 1
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
              f"label={label}; saved {out_path}")

    print(f"\nFinished: {saved} graphs saved, {skipped} files skipped.")
    print(f"Labels -> clean: {label_counts[0]}, infected: {label_counts[1]}, unknown: {label_counts[-1]}")
    if label_counts[-1]:
        print("Note: 'unknown' graphs have y = -1; exclude them (or label them) before training.")


if __name__ == "__main__":
    main()
