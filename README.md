# NeuroGuard-X: Hardware Trojan Detection Pipeline

NeuroGuard-X converts digital integrated circuit (IC) designs specified in Verilog RTL / Netlists into **PyTorch Geometric bipartite graphs** and processes them through a **Multi-View Graph Neural Network (GNN)** to rank suspicious hardware components and localize stealthy Hardware Trojans.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone or enter the project directory and create an isolated virtual environment:

```bash
cd "Minor project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Optional for Verilog macro preprocessing)*:
```bash
brew install icarus-verilog   # On macOS
# or sudo apt-get install iverilog on Ubuntu/Debian
```

---

## 🛠️ Pipeline Execution

### Step 1: Convert Verilog Designs into PyG Circuit Graphs
Run `main.py` to batch-process Verilog designs into `.pt` graph files:

```bash
# Convert a specific benchmark directory
./.venv/bin/python3 main.py "data/TRIT-TC" --output-dir "graphs"

# Or convert the entire dataset
./.venv/bin/python3 main.py data --output-dir graphs
```

Each circuit is mapped to a bipartite graph:
* **Instances** (logic gates, modules, flip-flops)
* **Nets** (wires, buses, clock/reset signals)
* **Features**: Module type one-hot encoding, instance/net kind, and normalized topological degree.

---

### Step 2: Run GNN Trojan Localization
Analyze any generated hardware graph using the multi-view GNN model:

```bash
# Run Trojan detection and 2-hop context extraction on a circuit graph
./.venv/bin/python3 demo_gnn.py "graphs/TRIT-TC/c2670_T000/c2670_T000.v.pt"
```

The script will output:
1. **Total Circuit Nodes & Connections (Edges)**
2. **Overall Design Trojan Risk Score**
3. **Top-k Most Suspicious Nodes & Their 2-Hop Circuit Context** (identifying the interconnected gates where triggers or payloads reside).

---

## 📂 Project Architecture

| File | Purpose |
| :--- | :--- |
| [`file_collector.py`](file:///Users/jayashisbarua/Downloads/Minor%20project/file_collector.py) | Recursively searches and collects `.v` Verilog source files. |
| [`verilog_parser.py`](file:///Users/jayashisbarua/Downloads/Minor%20project/verilog_parser.py) | Parses Verilog ASTs with include-path preprocessor resolution. |
| [`graph_builder.py`](file:///Users/jayashisbarua/Downloads/Minor%20project/graph_builder.py) | Constructs NetworkX bipartite graphs and converts to PyTorch Geometric `Data`. |
| [`main.py`](file:///Users/jayashisbarua/Downloads/Minor%20project/main.py) | CLI tool orchestrating batch graph extraction and serialization. |
| [`gnn_model.py`](file:///Users/jayashisbarua/Downloads/Minor%20project/gnn_model.py) | Defines `MultiViewTrojanGNN` (combining RGCN, GraphSAGE, GAT) and `suspicious_region`. |
| [`demo_gnn.py`](file:///Users/jayashisbarua/Downloads/Minor%20project/demo_gnn.py) | Inference runner for loading graphs, calculating scores, and extracting subcircuit contexts. |
| [`requirements.txt`](file:///Users/jayashisbarua/Downloads/Minor%20project/requirements.txt) | Python dependencies required to run the pipeline. |

---

## 🧠 Model Architecture: Multi-View GNN
The model in [`gnn_model.py`](file:///Users/jayashisbarua/Downloads/Minor%20project/gnn_model.py) utilizes three complementary views:
1. **RGCN (Relational GCN)**: Captures typed relationship patterns between instances and nets.
2. **GraphSAGE**: Aggregates local neighborhood structural contexts.
3. **GAT (Graph Attention Networks)**: Uses self-attention to focus on atypical connection distributions characteristic of stealthy Trojan triggers.

The three views are fused with **learnable softmax view weights**, yielding both graph-level classification and node-level localization.
