"""Directed instance/net graph + Module 2 structural features."""

import math
from collections import deque

import networkx as nx

UNK = "<UNK>"

# Edge types (relations) used by the RGCN view.
REL_IN_FWD = 0    # net  -> gate   : the net feeds a gate input
REL_OUT_FWD = 1   # gate -> net    : the gate drives the net
REL_IN_REV = 2    # gate -> net    : reverse of REL_IN_FWD
REL_OUT_REV = 3   # net  -> gate   : reverse of REL_OUT_FWD
REL_UNKNOWN = 4   # direction could not be determined (added both ways)
NUM_RELATIONS = 5
_REVERSE = {REL_IN_FWD: REL_IN_REV, REL_OUT_FWD: REL_OUT_REV}

EXTRA_FEATURES = [
    "is_instance", "is_net",
    "log1p_in_degree", "log1p_out_degree", "log1p_loads",
    "dist_to_output_norm", "is_primary_output", "is_primary_input",
]


def build_vocab(module_names):
    """Global gate/cell vocabulary shared by ALL graphs. Last entry is <UNK>."""
    return sorted(set(module_names)) + [UNK]


def build_graph(netlist):
    """Build a directed graph from the output of verilog_parser.extract_netlist."""
    connections = netlist["connections"]
    pis, pos = netlist["primary_inputs"], netlist["primary_outputs"]
    multi_scope = len({c[0] for c in connections}) > 1

    def label(scope, name):
        return f"{scope}/{name}" if multi_scope else name

    D = nx.DiGraph()
    for scope, inst, module, port, signal, direction in connections:
        i = ("instance", scope, inst)
        n = ("net", scope, signal)
        if i not in D:
            D.add_node(i, kind="instance", name=label(scope, inst), module=module, pi=0, po=0)
        if n not in D:
            D.add_node(n, kind="net", name=label(scope, signal), module="",
                       pi=int((scope, signal) in pis), po=int((scope, signal) in pos))
        if direction == "input":
            D.add_edge(n, i, port=port, rel=REL_IN_FWD)
        elif direction == "output":
            D.add_edge(i, n, port=port, rel=REL_OUT_FWD)
        else:
            D.add_edge(n, i, port=port, rel=REL_UNKNOWN)
            D.add_edge(i, n, port=port, rel=REL_UNKNOWN)
    return D


def _distance_to_outputs(D):
    """Shortest directed hop count from every node to any primary-output net."""
    sinks = [n for n, a in D.nodes(data=True) if a["po"]]
    dist = {s: 0 for s in sinks}
    queue = deque(sinks)
    while queue:
        u = queue.popleft()
        for p in D.predecessors(u):
            if p not in dist:
                dist[p] = dist[u] + 1
                queue.append(p)
    return dist


def graph_to_arrays(D, vocab):
    """Pure-python conversion (no torch). Returns features, edges, edge types, names."""
    nodes = list(D.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    vocab_index = {m: i for i, m in enumerate(vocab)}
    unk = vocab_index[UNK]
    base = len(vocab)
    width = base + len(EXTRA_FEATURES)

    dist = _distance_to_outputs(D)
    max_d = max(dist.values(), default=0)

    features = []
    for node in nodes:
        a = D.nodes[node]
        row = [0.0] * width
        is_inst = a["kind"] == "instance"
        if is_inst:
            row[vocab_index.get(a["module"], unk)] = 1.0
            loads = sum(D.out_degree(s) for s in D.successors(node))
        else:
            loads = D.out_degree(node)
        d = dist.get(node)
        row[base:] = [
            1.0 if is_inst else 0.0,
            0.0 if is_inst else 1.0,
            math.log1p(D.in_degree(node)),
            math.log1p(D.out_degree(node)),
            math.log1p(loads),
            1.0 if d is None else d / (max_d + 1),  # unreachable -> 1.0
            float(a["po"]),
            float(a["pi"]),
        ]
        features.append(row)

    src, dst, etype = [], [], []
    for u, v, attrs in D.edges(data=True):
        rel = attrs["rel"]
        src.append(idx[u]); dst.append(idx[v]); etype.append(rel)
        if rel in _REVERSE:  # add the reverse edge with its own relation type
            src.append(idx[v]); dst.append(idx[u]); etype.append(_REVERSE[rel])
        # REL_UNKNOWN already exists in both directions, nothing to add

    return {
        "features": features,
        "edges": (src, dst),
        "edge_type": etype,
        "node_names": [D.nodes[n].get("name", str(n)) for n in nodes],
        "node_types": [D.nodes[n]["kind"] for n in nodes],
        "used_modules": sorted({D.nodes[n]["module"] for n in nodes} - {""}),
        "feature_names": [f"module={m}" for m in vocab] + EXTRA_FEATURES,
    }


def nx_to_pyg(D, vocab, label=-1):
    """Convert the directed graph to a PyTorch Geometric Data object.

    label: graph label (0 = clean, 1 = Trojan-infected, -1 = unknown).
    """
    import torch
    from torch_geometric.data import Data

    a = graph_to_arrays(D, vocab)
    width = len(a["feature_names"])
    x = torch.tensor(a["features"], dtype=torch.float) if a["features"] else torch.empty((0, width))
    edge_index = torch.tensor(a["edges"], dtype=torch.long)  # shape [2, E]
    edge_type = torch.tensor(a["edge_type"], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                y=torch.tensor([label], dtype=torch.long))
    data.node_names = a["node_names"]
    data.node_types = a["node_types"]
    data.module_types = a["used_modules"]  # gate types present in THIS graph
    data.vocab = vocab                     # global vocabulary (same for every graph)
    data.feature_names = a["feature_names"]
    return data
