"""Multi-view graph model and suspicious-region extraction for NeuroGuard-X."""

import torch
from torch import nn
from torch_geometric.nn import GATConv, RGCNConv, SAGEConv, global_mean_pool


class MultiViewTrojanGNN(nn.Module):
    """Combine relational, inductive, and attention-based graph views.

    The model returns node-level logits for candidate ranking and graph-level
    logits for design classification. It must be trained on labeled designs
    before either output can be interpreted as a calibrated probability.

    Module 2 change: the RGCN view now uses `data.edge_type` (5 relations:
    input/output pins, their reverses, unknown direction), so it can actually
    tell "gate drives net" apart from "net feeds gate".
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.2, num_relations=5):
        super().__init__()
        self.rgcn = RGCNConv(input_dim, hidden_dim, num_relations=num_relations)
        self.sage = SAGEConv(input_dim, hidden_dim)
        self.gat = GATConv(input_dim, hidden_dim, heads=1, concat=False)
        self.view_weights = nn.Parameter(torch.zeros(3))
        self.dropout = nn.Dropout(dropout)
        self.node_classifier = nn.Linear(hidden_dim, num_classes)
        self.graph_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        edge_index = data.edge_index
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:  # graphs built before Module 2
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        views = torch.stack((
            torch.relu(self.rgcn(data.x, edge_index, edge_type)),
            torch.relu(self.sage(data.x, edge_index)),
            torch.relu(self.gat(data.x, edge_index)),
        ))
        weights = torch.softmax(self.view_weights, dim=0)
        nodes = self.dropout(torch.sum(views * weights[:, None, None], dim=0))
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(nodes.size(0), dtype=torch.long, device=nodes.device)
        graphs = global_mean_pool(nodes, batch)
        return self.node_classifier(nodes), self.graph_classifier(graphs)


def suspicious_region(graph, node_scores, top_k=10, hops=2):
    """Return top-k node metadata and their k-hop graph context.

    `node_scores` are caller-provided Trojan-class scores (typically softmax
    probabilities after training). The returned scores are preserved verbatim.
    """
    if top_k < 1 or hops < 0:
        raise ValueError("top_k must be positive and hops must be non-negative")
    scores = node_scores.detach().flatten().cpu().tolist()
    if len(scores) != len(graph.node_names):
        raise ValueError("node score count must match graph node count")

    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
    adjacency = [[] for _ in graph.node_names]
    for left, right in graph.edge_index.t().detach().cpu().tolist():
        adjacency[left].append(right)
    regions = []
    for index in ranked:
        included = {index}
        frontier = {index}
        for _ in range(hops):
            frontier = {neighbor for node in frontier for neighbor in adjacency[node]} - included
            included.update(frontier)
        context = [
            {"index": neighbor, "name": graph.node_names[neighbor], "type": graph.node_types[neighbor]}
            for neighbor in sorted(included)
        ]
        regions.append({
            "index": index,
            "name": graph.node_names[index],
            "type": graph.node_types[index],
            "score": scores[index],
            "context": context,
        })
    return regions
