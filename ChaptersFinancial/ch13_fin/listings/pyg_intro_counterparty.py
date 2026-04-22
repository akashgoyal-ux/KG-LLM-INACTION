"""
pyg_intro_counterparty.py
=========================
PyTorch Geometric introduction using the real financial entity-ownership graph
from Neo4j. Builds a PyG Data object from the ownership network and runs
a simple GCN for node classification.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def _load_graph_from_neo4j():
    """Load the entity-ownership graph from Neo4j into numpy arrays."""
    gp = GraphProvider()

    # Get nodes with features
    nodes = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.name IS NOT NULL
        RETURN le.lei AS lei,
               coalesce(le.feat_totalDegree, 0) AS degree,
               coalesce(le.pagerank, 0.0) AS pagerank,
               coalesce(le.feat_filingCount, 0) AS filings,
               coalesce(le.feat_mentionCount, 0) AS mentions,
               le.status AS status
        ORDER BY le.lei
    """)

    # Get edges
    edges = gp.run("""
        MATCH (a:LegalEntity)-[:OWNS|CONTROLS|PARENT_OF]->(b:LegalEntity)
        RETURN a.lei AS source, b.lei AS target
    """)

    gp.close()
    return nodes, edges


def run():
    print("[ch13_fin] PyG Introduction — Counterparty Graph")
    print("=" * 60)

    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        import torch.nn.functional as F
    except ImportError:
        print("  [ERROR] PyTorch Geometric not installed.")
        print("  Run: pip install torch torch-geometric")
        return

    print("\n1. Loading graph from Neo4j …")
    nodes, edges = _load_graph_from_neo4j()
    print(f"   Nodes: {len(nodes)}, Edges: {len(edges)}")

    if len(nodes) < 5 or len(edges) < 2:
        print("   Graph too small. Run earlier chapters first.")
        return

    # Build node index
    lei_to_idx = {n["lei"]: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Feature matrix
    X = np.array([
        [n["degree"], n["pagerank"], n["filings"], n["mentions"]]
        for n in nodes
    ], dtype=np.float32)

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    # Edge index
    src_idx = [lei_to_idx[e["source"]] for e in edges if e["source"] in lei_to_idx and e["target"] in lei_to_idx]
    tgt_idx = [lei_to_idx[e["target"]] for e in edges if e["source"] in lei_to_idx and e["target"] in lei_to_idx]

    # Labels (proxy: high-degree = positive)
    y = np.array([1 if n["degree"] > 3 else 0 for n in nodes])

    # Create PyG Data object
    x = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor([src_idx + tgt_idx, tgt_idx + src_idx], dtype=torch.long)  # undirected
    labels = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=labels)
    print(f"\n2. PyG Data: {data}")

    # Simple GCN
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, out_channels)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    model = GCN(X.shape[1], 16, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train
    print("\n3. Training GCN …")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            model.eval()
            pred = out.argmax(dim=1)
            acc = (pred == data.y).float().mean()
            print(f"   Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
            model.train()

    # Final eval
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred == data.y).float().mean()
    print(f"\n   Final accuracy: {acc:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
