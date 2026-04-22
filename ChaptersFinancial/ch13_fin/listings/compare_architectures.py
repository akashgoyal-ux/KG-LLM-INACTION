"""
compare_architectures.py
========================
Side-by-side comparison of GCN, GAT, GraphSAGE, and GIN architectures
on the financial entity-ownership graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch13_fin] GNN Architecture Comparison")
    print("=" * 60)

    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
        import torch.nn.functional as F
        from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
    except ImportError:
        print("  [ERROR] PyTorch Geometric not installed.")
        return

    # Load graph
    gp = GraphProvider()
    nodes = gp.run("""
        MATCH (le:LegalEntity) WHERE le.name IS NOT NULL
        RETURN le.lei AS lei,
               coalesce(le.feat_totalDegree, 0) AS degree,
               coalesce(le.pagerank, 0.0) AS pagerank,
               coalesce(le.feat_filingCount, 0) AS filings,
               coalesce(le.feat_mentionCount, 0) AS mentions
        ORDER BY le.lei
    """)
    edges = gp.run("""
        MATCH (a:LegalEntity)-[:OWNS|CONTROLS|PARENT_OF]->(b:LegalEntity)
        RETURN a.lei AS source, b.lei AS target
    """)
    gp.close()

    if len(nodes) < 5 or len(edges) < 2:
        print("  Graph too small.")
        return

    lei_to_idx = {n["lei"]: i for i, n in enumerate(nodes)}
    X = np.array([[n["degree"], n["pagerank"], n["filings"], n["mentions"]]
                  for n in nodes], dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = np.array([1 if n["degree"] > 3 else 0 for n in nodes])

    src = [lei_to_idx[e["source"]] for e in edges
           if e["source"] in lei_to_idx and e["target"] in lei_to_idx]
    tgt = [lei_to_idx[e["target"]] for e in edges
           if e["source"] in lei_to_idx and e["target"] in lei_to_idx]

    x_t = torch.tensor(X, dtype=torch.float)
    ei = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
    y_t = torch.tensor(y, dtype=torch.long)
    data = Data(x=x_t, edge_index=ei, y=y_t)

    in_c, hid, out_c = X.shape[1], 16, 2

    # Define architectures
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_c, hid)
            self.conv2 = GCNConv(hid, out_c)
        def forward(self, x, ei):
            return F.log_softmax(self.conv2(F.relu(self.conv1(x, ei)), ei), dim=1)

    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(in_c, hid, heads=2, concat=False)
            self.conv2 = GATConv(hid, out_c, heads=1, concat=False)
        def forward(self, x, ei):
            return F.log_softmax(self.conv2(F.relu(self.conv1(x, ei)), ei), dim=1)

    class SAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(in_c, hid)
            self.conv2 = SAGEConv(hid, out_c)
        def forward(self, x, ei):
            return F.log_softmax(self.conv2(F.relu(self.conv1(x, ei)), ei), dim=1)

    class GIN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            nn1 = Sequential(Linear(in_c, hid), ReLU(), Linear(hid, hid))
            nn2 = Sequential(Linear(hid, hid), ReLU(), Linear(hid, out_c))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        def forward(self, x, ei):
            return F.log_softmax(self.conv2(F.relu(self.conv1(x, ei)), ei), dim=1)

    # Train and evaluate each
    results = {}
    for name, ModelClass in [("GCN", GCN), ("GAT", GAT), ("GraphSAGE", SAGE), ("GIN", GIN)]:
        model = ModelClass()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=1)
        acc = (pred == data.y).float().mean().item()
        results[name] = acc

    print(f"\n  {'Architecture':<15s} {'Accuracy':>10s}")
    print(f"  {'-'*27}")
    for name, acc in results.items():
        print(f"  {name:<15s} {acc:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
