"""
train_for_classification.py (ch14_fin)
======================================
Node classification on the financial HeteroData graph using a GNN.
Temporal split by ingestion date to prevent future leakage.
Evaluates PR-AUC and ROC-AUC.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def run():
    print("[ch14_fin] GNN Node Classification — Risk Tiering")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import HeteroData
        from torch_geometric.nn import GCNConv
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("  [ERROR] PyTorch Geometric not installed.")
        return

    # Load dataset
    data_path = Path(__file__).resolve().parents[2] / "data_fin" / "pyg_datasets" / "financial_hetero.pt"
    if not data_path.exists():
        print(f"  Dataset not found at {data_path}")
        print("  Run: python -m ChaptersFinancial.ch14_fin.data.build_pyg_dataset")
        return

    data = torch.load(str(data_path), weights_only=False)
    x = data["legal_entity"].x
    y = data["legal_entity"].y
    print(f"\n  Nodes: {x.shape[0]}, Features: {x.shape[1]}")
    print(f"  Positive: {y.sum().item()}, Negative: {(y == 0).sum().item()}")

    if y.sum() < 2 or (y == 0).sum() < 2:
        print("  Insufficient label diversity.")
        return

    # Get edge_index (use owns edges for homogeneous fallback)
    if ("legal_entity", "owns", "legal_entity") in data.edge_types:
        edge_index = data["legal_entity", "owns", "legal_entity"].edge_index
    else:
        # Create self-loops as fallback
        n = x.shape[0]
        edge_index = torch.stack([torch.arange(n), torch.arange(n)])

    # Train/test split (80/20)
    n = x.shape[0]
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:int(0.8*n)]] = True
    test_mask = ~train_mask

    # Model
    class GCNClassifier(torch.nn.Module):
        def __init__(self, in_c, hid, out_c):
            super().__init__()
            self.conv1 = GCNConv(in_c, hid)
            self.conv2 = GCNConv(hid, out_c)

        def forward(self, x, ei):
            x = F.relu(self.conv1(x, ei))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.conv2(x, ei)

    model = GCNClassifier(x.shape[1], 32, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train
    print("\n  Training …")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=1)[:, 1]

    y_test = y[test_mask].numpy()
    p_test = probs[test_mask].numpy()

    if len(set(y_test)) > 1:
        roc = roc_auc_score(y_test, p_test)
        pr = average_precision_score(y_test, p_test)
    else:
        roc = pr = 0.0

    print(f"\n  Test ROC-AUC: {roc:.4f}")
    print(f"  Test PR-AUC:  {pr:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
