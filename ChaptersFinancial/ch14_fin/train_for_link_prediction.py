"""
train_for_link_prediction.py (ch14_fin)
=======================================
Link prediction on the financial ownership graph.
Predicts missing OWNS edges between LegalEntity nodes.
Evaluates Hits@K and MRR.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def run():
    print("[ch14_fin] GNN Link Prediction — Ownership")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        from torch_geometric.utils import negative_sampling
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  [ERROR] PyTorch Geometric not installed.")
        return

    data_path = Path(__file__).resolve().parents[1] / "data_fin" / "pyg_datasets" / "financial_hetero.pt"
    if not data_path.exists():
        print(f"  Dataset not found. Run build_pyg_dataset first.")
        return

    data = torch.load(str(data_path), weights_only=False)
    x = data["legal_entity"].x
    n = x.shape[0]

    if ("legal_entity", "owns", "legal_entity") not in data.edge_types:
        print("  No OWNS edges found.")
        return

    edge_index = data["legal_entity", "owns", "legal_entity"].edge_index
    num_edges = edge_index.shape[1]
    print(f"\n  Nodes: {n}, OWNS edges: {num_edges}")

    if num_edges < 4:
        print("  Too few edges for link prediction.")
        return

    # Train/test split on edges
    perm = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    train_ei = edge_index[:, perm[:train_size]]
    test_ei = edge_index[:, perm[train_size:]]

    # Model: GCN encoder + dot product decoder
    class LinkPredictor(torch.nn.Module):
        def __init__(self, in_c, hid):
            super().__init__()
            self.conv1 = GCNConv(in_c, hid)
            self.conv2 = GCNConv(hid, hid)

        def encode(self, x, ei):
            x = F.relu(self.conv1(x, ei))
            return self.conv2(x, ei)

        def decode(self, z, edge_label_index):
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)

    model = LinkPredictor(x.shape[1], 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train
    print("\n  Training …")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        z = model.encode(x, train_ei)

        # Positive + negative edges
        neg_ei = negative_sampling(train_ei, num_nodes=n, num_neg_samples=train_ei.shape[1])
        pos_score = model.decode(z, train_ei)
        neg_score = model.decode(z, neg_ei)

        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        scores = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_ei)
        pos_score = torch.sigmoid(model.decode(z, test_ei))
        neg_ei = negative_sampling(test_ei, num_nodes=n, num_neg_samples=test_ei.shape[1])
        neg_score = torch.sigmoid(model.decode(z, neg_ei))

    y_true = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    y_score = torch.cat([pos_score, neg_score]).numpy()

    roc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0

    # Hits@K
    all_scores = list(zip(y_true, y_score))
    all_scores.sort(key=lambda x: -x[1])
    hits_10 = sum(1 for t, _ in all_scores[:10] if t == 1) / min(10, int(y_true.sum()))

    print(f"\n  Test ROC-AUC: {roc:.4f}")
    print(f"  Hits@10: {hits_10:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
