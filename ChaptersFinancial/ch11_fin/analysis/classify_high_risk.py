"""
classify_high_risk.py
=====================
Binary classification: identify high-risk entities using Node2Vec embeddings
+ handcrafted graph features. Compares embedding-only vs features-only vs
combined approaches.

Uses entity properties and graph structure as a proxy for risk labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch11_fin] High-Risk Entity Classification")
    print("=" * 60)

    gp = GraphProvider()

    # Load entities with features
    entities = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.node2vecEmbedding IS NOT NULL
        OPTIONAL MATCH (le)-[r:OWNS|CONTROLS|PARENT_OF]-()
        WITH le, count(r) AS degree
        OPTIONAL MATCH (le)-[:CLASSIFIED_AS]->(oc:OntologyClass)
        RETURN le.lei AS lei, le.name AS name,
               le.node2vecEmbedding AS embedding,
               le.pagerank AS pagerank,
               le.communityLouvain AS community,
               degree,
               le.status AS status,
               oc.label AS sectorLabel
    """)
    print(f"\n  Entities: {len(entities)}")

    if len(entities) < 10:
        print("  Need more entities. Run earlier chapters first.")
        gp.close()
        return

    # Create feature matrix
    X_emb = np.array([e["embedding"] for e in entities])
    X_feat = np.array([
        [
            e.get("pagerank") or 0.0,
            e.get("degree") or 0,
            1.0 if e.get("status") == "INACTIVE" else 0.0,
        ]
        for e in entities
    ])
    X_combined = np.hstack([X_emb, X_feat])

    # Proxy labels: high degree or INACTIVE status = higher risk
    y = np.array([
        1 if (e.get("degree", 0) > 5 or e.get("status") == "INACTIVE") else 0
        for e in entities
    ])

    print(f"  Positive labels (high-risk proxy): {y.sum()}/{len(y)}")

    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        print("  Insufficient label diversity for classification. Skipping.")
        gp.close()
        return

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    results = {}
    for name, X in [("Embedding-only", X_emb), ("Features-only", X_feat), ("Combined", X_combined)]:
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(lr, X, y, cv=min(3, len(y)//2), scoring="roc_auc")
            results[name] = {"model": "LogReg", "roc_auc": scores.mean()}
        except Exception as exc:
            results[name] = {"model": "LogReg", "error": str(exc)}

    print("\n  Classification Results (3-fold CV ROC-AUC):")
    print(f"  {'Feature Set':<20s} {'ROC-AUC':>10s}")
    print(f"  {'-'*32}")
    for name, res in results.items():
        if "error" in res:
            print(f"  {name:<20s} {'ERROR':>10s}")
        else:
            print(f"  {name:<20s} {res['roc_auc']:>10.3f}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
