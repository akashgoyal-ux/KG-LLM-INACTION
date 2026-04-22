"""
cluster_issuers.py
==================
Cluster LegalEntity nodes using Node2Vec embeddings (from node2vec_issuers.py).
Applies KMeans clustering with silhouette score evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch11_fin] Cluster Issuers")
    print("=" * 60)

    gp = GraphProvider()

    # Load embeddings
    entities = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.node2vecEmbedding IS NOT NULL
        RETURN le.lei AS lei, le.name AS name,
               le.node2vecEmbedding AS embedding,
               le.jurisdiction AS jurisdiction
    """)
    print(f"\n  Entities with embeddings: {len(entities)}")

    if len(entities) < 5:
        print("  Need at least 5 entities. Run node2vec_issuers first.")
        gp.close()
        return

    leis = [e["lei"] for e in entities]
    names = [e["name"] for e in entities]
    X = np.array([e["embedding"] for e in entities])

    # KMeans clustering
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_score = 2, -1
    for k in range(2, min(10, len(entities))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        if score > best_score:
            best_k, best_score = k, score

    print(f"\n  Best k={best_k} (silhouette={best_score:.3f})")

    # Final clustering with best k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Write cluster labels back to Neo4j
    for lei, label in zip(leis, labels):
        gp.run(
            "MATCH (le:LegalEntity {lei: $lei}) SET le.clusterNode2vec = $cluster",
            {"lei": lei, "cluster": int(label)},
        )

    # Report clusters
    print(f"\n  Cluster assignments:")
    from collections import Counter
    cluster_counts = Counter(labels)
    for cluster_id, count in sorted(cluster_counts.items()):
        members = [names[i] for i in range(len(labels)) if labels[i] == cluster_id]
        print(f"    Cluster {cluster_id}: {count} entities")
        for m in members[:5]:
            print(f"      - {m}")
        if count > 5:
            print(f"      … and {count - 5} more")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
