"""
peer_search_by_embedding.py
============================
Find peer entities using cosine similarity over profile embeddings stored
in Neo4j. Compares embedding-based peers with sector-based peers (from
SIC/GICS classification).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


def run():
    print("[ch07_fin] Peer Search by Embedding")
    print("=" * 60)

    gp = GraphProvider()

    # Get entities with embeddings
    entities = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.profileEmbedding IS NOT NULL
        RETURN le.lei AS lei, le.name AS name,
               le.profileEmbedding AS embedding
        LIMIT 100
    """)
    print(f"\n  Entities with embeddings: {len(entities)}")

    if len(entities) < 2:
        print("  Need at least 2 embedded entities. Run embed_issuer_profiles first.")
        gp.close()
        return

    # For each entity, find top-5 peers by cosine similarity
    print("\n  Computing peer similarities …\n")
    for i, entity in enumerate(entities[:10]):  # show top 10 entities
        sims = []
        for j, other in enumerate(entities):
            if i == j:
                continue
            sim = cosine_similarity(entity["embedding"], other["embedding"])
            sims.append((other["name"], other["lei"], sim))
        sims.sort(key=lambda x: x[2], reverse=True)

        print(f"  {entity['name']} ({entity['lei'][:8]}…)")
        for name, lei, sim in sims[:5]:
            print(f"    → {name:<40s} sim={sim:.4f}")
        print()

    # Compare with sector-based peers
    print("  Sector-based peer comparison …")
    sector_peers = gp.run("""
        MATCH (le1:LegalEntity)-[:CLASSIFIED_AS]->(oc:OntologyClass)<-[:CLASSIFIED_AS]-(le2:LegalEntity)
        WHERE le1 <> le2 AND le1.profileEmbedding IS NOT NULL AND le2.profileEmbedding IS NOT NULL
        RETURN le1.name AS entity, le2.name AS peer, oc.label AS sector
        LIMIT 20
    """)
    for row in sector_peers:
        print(f"    {row['entity']:<30s} ↔ {row['peer']:<30s} ({row['sector']})")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
