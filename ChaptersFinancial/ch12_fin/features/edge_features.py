"""
edge_features.py
================
Compute edge-level features for risk and AML analysis:
  - Ownership percentage and recency
  - Multiplexity (number of relationship types between same pair)
  - Path-based features (shortest path length to high-centrality nodes)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch12_fin] Edge Feature Engineering")
    print("=" * 60)

    gp = GraphProvider()

    # 1. Multiplexity: count distinct relationship types between entity pairs
    print("\n1. Computing multiplexity features …")
    gp.run("""
        MATCH (a:LegalEntity)-[r]-(b:LegalEntity)
        WHERE id(a) < id(b)
        WITH a, b, collect(DISTINCT type(r)) AS relTypes, count(r) AS relCount
        WHERE relCount > 1
        SET a.feat_maxMultiplex = CASE
            WHEN a.feat_maxMultiplex IS NULL OR relCount > a.feat_maxMultiplex
            THEN relCount ELSE a.feat_maxMultiplex END
    """)
    print("   Multiplexity computed.")

    # 2. Ownership path features
    print("\n2. Computing ownership path features …")
    try:
        gp.run("""
            MATCH (le:LegalEntity)
            WHERE le.pagerank IS NOT NULL
            WITH le ORDER BY le.pagerank DESC LIMIT 5
            WITH collect(le) AS hubs
            UNWIND hubs AS hub
            MATCH (other:LegalEntity)
            WHERE other <> hub
            MATCH path = shortestPath((other)-[:OWNS|CONTROLS|PARENT_OF*..5]-(hub))
            WITH other, min(length(path)) AS distToHub
            SET other.feat_minDistToHub = distToHub
        """)
        print("   Path features computed.")
    except Exception as exc:
        print(f"   [WARN] Path feature computation: {exc}")

    # Summary
    print("\n3. Edge feature summary …")
    multi = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.feat_maxMultiplex IS NOT NULL
        RETURN count(le) AS withMultiplex, avg(le.feat_maxMultiplex) AS avgMultiplex
    """)
    if multi:
        print(f"   Entities with multiplex edges: {multi[0]['withMultiplex']}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
