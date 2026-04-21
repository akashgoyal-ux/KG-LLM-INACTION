"""
community_louvain.py
====================
Run Louvain community detection on the LegalEntity ownership/parent graph.

Two modes:
  1. GDS (Graph Data Science) — if the GDS plugin is installed.
  2. Cypher-only fallback — uses a simple label-propagation approximation
     via connected components when GDS is unavailable.

Results are written back to LegalEntity.communityLouvain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def _has_gds(g: GraphProvider) -> bool:
    """Check if GDS plugin is available."""
    try:
        result = g.run("RETURN gds.version() AS v")
        print(f"  GDS version: {result[0]['v']}")
        return True
    except Exception:
        return False


def _run_gds_louvain(g: GraphProvider):
    """Louvain via GDS projected graph."""
    print("  Creating GDS projection …")
    # Drop existing projection if any
    try:
        g.run("CALL gds.graph.drop('fin_ownership', false)")
    except Exception:
        pass

    g.run("""
        CALL gds.graph.project(
            'fin_ownership',
            'LegalEntity',
            {
                OWNS:       { orientation: 'UNDIRECTED' },
                PARENT_OF:  { orientation: 'UNDIRECTED' }
            }
        )
    """)

    print("  Running Louvain …")
    result = g.run("""
        CALL gds.louvain.write('fin_ownership', {
            writeProperty: 'communityLouvain'
        })
        YIELD communityCount, modularity, ranLevels
        RETURN communityCount, modularity, ranLevels
    """)
    rec = result[0]
    print(f"  Communities: {rec['communityCount']}, "
          f"Modularity: {rec['modularity']:.4f}, "
          f"Levels: {rec['ranLevels']}")

    # Clean up
    g.run("CALL gds.graph.drop('fin_ownership', false)")


def _run_cypher_fallback(g: GraphProvider):
    """
    Fallback: assign community IDs based on connected components
    using iterative label propagation in pure Cypher.
    """
    print("  GDS not available — using Cypher connected-component fallback …")

    # Step 1: Initialize each node with its own community
    g.run("""
        MATCH (le:LegalEntity)
        SET le.communityLouvain = id(le)
    """)

    # Step 2: Propagate minimum ID through relationships (5 iterations)
    for i in range(5):
        updated = g.run("""
            MATCH (a:LegalEntity)-[:OWNS|PARENT_OF]-(b:LegalEntity)
            WHERE a.communityLouvain > b.communityLouvain
            SET a.communityLouvain = b.communityLouvain
            RETURN count(a) AS updated
        """)
        cnt = updated[0]["updated"]
        if cnt == 0:
            print(f"    Converged after {i + 1} iterations.")
            break
        print(f"    Iteration {i + 1}: updated {cnt} nodes.")


def main():
    g = GraphProvider()
    print("[ch04_fin] Community Detection (Louvain)")
    print("=" * 60)

    # Check node count
    cnt = g.run("MATCH (le:LegalEntity) RETURN count(le) AS cnt")[0]["cnt"]
    print(f"  LegalEntity nodes: {cnt}")

    rel_cnt = g.run(
        "MATCH ()-[r:OWNS|PARENT_OF]-() RETURN count(r) AS cnt"
    )[0]["cnt"]
    print(f"  OWNS + PARENT_OF relationships: {rel_cnt}")

    if rel_cnt == 0:
        print("  No relationships to analyze. Run ch04_fin importers first.")
        g.close()
        return

    if _has_gds(g):
        _run_gds_louvain(g)
    else:
        _run_cypher_fallback(g)

    # Report
    print("\n  Community summary:")
    for rec in g.run("""
        MATCH (le:LegalEntity)
        WHERE le.communityLouvain IS NOT NULL
        WITH le.communityLouvain AS cid, count(le) AS size
        RETURN cid, size ORDER BY size DESC LIMIT 10
    """):
        print(f"    Community {rec['cid']}: {rec['size']} entities")

    # Show members of largest community
    print("\n  Largest community members (up to 15):")
    for rec in g.run("""
        MATCH (le:LegalEntity)
        WHERE le.communityLouvain IS NOT NULL
        WITH le.communityLouvain AS cid, count(le) AS size
        ORDER BY size DESC LIMIT 1
        MATCH (le2:LegalEntity {communityLouvain: cid})
        RETURN le2.name AS name, le2.lei AS lei
        LIMIT 15
    """):
        print(f"    {rec['name']}")

    g.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
