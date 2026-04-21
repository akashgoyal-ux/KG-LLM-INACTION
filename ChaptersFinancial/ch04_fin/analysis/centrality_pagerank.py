"""
centrality_pagerank.py
======================
Compute PageRank on the LegalEntity relationship graph.

Two modes:
  1. GDS — if the Graph Data Science plugin is installed.
  2. Cypher-only fallback — iterative approximation in pure Cypher.

Results are written back to LegalEntity.pagerank.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def _has_gds(g: GraphProvider) -> bool:
    try:
        g.run("RETURN gds.version() AS v")
        return True
    except Exception:
        return False


def _run_gds_pagerank(g: GraphProvider):
    """PageRank via GDS."""
    print("  Creating GDS projection …")
    try:
        g.run("CALL gds.graph.drop('fin_pagerank', false)")
    except Exception:
        pass

    g.run("""
        CALL gds.graph.project(
            'fin_pagerank',
            'LegalEntity',
            {
                OWNS:       { orientation: 'NATURAL' },
                PARENT_OF:  { orientation: 'NATURAL' },
                CONTROLS:   { orientation: 'NATURAL' }
            }
        )
    """)

    print("  Running PageRank …")
    result = g.run("""
        CALL gds.pageRank.write('fin_pagerank', {
            writeProperty: 'pagerank',
            maxIterations: 20,
            dampingFactor: 0.85
        })
        YIELD nodePropertiesWritten, ranIterations, didConverge
        RETURN nodePropertiesWritten, ranIterations, didConverge
    """)
    rec = result[0]
    print(f"  Written: {rec['nodePropertiesWritten']}, "
          f"Iterations: {rec['ranIterations']}, "
          f"Converged: {rec['didConverge']}")

    g.run("CALL gds.graph.drop('fin_pagerank', false)")


def _run_cypher_fallback(g: GraphProvider):
    """
    Iterative PageRank approximation in pure Cypher.
    Simple power-iteration: 10 rounds with damping=0.85.
    """
    print("  GDS not available — using Cypher PageRank approximation …")
    damping = 0.85
    iterations = 10

    # Get total node count for initialisation
    total = g.run("MATCH (le:LegalEntity) RETURN count(le) AS cnt")[0]["cnt"]
    if total == 0:
        return
    init_rank = 1.0 / total

    # Initialise
    g.run("MATCH (le:LegalEntity) SET le.pagerank = $init", {"init": init_rank})

    for i in range(iterations):
        # Compute incoming rank from neighbors
        # Note: OWNS goes LegalEntity→Instrument, not LegalEntity→LegalEntity
        # So we consider PARENT_OF and reverse OWNS (who owns *me*)
        g.run("""
            MATCH (le:LegalEntity)
            OPTIONAL MATCH (src)-[:OWNS|PARENT_OF|CONTROLS]->(le)
            WHERE src:LegalEntity OR src:Instrument
            WITH le,
                 CASE WHEN count(src) > 0
                      THEN sum(
                          CASE WHEN src.pagerank IS NOT NULL AND src.pagerank > 0
                               THEN src.pagerank / toFloat(
                                   CASE WHEN size([(src)-[:OWNS|PARENT_OF|CONTROLS]->() | 1]) > 0
                                        THEN size([(src)-[:OWNS|PARENT_OF|CONTROLS]->() | 1])
                                        ELSE 1 END)
                               ELSE 0 END)
                      ELSE 0 END AS incoming
            SET le.pagerank = $base + $damping * incoming
        """, {"base": (1.0 - damping) / total, "damping": damping})
        print(f"    Iteration {i + 1}/{iterations}")

    print("  PageRank computation complete.")


def main():
    g = GraphProvider()
    print("[ch04_fin] Centrality (PageRank)")
    print("=" * 60)

    cnt = g.run("MATCH (le:LegalEntity) RETURN count(le) AS cnt")[0]["cnt"]
    print(f"  LegalEntity nodes: {cnt}")

    if cnt == 0:
        print("  No nodes. Run ch03_fin + ch04_fin importers first.")
        g.close()
        return

    if _has_gds(g):
        _run_gds_pagerank(g)
    else:
        _run_cypher_fallback(g)

    # Report top-10
    print("\n  Top-10 PageRank entities:")
    for rec in g.run("""
        MATCH (le:LegalEntity)
        WHERE le.pagerank IS NOT NULL
        RETURN le.name AS name, le.pagerank AS pr
        ORDER BY le.pagerank DESC LIMIT 10
    """):
        print(f"    {rec['pr']:.6f}  {rec['name']}")

    g.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
