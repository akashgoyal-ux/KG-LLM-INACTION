"""
graph_summary.py
================
Print a comprehensive summary of the financial knowledge graph built by
ch03_fin and ch04_fin: node counts, relationship counts, community stats,
centrality leaders, sector distribution, and sample paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def main():
    g = GraphProvider()
    print("[ch04_fin] Graph Summary")
    print("=" * 60)

    # Node counts by label
    print("\n── Node Counts ──")
    for rec in g.run("""
        CALL db.labels() YIELD label
        CALL {
            WITH label
            MATCH (n) WHERE label IN labels(n)
            RETURN count(n) AS cnt
        }
        RETURN label, cnt ORDER BY cnt DESC
    """):
        print(f"  :{rec['label']:<25s} {rec['cnt']:>8,}")

    # Relationship counts
    print("\n── Relationship Counts ──")
    for rec in g.run("""
        CALL db.relationshipTypes() YIELD relationshipType AS type
        CALL {
            WITH type
            MATCH ()-[r]->() WHERE type(r) = type
            RETURN count(r) AS cnt
        }
        RETURN type, cnt ORDER BY cnt DESC
    """):
        print(f"  :{rec['type']:<25s} {rec['cnt']:>8,}")

    # Community stats
    print("\n── Community Stats ──")
    result = g.run("""
        MATCH (le:LegalEntity)
        WHERE le.communityLouvain IS NOT NULL
        WITH le.communityLouvain AS cid, count(le) AS size
        RETURN count(cid) AS numCommunities,
               max(size) AS largest,
               avg(size) AS avgSize,
               sum(size) AS totalCovered
    """)
    if result:
        r = result[0]
        print(f"  Communities: {r['numCommunities']}")
        print(f"  Largest: {r['largest']} entities")
        print(f"  Average: {r['avgSize']:.1f} entities")
        print(f"  Total covered: {r['totalCovered']}")

    # Top PageRank
    print("\n── Top-5 PageRank ──")
    for rec in g.run("""
        MATCH (le:LegalEntity)
        WHERE le.pagerank IS NOT NULL
        RETURN le.name AS name, le.pagerank AS pr
        ORDER BY le.pagerank DESC LIMIT 5
    """):
        print(f"  {rec['pr']:.6f}  {rec['name']}")

    # Sector distribution
    print("\n── Sector Distribution ──")
    for rec in g.run("""
        MATCH (n)-[:CLASSIFIED_AS]->(oc:OntologyClass {source: 'SIC'})
        RETURN oc.sector AS sector, count(n) AS cnt
        ORDER BY cnt DESC LIMIT 10
    """):
        print(f"  {rec['sector']:<45s} {rec['cnt']:>5}")

    # Sample ownership paths
    print("\n── Sample Ownership Paths ──")
    for rec in g.run("""
        MATCH (h:LegalEntity)-[r:OWNS]->(i:Instrument)
        RETURN h.name AS holder, i.ticker AS ticker, r.value AS value
        ORDER BY r.value DESC LIMIT 5
    """):
        val = f"${rec['value']:,.0f}" if rec['value'] else ""
        print(f"  {rec['holder']:<35s} → {rec['ticker']:<6s} {val}")

    # Sample issuer → exchange path
    print("\n── Sample Instrument → Exchange ──")
    for rec in g.run("""
        MATCH (i:Instrument)-[:LISTED_ON]->(ex:Exchange)
        RETURN i.ticker AS ticker, i.name AS name, ex.mic AS mic, ex.name AS exchange
        LIMIT 5
    """):
        print(f"  {rec['ticker']:<6s} ({rec['name']}) → {rec['mic']} ({rec['exchange']})")

    g.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
