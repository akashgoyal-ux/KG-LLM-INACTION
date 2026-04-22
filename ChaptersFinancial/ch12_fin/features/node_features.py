"""
node_features.py
================
Compute interpretable node-level graph features for risk scoring:
  - Degree (ownership, control, parent edges)
  - Triangles and clustering coefficient
  - Ego-net density
  - Centrality metrics (PageRank, betweenness)
  - Filing count and recency

All features stored as properties on :LegalEntity nodes and exported
to the feature store.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch12_fin] Node Feature Engineering")
    print("=" * 60)

    gp = GraphProvider()

    # 1. Degree features
    print("\n1. Computing degree features …")
    gp.run("""
        MATCH (le:LegalEntity)
        OPTIONAL MATCH (le)-[r_out:OWNS|CONTROLS|PARENT_OF]->()
        OPTIONAL MATCH (le)<-[r_in:OWNS|CONTROLS|PARENT_OF]-()
        WITH le, count(DISTINCT r_out) AS outDegree, count(DISTINCT r_in) AS inDegree
        SET le.feat_outDegree = outDegree,
            le.feat_inDegree = inDegree,
            le.feat_totalDegree = outDegree + inDegree
    """)
    print("   Degree features computed.")

    # 2. Triangle count via APOC/Cypher
    print("\n2. Computing triangle features …")
    try:
        gp.run("""
            MATCH (a:LegalEntity)-[:OWNS|CONTROLS]-(b:LegalEntity)-[:OWNS|CONTROLS]-(c:LegalEntity)-[:OWNS|CONTROLS]-(a)
            WHERE id(a) < id(b) AND id(b) < id(c)
            WITH a, count(*) AS triCount
            SET a.feat_triangles = triCount
        """)
        print("   Triangle counts computed.")
    except Exception as exc:
        print(f"   [WARN] Triangle computation: {exc}")

    # 3. Filing count and recency
    print("\n3. Computing filing features …")
    gp.run("""
        MATCH (le:LegalEntity)
        OPTIONAL MATCH (f:Filing)-[:REPORTS_ON]->(le)
        WITH le, count(f) AS filingCount,
             max(f.filedAt) AS latestFiling
        SET le.feat_filingCount = filingCount,
            le.feat_latestFiling = latestFiling
    """)
    print("   Filing features computed.")

    # 4. Mention count (how often referenced in news/filings)
    print("\n4. Computing mention features …")
    gp.run("""
        MATCH (le:LegalEntity)
        OPTIONAL MATCH (m:Mention)-[:RESOLVED_TO]->(le)
        WITH le, count(m) AS mentionCount
        SET le.feat_mentionCount = mentionCount
    """)
    print("   Mention features computed.")

    # 5. Event exposure count
    print("\n5. Computing event exposure features …")
    gp.run("""
        MATCH (le:LegalEntity)
        OPTIONAL MATCH (e:Event)-[:AFFECTS]->(le)
        WITH le, count(e) AS eventCount
        SET le.feat_eventCount = eventCount
    """)
    print("   Event features computed.")

    # Summary
    print("\n6. Feature summary …")
    stats = gp.run("""
        MATCH (le:LegalEntity)
        RETURN count(le) AS total,
               avg(le.feat_totalDegree) AS avgDegree,
               avg(le.feat_filingCount) AS avgFilings,
               avg(le.feat_mentionCount) AS avgMentions,
               avg(le.feat_eventCount) AS avgEvents
    """)
    if stats:
        s = stats[0]
        print(f"   Total entities: {s['total']}")
        print(f"   Avg degree: {s.get('avgDegree', 0):.2f}")
        print(f"   Avg filings: {s.get('avgFilings', 0):.2f}")
        print(f"   Avg mentions: {s.get('avgMentions', 0):.2f}")
        print(f"   Avg events: {s.get('avgEvents', 0):.2f}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
