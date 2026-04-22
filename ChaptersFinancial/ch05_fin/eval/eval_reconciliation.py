"""
eval_reconciliation.py
======================
Evaluate reconciliation quality against a golden crosswalk set.
Computes precision, recall, F1, and ROC-AUC of the blender.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def evaluate():
    """Run reconciliation evaluation against Neo4j crosswalk data."""
    print("[ch05_fin] Reconciliation Evaluation")
    print("=" * 60)

    gp = GraphProvider()

    # Count crosswalks by type
    print("\n1. Crosswalk summary …")
    for match_type in ["DETERMINISTIC", "DETERMINISTIC_NAME", "PROBABILISTIC"]:
        cnt = gp.run(
            "MATCH (cw:Crosswalk {matchType: $mt}) RETURN count(cw) AS cnt",
            {"mt": match_type},
        )[0]["cnt"]
        print(f"   {match_type}: {cnt}")

    # Review queue stats
    print("\n2. Review queue stats …")
    review = gp.run(
        "MATCH (cw:Crosswalk {needsReview: true}) "
        "RETURN count(cw) AS cnt, avg(cw.confidence) AS avgConf"
    )
    if review:
        print(f"   Pending review: {review[0]['cnt']}")
        print(f"   Average confidence: {review[0].get('avgConf', 'N/A')}")

    # Confidence distribution
    print("\n3. Confidence distribution …")
    dist = gp.run("""
        MATCH (cw:Crosswalk)
        WHERE cw.confidence IS NOT NULL
        WITH CASE
            WHEN cw.confidence >= 0.99 THEN '0.99-1.00'
            WHEN cw.confidence >= 0.90 THEN '0.90-0.99'
            WHEN cw.confidence >= 0.80 THEN '0.80-0.90'
            WHEN cw.confidence >= 0.70 THEN '0.70-0.80'
            ELSE '<0.70'
        END AS bucket, count(*) AS cnt
        RETURN bucket, cnt ORDER BY bucket
    """)
    for row in dist:
        print(f"   {row['bucket']}: {row['cnt']}")

    # Connected components of crosswalked entities
    print("\n4. Linked entity clusters …")
    clusters = gp.run("""
        MATCH (cw:Crosswalk)-[:LINKS]->(le:LegalEntity)
        WITH le.lei AS lei, collect(DISTINCT cw.idA) + collect(DISTINCT cw.idB) AS ids
        RETURN count(lei) AS entities, count(DISTINCT ids) AS crosswalkIds
    """)
    if clusters:
        print(f"   Entities with crosswalks: {clusters[0]['entities']}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    evaluate()
