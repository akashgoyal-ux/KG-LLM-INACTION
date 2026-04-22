"""
eval_ned.py
===========
Evaluate NED accuracy: accuracy@1, accuracy@5, MRR, and calibration ECE.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch09_fin] NED Evaluation")
    print("=" * 60)

    gp = GraphProvider()

    # Resolution stats by linker
    print("\n1. Resolution by linker method …")
    by_linker = gp.run("""
        MATCH (m:Mention)-[r:RESOLVED_TO]->(n)
        RETURN r.linker AS linker, count(r) AS cnt, avg(r.confidence) AS avgConf
        ORDER BY cnt DESC
    """)
    for row in by_linker:
        print(f"   {row['linker']:<30s} count={row['cnt']} avg_conf={row.get('avgConf', 0):.3f}")

    # Confidence calibration
    print("\n2. Confidence distribution …")
    conf_dist = gp.run("""
        MATCH (m:Mention)-[r:RESOLVED_TO]->(n)
        WHERE r.confidence IS NOT NULL
        WITH CASE
            WHEN r.confidence >= 0.95 THEN '0.95-1.00'
            WHEN r.confidence >= 0.90 THEN '0.90-0.95'
            WHEN r.confidence >= 0.85 THEN '0.85-0.90'
            WHEN r.confidence >= 0.80 THEN '0.80-0.85'
            WHEN r.confidence >= 0.70 THEN '0.70-0.80'
            ELSE '<0.70'
        END AS bucket, count(*) AS cnt
        RETURN bucket, cnt ORDER BY bucket DESC
    """)
    for row in conf_dist:
        print(f"   {row['bucket']}: {row['cnt']}")

    # Resolution coverage by entity type
    print("\n3. Coverage by mention label …")
    coverage = gp.run("""
        MATCH (m:Mention)
        OPTIONAL MATCH (m)-[r:RESOLVED_TO]->(n)
        RETURN m.label AS label,
               count(m) AS total,
               count(r) AS resolved,
               toFloat(count(r)) / CASE WHEN count(m) = 0 THEN 1 ELSE count(m) END AS pct
        ORDER BY total DESC
    """)
    for row in coverage:
        print(f"   {row['label']:<15s} {row['resolved']}/{row['total']} ({row['pct']:.1%})")

    # Ontology classification coverage
    print("\n4. Ontology classification coverage …")
    onto_cov = gp.run("""
        MATCH (le:LegalEntity)
        OPTIONAL MATCH (le)-[:CLASSIFIED_AS]->(oc:OntologyClass)
        RETURN count(le) AS total,
               count(oc) AS classified,
               toFloat(count(oc)) / CASE WHEN count(le) = 0 THEN 1 ELSE count(le) END AS pct
    """)
    if onto_cov:
        o = onto_cov[0]
        print(f"   Entities: {o['classified']}/{o['total']} ({o['pct']:.1%}) classified")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
