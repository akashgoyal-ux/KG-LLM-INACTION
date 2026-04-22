"""
lineage_emitter.py
==================
Record data lineage edges in Neo4j: which Run produced which nodes,
from which source, at what time.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def emit_lineage(gp: GraphProvider, run_id: str, label: str,
                 source: str, count: int):
    """Record that run_id produced `count` nodes of `label` from `source`."""
    gp.run("""
        MERGE (r:Run {runId: $runId})
        SET r.updatedAt = $ts
        WITH r
        MERGE (r)-[p:PRODUCED]->(meta:RunOutput {label: $label, source: $source, runId: $runId})
        SET meta.count = $count, meta.at = $ts
    """, {
        "runId": run_id,
        "label": label,
        "source": source,
        "count": count,
        "ts": datetime.utcnow().isoformat(),
    })


def lineage_summary(gp: GraphProvider):
    """Print lineage summary."""
    runs = gp.run("""
        MATCH (r:Run)-[:PRODUCED]->(ro:RunOutput)
        RETURN r.runId AS runId, collect({label: ro.label, source: ro.source, count: ro.count}) AS outputs
        ORDER BY r.updatedAt DESC LIMIT 20
    """)
    print(f"\n{'Run ID':<40s} {'Outputs':>10s}")
    print("-" * 52)
    for r in runs:
        print(f"  {r['runId']:<38s} {len(r['outputs']):>3d} batches")
        for o in r["outputs"]:
            print(f"    {o['label']:<20s} {o['source']:<15s} n={o['count']}")


if __name__ == "__main__":
    gp = GraphProvider()
    lineage_summary(gp)
    gp.close()
