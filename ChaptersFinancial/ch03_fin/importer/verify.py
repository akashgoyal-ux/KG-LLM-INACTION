"""
verify.py
=========
Quick post-import verification for ch03_fin.
Prints counts and sample data for OntologyClass (FIBO, ISO4217) and LegalEntity (GLEIF).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def main():
    g = GraphProvider()
    print("[ch03_fin] Verification")
    print("=" * 60)

    queries = [
        ("OntologyClass (FIBO)", "MATCH (oc:OntologyClass {source: 'FIBO'}) RETURN count(oc) AS cnt"),
        ("OntologyClass (ISO4217)", "MATCH (oc:OntologyClass {source: 'ISO4217'}) RETURN count(oc) AS cnt"),
        ("LegalEntity (total)", "MATCH (le:LegalEntity) RETURN count(le) AS cnt"),
        ("PARENT_OF rels", "MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS cnt"),
    ]

    for label, q in queries:
        result = g.run(q)
        cnt = result[0]["cnt"] if result else 0
        status = "✓" if cnt > 0 else "✗"
        print(f"  {status} {label}: {cnt}")

    # Sample FIBO classes
    print("\nSample FIBO classes:")
    for rec in g.run(
        "MATCH (oc:OntologyClass {source: 'FIBO'}) "
        "RETURN oc.label AS label LIMIT 5"
    ):
        print(f"    {rec['label']}")

    # Sample entities
    print("\nSample LegalEntities:")
    for rec in g.run(
        "MATCH (le:LegalEntity) "
        "RETURN le.lei AS lei, le.name AS name LIMIT 5"
    ):
        print(f"    {rec['lei']}  {rec['name']}")

    g.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
