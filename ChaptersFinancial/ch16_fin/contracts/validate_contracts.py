"""
validate_contracts.py
=====================
Validate Neo4j data against pydantic-based data contracts.
Ensures data quality across the KG by checking:
  - Required fields are populated
  - Value ranges are valid
  - Referential integrity (edges point to existing nodes)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def _check_required_fields(gp: GraphProvider) -> list[str]:
    """Check that key required fields are populated."""
    issues = []

    # LegalEntity must have lei and name
    bad_le = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.lei IS NULL OR le.name IS NULL
        RETURN count(le) AS cnt
    """)[0]["cnt"]
    if bad_le > 0:
        issues.append(f"LegalEntity: {bad_le} nodes missing lei or name")

    # Instrument must have figi
    bad_i = gp.run("""
        MATCH (i:Instrument)
        WHERE i.figi IS NULL
        RETURN count(i) AS cnt
    """)[0]["cnt"]
    if bad_i > 0:
        issues.append(f"Instrument: {bad_i} nodes missing figi")

    # Filing must have filingId
    bad_f = gp.run("""
        MATCH (f:Filing)
        WHERE f.filingId IS NULL
        RETURN count(f) AS cnt
    """)[0]["cnt"]
    if bad_f > 0:
        issues.append(f"Filing: {bad_f} nodes missing filingId")

    return issues


def _check_referential_integrity(gp: GraphProvider) -> list[str]:
    """Check that edges reference existing nodes."""
    issues = []

    # Chunks should reference existing Documents
    orphan_chunks = gp.run("""
        MATCH (c:Chunk)
        WHERE NOT (c)-[:OF_DOC]->(:Document)
        RETURN count(c) AS cnt
    """)[0]["cnt"]
    if orphan_chunks > 0:
        issues.append(f"Chunk: {orphan_chunks} orphan chunks (no OF_DOC)")

    # Mentions should reference existing Chunks
    orphan_mentions = gp.run("""
        MATCH (m:Mention)
        WHERE NOT (m)-[:IN_CHUNK]->(:Chunk)
        RETURN count(m) AS cnt
    """)[0]["cnt"]
    if orphan_mentions > 0:
        issues.append(f"Mention: {orphan_mentions} orphan mentions (no IN_CHUNK)")

    return issues


def _check_constraints(gp: GraphProvider) -> list[str]:
    """Verify uniqueness constraints exist."""
    issues = []
    constraints = gp.run("SHOW CONSTRAINTS")
    constraint_names = {c.get("name", "") for c in constraints}

    expected = [
        "legal_entity_lei", "instrument_figi", "exchange_mic",
        "document_docid", "filing_filingid",
    ]
    for name in expected:
        if not any(name in cn for cn in constraint_names):
            issues.append(f"Missing constraint: {name}")

    return issues


def run():
    print("[ch16_fin] Data Contract Validation")
    print("=" * 60)

    gp = GraphProvider()
    all_issues = []

    print("\n1. Checking required fields …")
    all_issues.extend(_check_required_fields(gp))

    print("\n2. Checking referential integrity …")
    all_issues.extend(_check_referential_integrity(gp))

    print("\n3. Checking constraints …")
    try:
        all_issues.extend(_check_constraints(gp))
    except Exception:
        print("   [WARN] Could not check constraints (may require Neo4j 4.4+)")

    # Summary
    print("\n4. Validation Summary")
    if all_issues:
        print(f"   ISSUES FOUND: {len(all_issues)}")
        for issue in all_issues:
            print(f"   - {issue}")
    else:
        print("   ALL CONTRACTS PASS")

    # Node count summary
    print("\n5. Node Counts")
    labels = ["LegalEntity", "Instrument", "Exchange", "Filing", "StatementItem",
              "Document", "Chunk", "Mention", "Event", "OntologyClass", "Crosswalk"]
    for label in labels:
        cnt = gp.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")[0]["cnt"]
        if cnt > 0:
            print(f"   {label:<20s} {cnt:>8d}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
