"""
apply_migrations.py
===================
Apply versioned Cypher migrations to Neo4j. Records applied migrations
as :Migration nodes for audit.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider

# Migrations are ordered Cypher statements
_MIGRATIONS = [
    {
        "id": "20260418_001_init",
        "description": "Initial schema constraints",
        "cypher": """
            CREATE CONSTRAINT legal_entity_lei IF NOT EXISTS
              FOR (le:LegalEntity) REQUIRE le.lei IS UNIQUE;
            CREATE CONSTRAINT instrument_figi IF NOT EXISTS
              FOR (i:Instrument) REQUIRE i.figi IS UNIQUE;
            CREATE CONSTRAINT exchange_mic IF NOT EXISTS
              FOR (e:Exchange) REQUIRE e.mic IS UNIQUE;
            CREATE CONSTRAINT document_docid IF NOT EXISTS
              FOR (d:Document) REQUIRE d.docId IS UNIQUE;
            CREATE CONSTRAINT filing_filingid IF NOT EXISTS
              FOR (f:Filing) REQUIRE f.filingId IS UNIQUE;
        """,
    },
    {
        "id": "20260418_002_indexes",
        "description": "Performance indexes",
        "cypher": """
            CREATE INDEX le_name_idx IF NOT EXISTS FOR (le:LegalEntity) ON (le.name);
            CREATE INDEX le_cik_idx IF NOT EXISTS FOR (le:LegalEntity) ON (le.cik);
            CREATE INDEX instrument_ticker_idx IF NOT EXISTS FOR (i:Instrument) ON (i.ticker);
            CREATE INDEX chunk_docid_idx IF NOT EXISTS FOR (c:Chunk) ON (c.docId);
            CREATE INDEX mention_chunkid_idx IF NOT EXISTS FOR (m:Mention) ON (m.chunkId);
        """,
    },
    {
        "id": "20260418_003_crosswalk",
        "description": "Crosswalk constraint for entity resolution",
        "cypher": """
            CREATE CONSTRAINT crosswalk_unique IF NOT EXISTS
              FOR (cw:Crosswalk) REQUIRE (cw.idA, cw.idTypeA, cw.idB, cw.idTypeB) IS UNIQUE;
        """,
    },
]


def run():
    print("[ch16_fin] Apply Schema Migrations")
    print("=" * 60)

    gp = GraphProvider()

    # Get already-applied migrations
    try:
        applied = {r["migrationId"] for r in gp.run(
            "MATCH (m:Migration) RETURN m.migrationId AS migrationId"
        )}
    except Exception:
        applied = set()

    print(f"\n  Already applied: {len(applied)}")

    for migration in _MIGRATIONS:
        mid = migration["id"]
        if mid in applied:
            print(f"  [SKIP] {mid} — already applied")
            continue

        print(f"  [APPLY] {mid}: {migration['description']}")
        statements = migration["cypher"].strip().split(";")
        for stmt in statements:
            stmt = stmt.strip()
            if stmt:
                try:
                    gp.run(stmt)
                except Exception as exc:
                    print(f"    [WARN] {exc}")

        # Record migration
        gp.run("""
            CREATE (m:Migration {
                migrationId: $id,
                description: $desc,
                appliedAt: $ts
            })
        """, {"id": mid, "desc": migration["description"],
              "ts": datetime.utcnow().isoformat()})

    # Summary
    total = gp.run("MATCH (m:Migration) RETURN count(m) AS cnt")[0]["cnt"]
    print(f"\n  Total migrations applied: {total}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
