"""
normalize_and_merge.py
======================
Normalize LLM-extracted entities and merge them into the canonical KG.
Uses the ch05_fin reconciliation logic to resolve aliases before merging.

Steps:
  1. Load LLM-extracted mentions with identifiers
  2. Attempt deterministic resolution (LEI/CIK/FIGI from identifiers)
  3. Fuzzy name-match for unresolved entities
  4. Create RESOLVED_TO edges with provenance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


def run():
    print("[ch08_fin] Normalize and Merge Extracted Entities")
    print("=" * 60)

    gp = GraphProvider()

    # 1. Deterministic resolution via identifiers
    print("\n1. Deterministic resolution via extracted identifiers …")
    det_query = """
    MATCH (m:Mention {extractor: 'llm_ch08'})
    WHERE m.identifiers IS NOT NULL AND NOT (m)-[:RESOLVED_TO]->()
    WITH m, apoc.convert.fromJsonMap(m.identifiers) AS ids
    WHERE ids IS NOT NULL

    // Try LEI match
    OPTIONAL MATCH (le_lei:LegalEntity {lei: ids.lei})
    WHERE ids.lei IS NOT NULL

    // Try CIK match
    OPTIONAL MATCH (le_cik:LegalEntity {cik: ids.cik})
    WHERE ids.cik IS NOT NULL

    // Try FIGI match
    OPTIONAL MATCH (i_figi:Instrument {figi: ids.figi})
    WHERE ids.figi IS NOT NULL

    WITH m, coalesce(le_lei, le_cik) AS le, i_figi AS inst
    WHERE le IS NOT NULL OR inst IS NOT NULL

    FOREACH (_ IN CASE WHEN le IS NOT NULL THEN [1] ELSE [] END |
        MERGE (m)-[r:RESOLVED_TO]->(le)
        SET r.confidence = 0.95, r.linker = 'deterministic_id'
    )
    FOREACH (_ IN CASE WHEN inst IS NOT NULL AND le IS NULL THEN [1] ELSE [] END |
        MERGE (m)-[r:RESOLVED_TO]->(inst)
        SET r.confidence = 0.95, r.linker = 'deterministic_id'
    )
    RETURN count(m) AS resolved
    """
    try:
        cnt = gp.run(det_query)[0]["resolved"]
        print(f"   Resolved {cnt} mentions via identifiers.")
    except Exception as exc:
        print(f"   [WARN] Deterministic resolution: {exc}")
        print("   APOC may not be installed. Skipping identifier-based resolution.")

    # 2. Fuzzy name matching for remaining mentions
    print("\n2. Fuzzy name matching for unresolved mentions …")
    fuzzy_query = """
    MATCH (m:Mention {extractor: 'llm_ch08'})
    WHERE NOT (m)-[:RESOLVED_TO]->() AND m.label IN ['ORG', 'PERSON']
    MATCH (le:LegalEntity)
    WHERE le.name IS NOT NULL
      AND apoc.text.jaroWinklerDistance(toLower(m.text), toLower(le.name)) > 0.85
    WITH m, le,
         apoc.text.jaroWinklerDistance(toLower(m.text), toLower(le.name)) AS sim
    ORDER BY sim DESC
    WITH m, collect({entity: le, score: sim})[0] AS best
    WHERE best.score > 0.85
    MERGE (m)-[r:RESOLVED_TO]->(best.entity)
    SET r.confidence = best.score, r.linker = 'fuzzy_name_ch08'
    RETURN count(r) AS resolved
    """
    try:
        cnt = gp.run(fuzzy_query)[0]["resolved"]
        print(f"   Resolved {cnt} mentions via fuzzy name matching.")
    except Exception as exc:
        print(f"   [WARN] Fuzzy matching: {exc}")

    # 3. Create DOCUMENT-[:MENTIONS]->LegalEntity edges
    print("\n3. Creating Document-MENTIONS-Entity edges …")
    doc_link = """
    MATCH (m:Mention)-[:IN_CHUNK]->(c:Chunk)-[:OF_DOC]->(d:Document),
          (m)-[:RESOLVED_TO]->(le:LegalEntity)
    MERGE (d)-[:MENTIONS]->(le)
    RETURN count(*) AS cnt
    """
    try:
        cnt = gp.run(doc_link)[0]["cnt"]
        print(f"   Created {cnt} Document-MENTIONS edges.")
    except Exception as exc:
        print(f"   [WARN] {exc}")

    # Summary
    print("\n4. Summary …")
    stats = gp.run("""
        MATCH (m:Mention {extractor: 'llm_ch08'})
        OPTIONAL MATCH (m)-[r:RESOLVED_TO]->(n)
        RETURN count(m) AS totalMentions,
               count(r) AS resolved,
               count(m) - count(r) AS unresolved
    """)
    if stats:
        s = stats[0]
        print(f"   Total LLM mentions: {s['totalMentions']}")
        print(f"   Resolved: {s['resolved']}")
        print(f"   Unresolved: {s['unresolved']}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
