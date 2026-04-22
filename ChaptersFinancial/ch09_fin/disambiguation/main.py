"""
main.py
=======
Orchestrate the full NED pipeline:
  1. Generate candidates for unresolved mentions
  2. Rank candidates
  3. Link to ontology classes
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider
from ChaptersFinancial.ch09_fin.disambiguation.candidate_generator import CandidateGenerator
from ChaptersFinancial.ch09_fin.disambiguation.ontology_linker import (
    link_entities_to_fibo,
    link_instruments_to_cfi,
)


def run():
    print("[ch09_fin] Financial NED Pipeline")
    print("=" * 60)

    gp = GraphProvider()
    llm = LLMProvider()
    cgen = CandidateGenerator(gp, llm)

    # 1. Get unresolved mentions
    print("\n1. Loading unresolved mentions …")
    mentions = gp.run("""
        MATCH (m:Mention)
        WHERE NOT (m)-[:RESOLVED_TO]->()
          AND m.label IN ['ORG', 'PERSON', 'TICKER']
        RETURN m.mentionId AS mentionId, m.text AS text, m.label AS label
        LIMIT 200
    """)
    print(f"   Found {len(mentions)} unresolved mentions.")

    # 2. Generate candidates and resolve
    print("\n2. Generating candidates and resolving …")
    resolved = 0
    for mention in mentions:
        candidates = cgen.generate_candidates(mention["text"], top_k=5)
        if not candidates:
            continue

        # Take the best candidate (highest score)
        best = max(candidates, key=lambda c: c.get("score", 0))
        if best.get("score", 0) < 0.70:
            continue

        # Create RESOLVED_TO edge
        if "lei" in best:
            gp.run("""
                MATCH (m:Mention {mentionId: $mid})
                MATCH (le:LegalEntity {lei: $lei})
                MERGE (m)-[r:RESOLVED_TO]->(le)
                SET r.confidence = $score,
                    r.linker = $source,
                    r.candidates = $numCandidates
            """, {
                "mid": mention["mentionId"],
                "lei": best["lei"],
                "score": best["score"],
                "source": best.get("source", "candidate_gen"),
                "numCandidates": len(candidates),
            })
            resolved += 1
        elif "figi" in best:
            gp.run("""
                MATCH (m:Mention {mentionId: $mid})
                MATCH (i:Instrument {figi: $figi})
                MERGE (m)-[r:RESOLVED_TO]->(i)
                SET r.confidence = $score,
                    r.linker = $source
            """, {
                "mid": mention["mentionId"],
                "figi": best["figi"],
                "score": best["score"],
                "source": best.get("source", "candidate_gen"),
            })
            resolved += 1

    print(f"   Resolved {resolved}/{len(mentions)} mentions.")

    # 3. Ontology linking
    print("\n3. Ontology linking …")
    fibo_cnt = link_entities_to_fibo(gp)
    cfi_cnt = link_instruments_to_cfi(gp)
    print(f"   FIBO links: {fibo_cnt}")
    print(f"   CFI links: {cfi_cnt}")

    # Summary
    print("\n4. NED Summary …")
    stats = gp.run("""
        MATCH (m:Mention)
        OPTIONAL MATCH (m)-[r:RESOLVED_TO]->(n)
        RETURN count(m) AS total,
               count(r) AS resolved,
               avg(r.confidence) AS avgConfidence
    """)
    if stats:
        s = stats[0]
        print(f"   Total mentions: {s['total']}")
        print(f"   Resolved: {s['resolved']} ({100*s['resolved']/max(s['total'],1):.1f}%)")
        print(f"   Avg confidence: {s.get('avgConfidence', 'N/A')}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
