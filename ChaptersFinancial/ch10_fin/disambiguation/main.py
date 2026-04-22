"""
main.py (ch10_fin)
==================
Run the same NED pipeline as ch09_fin but using Ollama (local open models).
Set LLM_PROVIDER=ollama to use local models.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider
from ChaptersFinancial.ch09_fin.disambiguation.candidate_generator import CandidateGenerator
from ChaptersFinancial.ch09_fin.disambiguation.ontology_linker import link_entities_to_fibo


def run():
    # Force Ollama provider for this chapter
    provider = os.getenv("LLM_PROVIDER", "ollama")

    print(f"[ch10_fin] Open-Model NED (provider={provider})")
    print("=" * 60)

    gp = GraphProvider()
    llm = LLMProvider(provider=provider)
    cgen = CandidateGenerator(gp, llm)

    # Get unresolved mentions
    mentions = gp.run("""
        MATCH (m:Mention)
        WHERE NOT (m)-[:RESOLVED_TO]->()
          AND m.label IN ['ORG', 'PERSON', 'TICKER']
        RETURN m.mentionId AS mentionId, m.text AS text, m.label AS label
        LIMIT 100
    """)
    print(f"\n  Unresolved mentions: {len(mentions)}")

    start = time.time()
    resolved = 0

    for mention in mentions:
        candidates = cgen.generate_candidates(mention["text"], top_k=5)
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c.get("score", 0))
        if best.get("score", 0) < 0.70:
            continue

        if "lei" in best:
            gp.run("""
                MATCH (m:Mention {mentionId: $mid})
                MATCH (le:LegalEntity {lei: $lei})
                MERGE (m)-[r:RESOLVED_TO]->(le)
                SET r.confidence = $score,
                    r.linker = $source
            """, {
                "mid": mention["mentionId"],
                "lei": best["lei"],
                "score": best["score"],
                "source": f"ollama_{best.get('source', 'gen')}",
            })
            resolved += 1

    elapsed = time.time() - start
    print(f"\n  Resolved: {resolved}/{len(mentions)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  LLM usage: {llm.usage_summary()}")

    link_entities_to_fibo(gp)

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
