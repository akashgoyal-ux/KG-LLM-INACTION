"""
embed_issuer_profiles.py
========================
Generate sentence embeddings for LegalEntity profile texts and store
them back in Neo4j as vector properties.

Uses LLMProvider.embed() which routes to the configured embedding backend
(OpenAI text-embedding-3-small / Ollama / mock).

The profile text is assembled from the entity's name, jurisdiction,
legal form, and sector classification.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider
from ChaptersFinancial._platform.providers.llm import LLMProvider


def build_profile_text(entity: dict) -> str:
    """Assemble a natural-language profile from entity properties."""
    parts = []
    if entity.get("name"):
        parts.append(entity["name"])
    if entity.get("jurisdiction"):
        parts.append(f"Jurisdiction: {entity['jurisdiction']}")
    if entity.get("legalForm"):
        parts.append(f"Legal form: {entity['legalForm']}")
    if entity.get("sectorName"):
        parts.append(f"Sector: {entity['sectorName']}")
    if entity.get("status"):
        parts.append(f"Status: {entity['status']}")
    return ". ".join(parts) if parts else "Unknown entity"


def run():
    print("[ch07_fin] Embed Issuer Profiles")
    print("=" * 60)

    gp = GraphProvider()
    llm = LLMProvider()

    # Fetch entities with their sector classification
    print("\n1. Loading entities …")
    entities = gp.run("""
        MATCH (le:LegalEntity)
        WHERE le.name IS NOT NULL
        OPTIONAL MATCH (le)-[:CLASSIFIED_AS]->(oc:OntologyClass)
        RETURN le.lei AS lei, le.name AS name,
               le.jurisdiction AS jurisdiction,
               le.legalForm AS legalForm,
               le.status AS status,
               oc.label AS sectorName
        LIMIT 200
    """)
    print(f"   Found {len(entities)} entities.")

    if not entities:
        gp.close()
        return

    # Build profile texts
    print("\n2. Building profile texts …")
    profiles = [(e["lei"], build_profile_text(e)) for e in entities]

    # Generate embeddings in batches
    print("\n3. Generating embeddings …")
    batch_size = 20
    total_embedded = 0
    for i in range(0, len(profiles), batch_size):
        batch = profiles[i:i + batch_size]
        texts = [p[1] for p in batch]
        leis = [p[0] for p in batch]

        embeddings = llm.embed(texts)

        # Store embeddings back in Neo4j
        for lei, embedding in zip(leis, embeddings):
            gp.run("""
                MATCH (le:LegalEntity {lei: $lei})
                SET le.profileEmbedding = $embedding
            """, {"lei": lei, "embedding": embedding})
            total_embedded += 1

        print(f"   Embedded {min(i + batch_size, len(profiles))}/{len(profiles)}")

    print(f"\n   Total embedded: {total_embedded}")
    print(f"   LLM usage: {llm.usage_summary()}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
