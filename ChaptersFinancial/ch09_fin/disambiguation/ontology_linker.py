"""
ontology_linker.py
==================
Link resolved entities to FIBO ontology classes using Neosemantics.
Assigns (:LegalEntity)-[:CLASSIFIED_AS]->(:OntologyClass) edges based on
entity type, legal form, and sector using FIBO class hierarchy in Neo4j.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider

# Mapping of legal forms to FIBO class IRIs
_LEGAL_FORM_TO_FIBO = {
    "CORP": "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/CorporateBodies/Corporation",
    "LLC": "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LimitedLiabilityCompanies/LimitedLiabilityCompany",
    "FUND": "https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/Funds/Fund",
    "BANK": "https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/Bank",
    "PARTNERSHIP": "https://spec.edmcouncil.org/fibo/ontology/BE/Partnerships/Partnerships/Partnership",
}


def link_entities_to_fibo(gp: GraphProvider) -> int:
    """
    Link LegalEntity nodes to FIBO OntologyClass nodes based on legal form.
    Uses existing FIBO classes loaded by ch03_fin via Neosemantics.
    """
    linked = 0

    # 1. Link by legal form
    for form, fibo_iri in _LEGAL_FORM_TO_FIBO.items():
        result = gp.run("""
            MATCH (le:LegalEntity)
            WHERE le.legalForm = $form
              AND NOT (le)-[:CLASSIFIED_AS]->(:OntologyClass {iri: $iri})
            MATCH (oc:OntologyClass {iri: $iri})
            MERGE (le)-[:CLASSIFIED_AS]->(oc)
            RETURN count(*) AS cnt
        """, {"form": form, "iri": fibo_iri})
        cnt = result[0]["cnt"] if result else 0
        if cnt > 0:
            linked += cnt

    # 2. Try to link via FIBO class name similarity using n10s + APOC
    try:
        result = gp.run("""
            MATCH (le:LegalEntity)
            WHERE le.legalForm IS NOT NULL
              AND NOT (le)-[:CLASSIFIED_AS]->(:OntologyClass)
            MATCH (oc:OntologyClass {source: 'FIBO'})
            WHERE oc.label IS NOT NULL
              AND apoc.text.jaroWinklerDistance(
                    toLower(le.legalForm), toLower(oc.label)) > 0.80
            WITH le, oc,
                 apoc.text.jaroWinklerDistance(
                    toLower(le.legalForm), toLower(oc.label)) AS sim
            ORDER BY sim DESC
            WITH le, collect(oc)[0] AS bestClass
            MERGE (le)-[:CLASSIFIED_AS]->(bestClass)
            RETURN count(*) AS cnt
        """)
        linked += result[0]["cnt"] if result else 0
    except Exception:
        pass  # APOC not available

    return linked


def link_instruments_to_cfi(gp: GraphProvider) -> int:
    """Link Instrument nodes to CFI OntologyClass based on cfiCode."""
    result = gp.run("""
        MATCH (i:Instrument)
        WHERE i.cfiCode IS NOT NULL
        MATCH (oc:OntologyClass {source: 'CFI'})
        WHERE oc.label = i.cfiCode OR oc.iri CONTAINS i.cfiCode
        MERGE (i)-[:CLASSIFIED_AS]->(oc)
        RETURN count(*) AS cnt
    """)
    return result[0]["cnt"] if result else 0


def run():
    print("[ch09_fin] Ontology Linking")
    print("=" * 60)

    gp = GraphProvider()

    print("\n1. Linking entities to FIBO classes …")
    entity_cnt = link_entities_to_fibo(gp)
    print(f"   Linked {entity_cnt} entities to FIBO.")

    print("\n2. Linking instruments to CFI classes …")
    instr_cnt = link_instruments_to_cfi(gp)
    print(f"   Linked {instr_cnt} instruments to CFI.")

    # Summary
    print("\n3. Ontology coverage …")
    classified = gp.run("""
        MATCH (n)-[:CLASSIFIED_AS]->(oc:OntologyClass)
        RETURN labels(n)[0] AS nodeType, oc.source AS ontology, count(*) AS cnt
        ORDER BY cnt DESC
    """)
    for row in classified:
        print(f"   {row['nodeType']} → {row['ontology']}: {row['cnt']}")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
