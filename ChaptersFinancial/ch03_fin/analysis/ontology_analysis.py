"""
ontology_analysis.py
====================
Structural analysis of the ch03_fin ontology layer:
  - FIBO module coverage (BE / FBC / SEC / IND domains)
  - ISO 4217 currency distribution by first letter / numeric range
  - GLEIF entity distribution by country, entity type, and legal form
  - Cross-ontology connectivity (OntologyClass ↔ LegalEntity links)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.providers.graph import GraphProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(value: int, max_val: int, width: int = 30) -> str:
    filled = int(width * value / max_val) if max_val else 0
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# FIBO Module Analysis
# ---------------------------------------------------------------------------

def fibo_analysis(g: GraphProvider) -> None:
    _section("FIBO Ontology Modules")

    total = g.run("MATCH (oc:OntologyClass {source: 'FIBO'}) RETURN count(oc) AS cnt")[0]["cnt"]
    print(f"  Total FIBO OntologyClass nodes: {total}")

    domains = ["BE", "FBC", "SEC", "IND"]
    domain_counts: dict[str, int] = {}
    for d in domains:
        rows = g.run(
            "MATCH (oc:OntologyClass {source: 'FIBO'}) "
            "WHERE oc.iri CONTAINS $domain "
            "RETURN count(oc) AS cnt",
            {"domain": f"/fibo/ontology/{d}/"},
        )
        domain_counts[d] = rows[0]["cnt"] if rows else 0

    max_c = max(domain_counts.values(), default=1)
    print(f"\n  {'Domain':<8} {'Count':>6}  Distribution")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:<8} {c:>6}  {_bar(c, max_c)}")

    # Top FIBO labels
    print("\n  Sample FIBO labels:")
    for rec in g.run(
        "MATCH (oc:OntologyClass {source: 'FIBO'}) "
        "WHERE oc.label IS NOT NULL AND oc.label <> '' "
        "RETURN oc.label AS label, oc.iri AS iri "
        "ORDER BY oc.label LIMIT 10"
    ):
        label = rec["label"]
        iri_parts = (rec["iri"] or "").rstrip("/").split("/")
        short = iri_parts[-1] or iri_parts[-2]
        print(f"    {label:<45} ({short})")


# ---------------------------------------------------------------------------
# ISO 4217 Currency Analysis
# ---------------------------------------------------------------------------

def currency_analysis(g: GraphProvider) -> None:
    _section("ISO 4217 Currencies")

    total = g.run("MATCH (oc:OntologyClass {source: 'ISO4217'}) RETURN count(oc) AS cnt")[0]["cnt"]
    print(f"  Total currency nodes: {total}")

    rows = g.run(
        "MATCH (oc:OntologyClass {source: 'ISO4217'}) "
        "RETURN oc.currencyCode AS code, oc.label AS name "
        "ORDER BY oc.currencyCode"
    )

    # Distribution by first letter
    letter_counts: Counter = Counter()
    for r in rows:
        code = r["code"] or ""
        if code:
            letter_counts[code[0]] += 1

    print(f"\n  Currencies per first-letter group:")
    max_c = max(letter_counts.values(), default=1)
    for letter in sorted(letter_counts):
        c = letter_counts[letter]
        print(f"    {letter}  {c:>3}  {_bar(c, max_c, 20)}")

    # Sample major currencies
    major = ["USD", "EUR", "GBP", "JPY", "CHF", "CNY", "CAD", "AUD"]
    print("\n  Major reserve currencies present:")
    for code in major:
        rows2 = g.run(
            "MATCH (oc:OntologyClass {source: 'ISO4217', currencyCode: $code}) "
            "RETURN oc.label AS name",
            {"code": code},
        )
        name = rows2[0]["name"] if rows2 else None
        status = "✓" if name else "✗"
        print(f"    {status} {code:<4} {name or '(not found)'}")


# ---------------------------------------------------------------------------
# GLEIF Entity Analysis
# ---------------------------------------------------------------------------

def gleif_analysis(g: GraphProvider) -> None:
    _section("GLEIF Legal Entities")

    total = g.run("MATCH (le:LegalEntity) RETURN count(le) AS cnt")[0]["cnt"]
    print(f"  Total LegalEntity nodes: {total}")

    # By jurisdiction
    jur_rows = g.run(
        "MATCH (le:LegalEntity) "
        "WHERE le.jurisdiction IS NOT NULL "
        "RETURN le.jurisdiction AS jur, count(le) AS cnt "
        "ORDER BY cnt DESC LIMIT 15"
    )
    print(f"\n  Top jurisdictions:")
    max_c = max((r["cnt"] for r in jur_rows), default=1)
    for r in jur_rows:
        print(f"    {r['jur']:<12} {r['cnt']:>5}  {_bar(r['cnt'], max_c, 25)}")

    # By legal form
    cat_rows = g.run(
        "MATCH (le:LegalEntity) "
        "WHERE le.legalForm IS NOT NULL "
        "RETURN le.legalForm AS cat, count(le) AS cnt "
        "ORDER BY cnt DESC LIMIT 10"
    )
    if cat_rows:
        print(f"\n  Top legal forms (GLEIF codes):")
        max_c = max((r["cnt"] for r in cat_rows), default=1)
        for r in cat_rows:
            print(f"    {r['cat']:<30} {r['cnt']:>5}  {_bar(r['cnt'], max_c, 20)}")

    # PARENT_OF relationships
    parent_cnt = g.run("MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS cnt")[0]["cnt"]
    print(f"\n  PARENT_OF relationships: {parent_cnt}")
    if parent_cnt > 0:
        print("  Sample parent-child chains:")
        for r in g.run(
            "MATCH (p:LegalEntity)-[:PARENT_OF]->(c:LegalEntity) "
            "RETURN p.name AS parent, c.name AS child LIMIT 5"
        ):
            print(f"    {r['parent']}  →  {r['child']}")


# ---------------------------------------------------------------------------
# Cross-Ontology Connectivity
# ---------------------------------------------------------------------------

def connectivity_analysis(g: GraphProvider) -> None:
    _section("Cross-Ontology Connectivity")

    # Check for any link between LegalEntity and OntologyClass
    linked = g.run(
        "MATCH (le:LegalEntity)-[r]-(oc:OntologyClass) "
        "RETURN type(r) AS rel, count(*) AS cnt "
        "ORDER BY cnt DESC LIMIT 10"
    )
    if linked:
        print("  LegalEntity ↔ OntologyClass relationships:")
        for r in linked:
            print(f"    {r['rel']:<30} {r['cnt']:>5}")
    else:
        print("  No direct LegalEntity ↔ OntologyClass edges yet.")
        print("  (These will be added in ch04_fin via CLASSIFIED_AS / DENOMINATED_IN.)")

    # Node label summary
    _section("Overall Graph Summary (ch03_fin perspective)")
    label_rows = g.run(
        "CALL db.labels() YIELD label "
        "CALL (label) { "
        "  MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt "
        "} "
        "RETURN label, cnt ORDER BY cnt DESC"
    )
    print(f"  {'Label':<25} {'Count':>8}")
    print(f"  {'─'*25} {'─'*8}")
    for r in label_rows:
        print(f"  {r['label']:<25} {r['cnt']:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    g = GraphProvider()
    print("[ch03_fin] Ontology Analysis")
    print("=" * 60)

    fibo_analysis(g)
    currency_analysis(g)
    gleif_analysis(g)
    connectivity_analysis(g)

    g.close()
    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    main()
