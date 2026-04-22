"""
step3_enrich_via_apis.py
========================
Enrich LegalEntity/Instrument nodes using real-time API lookups:
  - GLEIF API: enrich entity details by LEI or name
  - OpenFIGI API: resolve ticker mentions to instruments

Results cached to data_fin/cache_api/ for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx
from ChaptersFinancial._platform.providers.graph import GraphProvider

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data_fin" / "cache_api"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": "KG-LLM-INACTION/1.0 (financial-kg-research)",
    "Accept": "application/json",
}


def _cache_get(key: str) -> dict | None:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _cache_set(key: str, data: dict):
    path = _CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(data, indent=2))


def enrich_via_gleif(gp: GraphProvider):
    """Look up unresolved ORG mentions via GLEIF name search."""
    print("\n  GLEIF enrichment …")
    mentions = gp.run("""
        MATCH (m:Mention {label: 'ORG'})
        WHERE NOT (m)-[:RESOLVED_TO]->()
        RETURN DISTINCT m.text AS name
        LIMIT 50
    """)
    enriched = 0
    for row in mentions:
        name = row["name"]
        cache_key = hashlib.sha256(f"gleif_name_{name}".encode()).hexdigest()[:16]
        cached = _cache_get(cache_key)
        if cached is None:
            try:
                resp = httpx.get(
                    "https://api.gleif.org/api/v1/lei-records",
                    params={"filter[entity.legalName]": name, "page[size]": "1"},
                    headers=_HEADERS, timeout=15,
                )
                resp.raise_for_status()
                cached = resp.json()
                _cache_set(cache_key, cached)
                time.sleep(0.5)  # rate limit
            except Exception as exc:
                print(f"    [WARN] GLEIF lookup failed for '{name}': {exc}")
                continue

        records = cached.get("data", [])
        if records:
            lei_data = records[0]["attributes"]["entity"]
            lei = records[0]["attributes"]["lei"]
            # Merge enriched entity
            gp.run("""
                MERGE (le:LegalEntity {lei: $lei})
                SET le.name         = coalesce($name, le.name),
                    le.jurisdiction = coalesce($jurisdiction, le.jurisdiction),
                    le.legalForm    = coalesce($legalForm, le.legalForm),
                    le.status       = coalesce($status, le.status)
            """, {
                "lei": lei,
                "name": lei_data.get("legalName", {}).get("name"),
                "jurisdiction": lei_data.get("jurisdiction"),
                "legalForm": lei_data.get("legalForm", {}).get("id"),
                "status": lei_data.get("status"),
            })
            # Link mention to entity
            gp.run("""
                MATCH (m:Mention {text: $text})
                WHERE NOT (m)-[:RESOLVED_TO]->()
                MATCH (le:LegalEntity {lei: $lei})
                MERGE (m)-[r:RESOLVED_TO]->(le)
                SET r.confidence = 0.90, r.linker = 'gleif_api'
            """, {"text": name, "lei": lei})
            enriched += 1

    print(f"    Enriched {enriched} entities via GLEIF.")


def enrich_via_openfigi(gp: GraphProvider):
    """Resolve TICKER mentions via OpenFIGI API."""
    print("\n  OpenFIGI enrichment …")
    mentions = gp.run("""
        MATCH (m:Mention {label: 'TICKER'})
        WHERE NOT (m)-[:RESOLVED_TO]->()
        RETURN DISTINCT m.text AS ticker
        LIMIT 50
    """)
    enriched = 0
    for row in mentions:
        ticker = row["ticker"].replace("$", "")
        cache_key = hashlib.sha256(f"figi_{ticker}".encode()).hexdigest()[:16]
        cached = _cache_get(cache_key)
        if cached is None:
            try:
                resp = httpx.post(
                    "https://api.openfigi.com/v3/mapping",
                    json=[{"idType": "TICKER", "idValue": ticker, "exchCode": "US"}],
                    headers={**_HEADERS, "Content-Type": "application/json"},
                    timeout=15,
                )
                resp.raise_for_status()
                cached = resp.json()
                _cache_set(cache_key, cached)
                time.sleep(0.5)
            except Exception as exc:
                print(f"    [WARN] OpenFIGI lookup failed for '{ticker}': {exc}")
                continue

        if cached and isinstance(cached, list) and cached[0].get("data"):
            figi_data = cached[0]["data"][0]
            figi = figi_data.get("figi")
            if figi:
                gp.run("""
                    MERGE (i:Instrument {figi: $figi})
                    SET i.ticker    = coalesce($ticker, i.ticker),
                        i.name     = coalesce($name, i.name),
                        i.exchCode = coalesce($exchCode, i.exchCode)
                """, {
                    "figi": figi,
                    "ticker": ticker,
                    "name": figi_data.get("name"),
                    "exchCode": figi_data.get("exchCode"),
                })
                gp.run("""
                    MATCH (m:Mention {text: $text})
                    WHERE NOT (m)-[:RESOLVED_TO]->()
                    MATCH (i:Instrument {figi: $figi})
                    MERGE (m)-[r:RESOLVED_TO]->(i)
                    SET r.confidence = 0.95, r.linker = 'openfigi_api'
                """, {"text": row["ticker"], "figi": figi})
                enriched += 1

    print(f"    Enriched {enriched} tickers via OpenFIGI.")


def run():
    print("[ch06_fin] API Enrichment Pipeline")
    print("=" * 60)

    gp = GraphProvider()
    enrich_via_gleif(gp)
    enrich_via_openfigi(gp)

    print("\n  Summary …")
    resolved = gp.run(
        "MATCH (m:Mention)-[:RESOLVED_TO]->(n) RETURN count(m) AS cnt"
    )[0]["cnt"]
    total = gp.run("MATCH (m:Mention) RETURN count(m) AS cnt")[0]["cnt"]
    print(f"    Resolved: {resolved}/{total} mentions ({100*resolved/max(total,1):.1f}%)")

    gp.close()
    print("\nDone.")


if __name__ == "__main__":
    run()
