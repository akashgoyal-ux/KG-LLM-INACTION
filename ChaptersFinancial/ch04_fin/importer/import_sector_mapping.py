"""
import_sector_mapping.py
========================
Classify Instrument nodes by sector using SEC SIC codes.

For each instrument with a ticker, looks up its issuer's CIK from the SEC
company_tickers_exchange.json, then fetches the SIC code from the SEC
submissions API.  Creates:

  (:Instrument)-[:CLASSIFIED_AS]->(:OntologyClass {source: 'SIC'})

Also enriches LegalEntity nodes found via OWNS relationships to those
instruments.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
_SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinKG research@example.com")

# SIC division ranges → sector names
_SIC_SECTORS = {
    range(100, 1000): "Agriculture, Forestry & Fishing",
    range(1000, 1500): "Mining",
    range(1500, 1800): "Construction",
    range(2000, 4000): "Manufacturing",
    range(4000, 5000): "Transportation & Utilities",
    range(5000, 5200): "Wholesale Trade",
    range(5200, 6000): "Retail Trade",
    range(6000, 6800): "Finance, Insurance & Real Estate",
    range(7000, 9000): "Services",
    range(9100, 9730): "Public Administration",
}


def _sic_to_sector(sic: int) -> str:
    for rng, name in _SIC_SECTORS.items():
        if sic in rng:
            return name
    return "Other"


_MERGE_SIC_CLASS = """
UNWIND $batch AS row
MERGE (oc:OntologyClass {iri: row.iri})
SET oc.label       = row.label,
    oc.source      = 'SIC',
    oc.sicCode     = row.sicCode,
    oc.sector      = row.sector,
    oc.description = row.description,
    oc.runId       = row.runId,
    oc.ingestedAt  = row.ingestedAt
"""

_LINK_INSTRUMENT_CLASSIFIED_AS = """
UNWIND $batch AS row
MATCH (i:Instrument {ticker: row.ticker})
MATCH (oc:OntologyClass {iri: row.iri})
MERGE (i)-[:CLASSIFIED_AS]->(oc)
"""

_SET_INSTRUMENT_CIK = """
UNWIND $batch AS row
MATCH (i:Instrument {ticker: row.ticker})
SET i.cik      = row.cik,
    i.sicCode  = row.sicCode,
    i.sector   = row.sector
"""


class SectorImporter(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch04_fin.import_sector_mapping")

    def _get_ticker_to_cik(self) -> dict[str, int]:
        """Download SEC company tickers and build ticker→CIK map."""
        print("  Downloading SEC company ticker data …")
        resp = httpx.get(
            _SEC_TICKERS_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        fields = data.get("fields", [])
        rows = data.get("data", [])
        mapping = {}
        for row in rows:
            d = dict(zip(fields, row))
            ticker = d.get("ticker", "")
            cik = d.get("cik", 0)
            if ticker and cik:
                mapping[ticker] = int(cik)
        return mapping

    def _fetch_sic(self, client: httpx.Client, cik: int) -> tuple[str, str]:
        """Fetch SIC code + description from SEC submissions API."""
        padded = str(cik).zfill(10)
        url = _SEC_SUBMISSIONS.format(cik=padded)
        resp = client.get(url, headers={"User-Agent": _USER_AGENT}, timeout=15)
        if resp.status_code != 200:
            return "", ""
        data = resp.json()
        return str(data.get("sic", "")), data.get("sicDescription", "")

    def run(self):
        print("[ch04_fin] Sector (SIC) Import")
        print("=" * 60)

        print("\n1. Ensuring schema constraints …")
        self.ensure_constraints()

        # Get our known instruments
        with self._driver.session(database=self._database) as session:
            result = session.run("MATCH (i:Instrument) RETURN i.ticker AS ticker")
            our_tickers = [rec["ticker"] for rec in result if rec["ticker"]]

        print(f"\n2. Our instruments: {len(our_tickers)}")

        print("\n3. Looking up CIKs from SEC data …")
        ticker_to_cik = self._get_ticker_to_cik()
        matched = {t: ticker_to_cik[t] for t in our_tickers if t in ticker_to_cik}
        print(f"  Matched {len(matched)} / {len(our_tickers)} tickers to CIKs.")

        print("\n4. Fetching SIC codes from SEC submissions API …")
        sic_classes = {}      # iri -> class dict
        instrument_links = [] # {ticker, iri}
        enrich_rows = []      # {ticker, cik, sicCode, sector}

        with httpx.Client() as client:
            for ticker, cik in tqdm(matched.items(), desc="SIC lookup"):
                sic_code, sic_desc = self._fetch_sic(client, cik)
                if not sic_code:
                    continue

                try:
                    sic_int = int(sic_code)
                except (ValueError, TypeError):
                    continue

                sector = _sic_to_sector(sic_int)
                iri = f"sic:{sic_int}"

                if iri not in sic_classes:
                    sic_classes[iri] = {
                        "iri": iri,
                        "label": f"SIC {sic_int} — {sector}",
                        "sicCode": sic_code,
                        "sector": sector,
                        "description": sic_desc,
                    }

                instrument_links.append({"ticker": ticker, "iri": iri})
                enrich_rows.append({
                    "ticker": ticker,
                    "cik": str(cik).zfill(10),
                    "sicCode": sic_code,
                    "sector": sector,
                })
                time.sleep(0.1)  # SEC rate limit

        print(f"\n  Unique SIC classes: {len(sic_classes)}")
        print(f"  Instruments with SIC: {len(instrument_links)}")

        print("\n5. Merging SIC OntologyClass nodes …")
        sic_rows = list(sic_classes.values())
        if sic_rows:
            self.batch_store(_MERGE_SIC_CLASS, iter(sic_rows), size=len(sic_rows),
                             desc="SIC classes")

        print("\n6. Linking instruments to sectors …")
        if instrument_links:
            self.batch_store(_LINK_INSTRUMENT_CLASSIFIED_AS, iter(instrument_links),
                             size=len(instrument_links), desc="CLASSIFIED_AS")

        print("\n7. Enriching instruments with CIK/sector …")
        if enrich_rows:
            self.batch_store(_SET_INSTRUMENT_CIK, iter(enrich_rows),
                             size=len(enrich_rows), desc="Instrument enrichment")

        print("\n8. Verification …")
        with self._driver.session(database=self._database) as session:
            cnt = session.run(
                "MATCH (oc:OntologyClass {source: 'SIC'}) RETURN count(oc) AS cnt"
            ).single()["cnt"]
            print(f"  SIC OntologyClass nodes: {cnt}")

            links = session.run(
                "MATCH ()-[r:CLASSIFIED_AS]->(:OntologyClass {source: 'SIC'}) "
                "RETURN count(r) AS cnt"
            ).single()["cnt"]
            print(f"  CLASSIFIED_AS relationships: {links}")

            print("\n  Sector distribution:")
            for rec in session.run(
                "MATCH (i:Instrument)-[:CLASSIFIED_AS]->(oc:OntologyClass {source: 'SIC'}) "
                "RETURN oc.sector AS sector, oc.description AS desc, count(i) AS cnt "
                "ORDER BY cnt DESC LIMIT 10"
            ):
                desc = rec["desc"] or ""
                print(f"    {rec['sector']:<35s} {desc:<30s} {rec['cnt']:>3}")

        print("\nDone.")


if __name__ == "__main__":
    importer = SectorImporter(argv=sys.argv[1:])
    try:
        importer.run()
    finally:
        importer.close()
