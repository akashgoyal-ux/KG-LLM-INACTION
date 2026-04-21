"""
import_ownership_13f.py
=======================
Seed OWNS relationships from SEC 13F institutional-ownership filings.

Uses the SEC EDGAR Full-Text Search (EFTS) API to discover which institutional
investors hold positions in our known instruments.  For each instrument, we
search for 13F filings that mention the company name and extract the filer
(institutional holder) CIK and name.

Creates:
  (:LegalEntity {cik})-[:OWNS {source: 'SEC_13F', asOf}]->(i:Instrument)

The EFTS search is free, requires no API key, and provides structured metadata
about filers and their filing dates.
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

_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinKG research@example.com")

_MERGE_HOLDER = """
UNWIND $batch AS row
MERGE (le:LegalEntity {cik: row.cik})
SET le.name       = coalesce(row.name, le.name),
    le.runId      = row.runId,
    le.ingestedAt = row.ingestedAt
"""

_MERGE_OWNS = """
UNWIND $batch AS row
MATCH (holder:LegalEntity {cik: row.holderCik})
MATCH (inst:Instrument {ticker: row.ticker})
MERGE (holder)-[r:OWNS]->(inst)
SET r.source  = 'SEC_13F',
    r.asOf    = row.asOf,
    r.runId   = row.runId
"""


class Ownership13FImporter(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch04_fin.import_ownership_13f")

    def _get_known_instruments(self) -> list[dict]:
        """Fetch our known instruments from the graph."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                "MATCH (i:Instrument) WHERE i.ticker IS NOT NULL "
                "RETURN i.ticker AS ticker, i.name AS name"
            )
            return [{"ticker": r["ticker"], "name": r["name"] or r["ticker"]}
                    for r in result]

    def _search_13f_holders(self, client: httpx.Client, company_name: str) -> list[dict]:
        """
        Search SEC EDGAR EFTS for 13F filings mentioning a company.
        Returns a list of {cik, name, asOf} for the institutional filers.
        """
        # Use first two words for search (avoid overly specific queries)
        words = company_name.replace(",", "").replace(".", "").split()
        search_term = " ".join(words[:2]) if len(words) >= 2 else company_name

        params = {
            "q": f'"{search_term}"',
            "forms": "13F-HR",
            "dateRange": "custom",
            "startdt": "2024-01-01",
            "enddt": "2025-12-31",
        }
        try:
            resp = client.get(
                _EFTS_URL, params=params,
                headers={"User-Agent": _USER_AGENT},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        # Extract unique filer CIKs from the search hits
        holders = {}
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:50]:
            src = hit.get("_source", {})
            ciks = src.get("ciks", [])
            names = src.get("display_names", [])
            filing_date = src.get("file_date", "")

            for i, cik in enumerate(ciks):
                if cik not in holders:
                    display = names[i] if i < len(names) else f"CIK-{cik}"
                    # Parse "Company Name  (TICK)  (CIK 0001234567)"
                    clean_name = display.split("  (")[0].strip() if "  (" in display else display
                    holders[cik] = {
                        "cik": cik,
                        "name": clean_name,
                        "asOf": filing_date,
                    }

        return list(holders.values())

    def run(self):
        print("[ch04_fin] SEC 13F Ownership Import")
        print("=" * 60)

        print("\n1. Ensuring schema constraints …")
        self.ensure_constraints()

        print("\n2. Loading known instruments from graph …")
        instruments = self._get_known_instruments()
        print(f"  Known instruments: {len(instruments)}")

        if not instruments:
            print("  No instruments found. Run import_instruments_figi.py first.")
            return

        print("\n3. Searching SEC EDGAR EFTS for 13F holders …")
        all_links = []    # {holderCik, ticker, asOf}
        all_holders = {}  # cik -> {cik, name}

        with httpx.Client() as client:
            # Search for a subset of instruments (keep API calls reasonable)
            search_batch = instruments[:20]
            for inst in tqdm(search_batch, desc="EFTS 13F search"):
                holders = self._search_13f_holders(client, inst["name"])
                for h in holders:
                    all_holders[h["cik"]] = h
                    all_links.append({
                        "holderCik": h["cik"],
                        "ticker": inst["ticker"],
                        "asOf": h.get("asOf", "2024"),
                        "runId": self.run_id,
                    })
                time.sleep(0.12)  # SEC rate limit: ~10 req/sec

        # Deduplicate (same holder + ticker)
        seen = set()
        unique_links = []
        for link in all_links:
            key = (link["holderCik"], link["ticker"])
            if key not in seen:
                seen.add(key)
                unique_links.append(link)

        print(f"\n  Unique institutional holders found: {len(all_holders)}")
        print(f"  Unique ownership links: {len(unique_links)}")

        if not unique_links:
            print("  No ownership data found.")
            return

        print("\n4. Merging holder LegalEntity nodes …")
        holder_rows = [{"cik": h["cik"], "name": h["name"]} for h in all_holders.values()]
        self.batch_store(_MERGE_HOLDER, iter(holder_rows), size=len(holder_rows),
                         desc="Holders")

        print("\n5. Merging OWNS relationships …")
        self.batch_store(_MERGE_OWNS, iter(unique_links), size=len(unique_links),
                         desc="OWNS")

        print("\n6. Verification …")
        with self._driver.session(database=self._database) as session:
            owns = session.run("MATCH ()-[r:OWNS]->() RETURN count(r) AS cnt").single()["cnt"]
            print(f"  OWNS relationships: {owns}")

            # Top holders by number of holdings
            print("\n  Top holders (by # of holdings):")
            for rec in session.run("""
                MATCH (h:LegalEntity)-[r:OWNS]->(i:Instrument)
                RETURN h.name AS holder, h.cik AS cik, count(i) AS holdings
                ORDER BY holdings DESC LIMIT 10
            """):
                print(f"    {rec['holder']:<50s} {rec['holdings']:>3} holdings")

            # Sample edges
            print("\n  Sample OWNS edges:")
            for rec in session.run("""
                MATCH (h:LegalEntity)-[r:OWNS]->(i:Instrument)
                RETURN h.name AS holder, i.ticker AS ticker
                ORDER BY h.name, i.ticker LIMIT 10
            """):
                print(f"    {rec['holder']:<50s} → {rec['ticker']}")

        print("\nDone.")


if __name__ == "__main__":
    importer = Ownership13FImporter(argv=sys.argv[1:])
    try:
        importer.run()
    finally:
        importer.close()
