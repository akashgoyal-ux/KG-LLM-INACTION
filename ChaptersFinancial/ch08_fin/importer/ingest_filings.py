"""
ingest_filings.py
=================
Fetch real SEC EDGAR filings (10-K, 10-Q, 8-K) and XBRL companyfacts,
then store as Filing + StatementItem nodes in Neo4j.

Uses SEC EDGAR's public APIs:
  - EFTS full-text search for recent filings
  - companyfacts JSON for XBRL financial data
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx
from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

_SEC_HEADERS = {
    "User-Agent": "KG-LLM-INACTION/1.0 (financial-kg-research; contact@example.com)",
    "Accept": "application/json",
}

# Sample CIKs for major companies (real SEC CIK numbers)
_PILOT_CIKS = {
    "0000320193": "Apple Inc.",
    "0000789019": "Microsoft Corporation",
    "0001652044": "Alphabet Inc.",
    "0001018724": "Amazon.com Inc.",
    "0001318605": "Tesla Inc.",
    "0000051143": "International Business Machines Corp.",
    "0000078003": "Pfizer Inc.",
    "0000019617": "JPMorgan Chase & Co.",
    "0000070858": "Bank of America Corp.",
    "0000886982": "Goldman Sachs Group Inc.",
}


class FilingIngester(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch08_fin.ingest_filings")
        self._cache_dir = Path(__file__).resolve().parents[3] / "data_fin" / "cache_api"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_submissions(self, cik: str) -> dict | None:
        """Fetch filing submissions for a CIK from SEC EDGAR."""
        cache_path = self._cache_dir / f"sec_sub_{cik}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            resp = httpx.get(url, headers=_SEC_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cache_path.write_text(json.dumps(data, indent=2))
            time.sleep(0.2)  # SEC rate limit
            return data
        except Exception as exc:
            print(f"      [WARN] Could not fetch submissions for CIK {cik}: {exc}")
            return None

    def _fetch_companyfacts(self, cik: str) -> dict | None:
        """Fetch XBRL companyfacts for a CIK."""
        cache_path = self._cache_dir / f"sec_facts_{cik}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            resp = httpx.get(url, headers=_SEC_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            cache_path.write_text(json.dumps(data, indent=2))
            time.sleep(0.2)
            return data
        except Exception as exc:
            print(f"      [WARN] Could not fetch companyfacts for CIK {cik}: {exc}")
            return None

    def _ingest_filings_for_cik(self, cik: str, company: str):
        """Process a single CIK's filings."""
        print(f"    {company} (CIK {cik}) …")

        # 1. Get filing submissions
        subs = self._fetch_submissions(cik)
        if not subs:
            return

        # Store Filing nodes from recent filings
        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        filings = []
        for i in range(min(len(forms), 20)):  # last 20 filings
            form_type = forms[i] if i < len(forms) else ""
            if form_type not in ("10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"):
                continue
            filing_id = (accessions[i] if i < len(accessions) else "").replace("-", "")
            filings.append({
                "filingId": filing_id,
                "formType": form_type,
                "filedAt": dates[i] if i < len(dates) else "",
                "cik": cik.lstrip("0"),
                "sourceUrl": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{filing_id}/{primary_docs[i] if i < len(primary_docs) else ''}",
            })

        if filings:
            query = """
            UNWIND $batch AS row
            MERGE (f:Filing {filingId: row.filingId})
            SET f.formType = row.formType,
                f.filedAt  = row.filedAt,
                f.cik      = row.cik,
                f.sourceUrl = row.sourceUrl,
                f.runId    = row.runId,
                f.ingestedAt = row.ingestedAt
            WITH f, row
            MATCH (le:LegalEntity {cik: row.cik})
            MERGE (f)-[:REPORTS_ON]->(le)
            """
            self.batch_store(query, iter(filings), size=len(filings),
                             desc=f"  Filings for {company}")

        # 2. Get XBRL companyfacts and store StatementItems
        facts = self._fetch_companyfacts(cik)
        if not facts:
            return

        items = []
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        key_concepts = [
            "Revenues", "NetIncomeLoss", "Assets", "Liabilities",
            "StockholdersEquity", "EarningsPerShareBasic",
            "OperatingIncomeLoss", "CashAndCashEquivalentsAtCarryingValue",
        ]
        for concept in key_concepts:
            concept_data = us_gaap.get(concept, {})
            units = concept_data.get("units", {})
            for unit_type, entries in units.items():
                for entry in entries[-10:]:  # last 10 periods
                    if not entry.get("val"):
                        continue
                    filing_acc = (entry.get("accn") or "").replace("-", "")
                    period = entry.get("end") or entry.get("start") or ""
                    item_id = hashlib.sha256(
                        f"{filing_acc}_{concept}_{period}".encode()
                    ).hexdigest()[:16]
                    items.append({
                        "itemId": item_id,
                        "filingId": filing_acc,
                        "concept": f"us-gaap:{concept}",
                        "period": period,
                        "value": entry["val"],
                        "unit": unit_type,
                        "xbrlConcept": concept,
                    })

        if items:
            query = """
            UNWIND $batch AS row
            MERGE (si:StatementItem {
                filingId: row.filingId,
                concept: row.concept,
                period: row.period
            })
            SET si.value       = row.value,
                si.unit        = row.unit,
                si.xbrlConcept = row.xbrlConcept,
                si.runId       = row.runId,
                si.ingestedAt  = row.ingestedAt
            WITH si, row
            MATCH (f:Filing {filingId: row.filingId})
            MERGE (si)-[:FROM_FILING]->(f)
            """
            self.batch_store(query, iter(items), size=len(items),
                             desc=f"  StatementItems for {company}")

        print(f"      {len(filings)} filings, {len(items)} statement items")

    def run(self):
        print("[ch08_fin] SEC Filing Ingestion")
        print("=" * 60)

        self.ensure_constraints()

        print("\n1. Ingesting filings for pilot companies …")
        for cik, company in _PILOT_CIKS.items():
            self._ingest_filings_for_cik(cik, company)

        print("\n2. Verification …")
        with self._driver.session(database=self._database) as session:
            f_cnt = session.run("MATCH (f:Filing) RETURN count(f) AS cnt").single()["cnt"]
            si_cnt = session.run("MATCH (si:StatementItem) RETURN count(si) AS cnt").single()["cnt"]
            linked = session.run(
                "MATCH (f:Filing)-[:REPORTS_ON]->(le:LegalEntity) RETURN count(f) AS cnt"
            ).single()["cnt"]
            print(f"   Filing nodes: {f_cnt}")
            print(f"   StatementItem nodes: {si_cnt}")
            print(f"   Filings linked to entities: {linked}")

        print("\nDone.")


if __name__ == "__main__":
    ingester = FilingIngester(argv=sys.argv[1:])
    try:
        ingester.run()
    finally:
        ingester.close()
