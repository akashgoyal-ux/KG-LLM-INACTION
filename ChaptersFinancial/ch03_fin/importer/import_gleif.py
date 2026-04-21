"""
import_gleif.py
===============
Stream real Legal Entity Identifier (LEI) records from the GLEIF public API
and merge them as :LegalEntity nodes into Neo4j.

The GLEIF API is free, requires no authentication, and provides authoritative
legal-entity reference data.

Configuration (via env vars or defaults):
  GLEIF_COUNTRY   — ISO 3166-1 alpha-2 country filter (default: US)
  GLEIF_PAGE_SIZE — number of records per API page   (default: 200)
  GLEIF_MAX_PAGES — max pages to fetch               (default: 5)
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

_GLEIF_API = "https://api.gleif.org/api/v1/lei-records"


class GLEIFImporter(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch03_fin.import_gleif")
        self.country = os.getenv("GLEIF_COUNTRY", "US")
        self.page_size = int(os.getenv("GLEIF_PAGE_SIZE", "200"))
        self.max_pages = int(os.getenv("GLEIF_MAX_PAGES", "5"))

    # ------------------------------------------------------------------
    # GLEIF API streaming
    # ------------------------------------------------------------------
    def _fetch_page(self, client: httpx.Client, page: int) -> list[dict]:
        """Fetch one page of LEI records from GLEIF API with retry/backoff."""
        params = {
            "filter[entity.legalAddress.country]": self.country,
            "filter[entity.status]": "ACTIVE",
            "page[size]": self.page_size,
            "page[number]": page + 1,  # 1-indexed
        }
        max_retries = 4
        for attempt in range(max_retries):
            try:
                # Use a fresh client per retry to avoid stale connections
                resp = client.get(_GLEIF_API, params=params, timeout=90)
                resp.raise_for_status()
                return resp.json().get("data", [])
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt * 3  # 3s, 6s, 12s
                print(f"\n  [WARN] page {page + 1} attempt {attempt + 1} failed ({exc!r}); retrying in {wait}s …")
                time.sleep(wait)
        return []  # unreachable

    def _parse_record(self, rec: dict) -> dict:
        """Convert a GLEIF JSON record to our canonical LegalEntity dict."""
        attrs = rec.get("attributes", {})
        entity = attrs.get("entity", {})
        legal_name = entity.get("legalName", {})
        legal_addr = entity.get("legalAddress", {})
        reg = attrs.get("registration", {})

        # Build aliases from otherNames
        other_names = entity.get("otherNames", [])
        aliases = [n.get("name", "") for n in other_names if n.get("name")]

        return {
            "lei": attrs.get("lei", ""),
            "name": legal_name.get("name", ""),
            "legalForm": entity.get("legalForm", {}).get("id", ""),
            "jurisdiction": entity.get("jurisdiction", ""),
            "status": entity.get("status", "ACTIVE"),
            "registeredAddress": ", ".join(
                filter(None, [
                    legal_addr.get("addressLines", [""])[0] if legal_addr.get("addressLines") else "",
                    legal_addr.get("city", ""),
                    legal_addr.get("region", ""),
                    legal_addr.get("postalCode", ""),
                    legal_addr.get("country", ""),
                ])
            ),
            "aliases": aliases,
        }

    def _fetch_all(self) -> list[dict]:
        """Fetch up to max_pages of LEI records."""
        all_records = []
        with httpx.Client() as client:
            for page in tqdm(range(self.max_pages), desc="GLEIF pages"):
                records = self._fetch_page(client, page)
                if not records:
                    break
                for rec in records:
                    parsed = self._parse_record(rec)
                    if parsed["lei"]:
                        all_records.append(parsed)
                # Be nice to the API
                time.sleep(0.5)
        return all_records

    # ------------------------------------------------------------------
    # Import GLEIF Level 2 (parent relationships)
    # ------------------------------------------------------------------
    def _fetch_relationships(self, client: httpx.Client, lei: str) -> list[dict]:
        """Fetch parent/child relationships for a given LEI."""
        url = f"https://api.gleif.org/api/v1/lei-records/{lei}/direct-parent-relationship"
        try:
            resp = client.get(url, timeout=15)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json().get("data", {})
            if not data:
                return []
            attrs = data.get("attributes", data) if isinstance(data, dict) else {}
            rel = attrs.get("relationship", {})
            parent_lei = rel.get("startNode", {}).get("id", "")
            child_lei = rel.get("endNode", {}).get("id", "")
            if parent_lei and child_lei:
                return [{"parentLei": parent_lei, "childLei": child_lei}]
        except Exception:
            pass
        return []

    _MERGE_PARENT_OF = """
    UNWIND $batch AS row
    MATCH (parent:LegalEntity {lei: row.parentLei})
    MATCH (child:LegalEntity {lei: row.childLei})
    MERGE (parent)-[r:PARENT_OF]->(child)
    SET r.source = 'GLEIF_L2', r.asOf = datetime(), r.runId = row.runId
    """

    def _import_relationships(self, leis: list[str]):
        """Fetch and import parent-child relationships for imported LEIs."""
        parent_rels = []
        with httpx.Client() as client:
            for lei in tqdm(leis[:100], desc="GLEIF relationships"):
                rels = self._fetch_relationships(client, lei)
                for r in rels:
                    r["runId"] = self.run_id
                parent_rels.extend(rels)
                time.sleep(0.3)

        if parent_rels:
            print(f"  Found {len(parent_rels)} parent-child relationships.")
            self.batch_store(
                self._MERGE_PARENT_OF,
                iter(parent_rels),
                size=len(parent_rels),
                desc="Merging PARENT_OF",
            )
        else:
            print("  No parent-child relationships found in this batch.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self):
        print("[ch03_fin] GLEIF LEI Import")
        print("=" * 60)
        print(f"  Country: {self.country}")
        print(f"  Page size: {self.page_size}, Max pages: {self.max_pages}")

        print("\n1. Ensuring schema constraints …")
        self.ensure_constraints()

        print("\n2. Fetching LEI records from GLEIF API …")
        records = self._fetch_all()
        print(f"  Fetched {len(records)} active legal entities.")

        if not records:
            print("  No records to import. Check GLEIF_COUNTRY setting.")
            return

        print("\n3. Merging LegalEntity nodes …")
        self.merge_legal_entities(records)

        print("\n4. Fetching parent-child relationships …")
        leis = [r["lei"] for r in records]
        self._import_relationships(leis)

        print("\n5. Verification …")
        self._verify()
        print("\nDone.")

    def _verify(self):
        with self._driver.session(database=self._database) as session:
            cnt = session.run(
                "MATCH (le:LegalEntity) RETURN count(le) AS cnt"
            ).single()["cnt"]
            print(f"  LegalEntity nodes: {cnt}")

            rels = session.run(
                "MATCH ()-[r:PARENT_OF]->() RETURN count(r) AS cnt"
            ).single()["cnt"]
            print(f"  PARENT_OF relationships: {rels}")

            # Sample
            sample = session.run(
                "MATCH (le:LegalEntity) "
                "RETURN le.lei AS lei, le.name AS name, le.jurisdiction AS jurisdiction "
                "ORDER BY le.name LIMIT 10"
            )
            print("\n  Sample entities:")
            for rec in sample:
                print(f"    {rec['lei']}  {rec['name']:<40s}  {rec['jurisdiction']}")


if __name__ == "__main__":
    importer = GLEIFImporter(argv=sys.argv[1:])
    try:
        importer.run()
    finally:
        importer.close()
