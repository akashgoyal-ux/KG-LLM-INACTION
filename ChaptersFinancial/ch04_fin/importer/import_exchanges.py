"""
import_exchanges.py
===================
Import trading venues from the ISO 10383 MIC (Market Identifier Code) registry.
Uses the official CSV published by ISO via SWIFT.

Each exchange becomes:
  (:Exchange {mic, name, country, operatingMic})
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ChaptersFinancial._platform.fin_importer_base import FinImporterBase

# Official ISO 10383 MIC CSV (published by SWIFT / ISO)
_MIC_CSV_URL = "https://www.iso20022.org/sites/default/files/ISO10383_MIC/ISO10383_MIC.csv"

_MERGE_EXCHANGE = """
UNWIND $batch AS row
MERGE (ex:Exchange {mic: row.mic})
SET ex.name         = row.name,
    ex.country      = row.country,
    ex.operatingMic = row.operatingMic,
    ex.city         = row.city,
    ex.status       = row.status,
    ex.runId        = row.runId,
    ex.ingestedAt   = row.ingestedAt
"""


class ExchangeImporter(FinImporterBase):
    def __init__(self, argv=None):
        super().__init__(argv=argv)
        self.run_id = self.new_run_id("ch04_fin.import_exchanges")

    def _fetch_mic_csv(self) -> list[dict]:
        """Download and parse the ISO 10383 MIC CSV."""
        print("  Downloading MIC registry …")
        resp = httpx.get(_MIC_CSV_URL, timeout=30, follow_redirects=True)
        resp.raise_for_status()

        # The CSV may use various encodings; try utf-8-sig first
        text = resp.content.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))

        rows = []
        for row in reader:
            mic = row.get("MIC", row.get("mic", "")).strip()
            if not mic:
                continue
            rows.append({
                "mic": mic,
                "name": (row.get("MARKET NAME-INSTITUTION DESCRIPTION", "")
                         or row.get("NAME-INSTITUTION DESCRIPTION", "")
                         or row.get("INSTITUTION DESCRIPTION", "")).strip(),
                "country": (row.get("ISO COUNTRY CODE (ISO 3166)", "")
                            or row.get("COUNTRY CODE (ISO 3166)", "")).strip(),
                "operatingMic": row.get("OPERATING MIC", "").strip(),
                "city": row.get("CITY", "").strip(),
                "status": row.get("STATUS", "ACTIVE").strip(),
            })
        return rows

    def run(self):
        print("[ch04_fin] Exchange (MIC) Import")
        print("=" * 60)

        print("\n1. Ensuring schema constraints …")
        self.ensure_constraints()

        print("\n2. Fetching MIC registry …")
        rows = self._fetch_mic_csv()
        # ISO 10383 statuses: ACTIVE = new entry, UPDATED = modified but current, EXPIRED/DELETED = inactive
        active = [r for r in rows if r.get("status", "").upper() in ("ACTIVE", "UPDATED")]
        print(f"  Total MICs: {len(rows)}, Active+Updated: {len(active)}")

        print("\n3. Merging Exchange nodes …")
        self.batch_store(_MERGE_EXCHANGE, iter(active), size=len(active), desc="Exchanges")

        print("\n4. Verification …")
        with self._driver.session(database=self._database) as session:
            cnt = session.run("MATCH (ex:Exchange) RETURN count(ex) AS cnt").single()["cnt"]
            print(f"  Exchange nodes: {cnt}")
            for rec in session.run(
                "MATCH (ex:Exchange) RETURN ex.mic AS mic, ex.name AS name, ex.country AS country "
                "ORDER BY ex.name LIMIT 10"
            ):
                print(f"    {rec['mic']:<6s} {rec['country']}  {rec['name']}")

        print("\nDone.")


if __name__ == "__main__":
    importer = ExchangeImporter(argv=sys.argv[1:])
    try:
        importer.run()
    finally:
        importer.close()
